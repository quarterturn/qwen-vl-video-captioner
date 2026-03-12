#!/usr/bin/env python3
"""
Qwen3-VL-30B-A3B Video Character Top-4 Matches Extractor
Uses reference images to locate a single character in video clips
Saves detection JSON + up to 4 best-matching frames as PNGs
"""

import os
import argparse
import gc
import json
import re
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import cv2
from huggingface_hub import snapshot_download

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

REPO_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
LOCAL_MODEL_DIR = Path("models/qwen3-vl-fp8")

llm = None
processor = None


def load_model_and_processor():
    global llm, processor
    if llm is None:
        print("Loading Qwen3-VL-30B-A3B-Instruct-FP8...")
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(LOCAL_MODEL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.py", "*.json", "*.txt", "*.md", "*.tiktoken", "*.model", "LICENSE", "*.yml", "*.xml", "*.msgpack"],
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.h5", "*.onnx"],
        )
        model_path = str(LOCAL_MODEL_DIR.resolve())
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=False,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=42,
            max_model_len=32768,
        )
        print("Model loaded.")
    return llm, processor


def build_prompt(num_refs: int, character_name: str, anime_title: Optional[str]) -> str:
    refs = f"The first {num_refs} images show the character \"{character_name}\""
    if anime_title:
        refs += f" from \"{anime_title}\""
    refs += " in different styles, expressions, angles, etc."

    instruction = (
        "Find prominent appearances of this character in the video. "
        "Output **at most 10 detections**. Prefer clear, central, varied shots. "
        "Use tight bounding boxes around head+body. "
        "Avoid near-full-frame boxes unless the character fills the screen."
    )

    format_guide = """Respond **only** with valid JSON — no extra text, no markdown, no fences.

{
  "character_found": boolean,
  "overall_confidence": "high"|"medium"|"low",
  "detections": [
    {
      "time_seconds": float,
      "bbox_normalized": [x_min, y_min, x_max, y_max],
      "confidence": "high"|"medium"|"low",
      "reason": "short reason"
    },
    ...
  ],
  "reason": "overall explanation"
}

- detections sorted by time_seconds
- tight boxes only
- at most 10 detections
"""

    return f"{refs}\n\nThe following is a video clip from the same series.\n\n{instruction}\n\n{format_guide}"


def parse_or_repair_json(raw: str) -> Dict:
    cleaned = re.sub(r'^```json\s*|\s*```$', '', raw.strip())
    try:
        return json.loads(cleaned)
    except:
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        cleaned = re.sub(r'([{\[,])\s*([a-zA-Z_]\w*)\s*:', r'\1"\2":', cleaned)
        try:
            return json.loads(cleaned)
        except:
            Path("failed_raw.txt").write_text(raw)
            return {"parse_error": True, "raw": raw}


def load_and_resize_image(p: Path, max_size: int):
    if max_size <= 0:
        return str(p.absolute())
    try:
        img = Image.open(p).convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        return img
    except:
        return None


def validate_video(p: Path) -> bool:
    cap = cv2.VideoCapture(str(p))
    ok = cap.isOpened()
    cap.release()
    return ok


def prepare_inputs(messages, processor):
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs:
            mm_data['image'] = image_inputs
        if video_inputs:
            mm_data['video'] = video_inputs
        else:
            return None
        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
    except Exception as e:
        print(f"Input prep failed: {e}")
        return None


def save_frame(video_path: Path, time_sec: float, out_png: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return False
    frame_idx = int(time_sec * fps + 0.5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    try:
        cv2.imwrite(str(out_png), frame)
        return True
    except:
        return False


def process_video(video: Path, refs: List, prompt: str, processor, debug: bool, output_dir: Path):
    if not validate_video(video):
        print(f"Invalid video: {video.name}")
        return

    valid_refs = [r for r in refs if r is not None]
    if not valid_refs:
        print("No valid reference images")
        return

    content = [{"type": "image", "image": r} for r in valid_refs]
    content += [
        {"type": "video", "video": str(video.absolute())},
        {"type": "text", "text": prompt}
    ]
    messages = [{"role": "user", "content": content}]
    inputs = [prepare_inputs(messages, processor)]
    if not inputs or inputs[0] is None:
        return

    try:
        outputs = llm.generate(inputs, SamplingParams(
            temperature=0.1,
            max_tokens=2048,
            top_p=0.8,
            repetition_penalty=1.05,
        ))
        raw = outputs[0].outputs[0].text.strip()
        result = parse_or_repair_json(raw)

        if "parse_error" in result:
            print("JSON parse failed")
            return

        detections = result.get("detections", [])
        if not detections:
            print("No detections")
            json_path = output_dir / f"{video.stem}_detections.json"
            json_path.write_text(json.dumps(result, indent=2))
            return

        # Sort by confidence priority, then by time
        conf_score = {"high": 3, "medium": 2, "low": 1, None: 0}
        sorted_dets = sorted(
            detections,
            key=lambda d: (-conf_score.get(d.get("confidence"), 0), d.get("time_seconds", 9999))
        )

        top4 = sorted_dets[:4]

        # Save full JSON
        json_path = output_dir / f"{video.stem}_detections.json"
        json_path.write_text(json.dumps(result, indent=2))

        print(f"  Found {len(detections)} detections → saving top {len(top4)}")

        for i, det in enumerate(top4, 1):
            t = det.get("time_seconds", 0)
            conf = det.get("confidence", "unknown")
            png = output_dir / f"{video.stem}_match_{i}.png"
            if save_frame(video, t, png):
                print(f"  Saved match {i}: {conf} @ {t:.2f}s → {png.name}")
            else:
                print(f"  Failed to extract frame for match {i} @ {t:.2f}s")

    except Exception as e:
        print(f"Processing failed: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract top 4 best frames of a character from videos")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--character-name", type=str, required=True)
    parser.add_argument("--anime-title", type=str, default="")
    parser.add_argument("--ref-max-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("top4_matches"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_model_and_processor()

    # Load refs
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    ref_paths = []
    for e in exts:
        ref_paths.extend(args.ref_dir.glob(e))
        ref_paths.extend(args.ref_dir.glob(e.upper()))
    ref_paths = sorted(set(ref_paths))[:10]  # cap at 10

    if not ref_paths:
        print("No reference images found")
        return

    print(f"Loading {len(ref_paths)} reference images for {args.character_name}")
    refs = [load_and_resize_image(p, args.ref_max_size) for p in ref_paths]

    prompt = build_prompt(len(ref_paths), args.character_name, args.anime_title or None)

    args.output_dir.mkdir(exist_ok=True)

    videos = []
    for e in ["*.mp4", "*.mkv", "*.mov", "*.webm", "*.avi"]:
        videos.extend(args.input_dir.glob(e))
        videos.extend(args.input_dir.glob(e.upper()))
    videos = sorted(v for v in set(videos) if validate_video(v))

    if args.limit:
        videos = videos[:args.limit]

    if not videos:
        print("No videos found")
        return

    print(f"Processing {len(videos)} videos")

    count = 0
    for i, vid in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {vid.name}")
        process_video(vid, refs, prompt, processor, args.debug, args.output_dir)
        # Rough success check
        if any((args.output_dir / f"{vid.stem}_match_{k}.png").exists() for k in range(1, 5)):
            count += 1

    print(f"\nFinished — {count} videos had at least one match saved")


if __name__ == "__main__":
    main()
