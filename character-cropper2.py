#!/usr/bin/env python3
"""
Qwen3.5 video Character Detector + Optional Square Crop
Uses reference images to locate a character in video clips
Outputs detection JSON with timestamped bounding boxes
Optionally crops video to a static SQUARE union crop centered on the character
"""

import os
import argparse
import gc
import json
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import cv2
from huggingface_hub import snapshot_download

# Set vLLM worker method
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Config
REPO_ID = "Qwen/Qwen3.5-27B-FP8"
LOCAL_MODEL_DIR = Path("/home/alex/Documents/qwen-vl-video-captioner/models/qwen3.5-27b-fp8")

# Global instances
llm = None
processor = None

def load_model_and_processor():
    global llm, processor
    if llm is None:
        print("🚀 Loading Qwen3-VL-30B-A3B-Instruct-FP8 locally...")

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
            gpu_memory_utilization=0.65,
            enforce_eager=True,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=42,
            max_model_len=32768,  # Increased as requested
            kv_cache_dtype="fp8",  # Uncomment for extra memory savings on H100+
        )
        print("✓ vLLM FP8 engine loaded from local directory!")
    return llm, processor

def build_prompt(num_refs: int, character_name: str, anime_title: Optional[str]) -> str:
    refs_part = f"The first {num_refs} images are reference images of the character \"{character_name}\""
    if anime_title:
        refs_part += f" from the anime \"{anime_title}\""
    refs_part += " in various styles, expressions, poses, angles, and deformations (including chibi and exaggerated forms)."

    target_part = "The following video clip is the target scene from the same series."

    instruction = (
        "Locate **every prominent instance** of this character in the video. "
        "For each distinct appearance, provide the **tightest possible** bounding box that closely fits the character's body/head (do not include large empty space around them). "
        "Be precise — boxes should hug the character, not the whole frame. "
        "Account for movement, anime stylization, and partial views, but **never** output full-frame boxes like [0.0, 0.0, 1.0, 1.0] unless the character truly fills the entire image."
    )

    format_guide = """Your response MUST be ONLY a valid JSON object. Start directly with '{' and end with '}', with no extra characters, whitespace, text, markdown, code blocks, or ``` fences whatsoever.

Example of correct output (do not copy this exactly; adapt to the video):
{
  "character_found": true,
  "overall_confidence": "high",
  "detections": [
    {
      "time_seconds": 0.5,
      "bbox_normalized": [0.35, 0.2, 0.65, 0.8],
      "confidence": "high",
      "reason": "Character centered with clear face and body in frame."
    },
    {
      "time_seconds": 1.2,
      "bbox_normalized": [0.1, 0.3, 0.4, 0.9],
      "confidence": "medium",
      "reason": "Partial side view with some occlusion."
    }
  ],
  "reason": "Character appears multiple times with consistent features."
}

CRITICAL RULES (follow exactly or output will be invalid):
- "character_found": boolean (true or false).
- "overall_confidence": string ("high", "medium", or "low").
- "detections": array of objects, sorted ascending by "time_seconds". Output at least 10–20 if the character appears frequently; empty array [] if none.
- For each detection:
  - "time_seconds": float (e.g., 0.5, 2.3).
  - "bbox_normalized": array of 4 floats strictly between 0.0 and 1.0 [x_min, y_min, x_max, y_max] — NEVER use integers or pixel values; always normalize by dividing by width/height.
  - "confidence": string ("high", "medium", or "low").
  - "reason": string (brief one-sentence explanation, no quotes inside).
- "reason": string (overall brief explanation).
- Make bboxes as tight as possible: hug the character's head + body; center horizontally on face/body; avoid near-full-frame like [0.0, 0.0, 0.99, 1.0] unless truly screen-filling.
- Ensure valid JSON: proper commas, no trailing commas in arrays/objects, escaped quotes if needed.
- ABSOLUTELY NO markdown, ```json, extra text, or explanations outside the JSON. Response starts with '{' and ends with '}'."""

    return f"{refs_part}\n\n{target_part}\n\n{instruction}\n\n{format_guide}"

def load_and_resize_image(path: Path, max_size: int) -> Any:
    if max_size <= 0:
        return str(path.absolute())

    try:
        img = Image.open(path).convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            print(f"  Resized reference {path.name} → {new_size}")
        return img
    except Exception as e:
        print(f"  ✗ Failed to load reference {path.name}: {e}")
        return None

def validate_video(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    valid = cap.isOpened()
    cap.release()
    return valid

def prepare_inputs_for_vllm(messages: list, processor) -> Optional[dict]:
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        else:
            print("  ⚠ No video_inputs detected")
            return None

        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
    except Exception as e:
        print(f"  ✗ prepare_inputs failed: {e}")
        return None

def normalize_bbox(bbox):
    """Normalize a single bbox if it looks like pixel coordinates"""
    if len(bbox) != 4:
        return None
    coords = [float(c) for c in bbox]
    if max(coords) > 1.1 or any(c > 1.0 for c in coords):
        # Assume pixel coords; normalize assuming max ~ width/height ≈ 1000
        # But to be safe, use the largest coord as reference
        ref = max(coords)
        if ref <= 0:
            return None
        return [max(0.0, min(1.0, c / ref)) for c in coords]
    return [max(0.0, min(1.0, float(c))) for c in bbox]  # already normalized, just clamp

def compute_square_crop_bbox(detections: List[Dict]) -> Optional[List[float]]:
    """
    Full-height square crop, horizontally centered on average character x-center.
    - Ignores y entirely (crop y=0.0 to 1.0)
    - Width = height → square
    - Shifts horizontally to place average center at 0.5
    """
    centers_x = []
    for d in detections:
        bbox = d.get('bbox_normalized')
        if not (bbox and len(bbox) == 4):
            continue

        # Quick per-bbox normalization / clamping
        try:
            x_min, y_min, x_max, y_max = map(float, bbox)
            if max(x_min, y_min, x_max, y_max) > 1.1:  # looks like pixels
                ref = max(x_max, y_max)  # rough scale
                if ref > 0:
                    x_min /= ref
                    x_max /= ref
            # Clamp
            x_min = max(0.0, min(1.0, x_min))
            x_max = max(0.0, min(1.0, x_max))

            if x_max > x_min:
                center_x = (x_min + x_max) / 2
                weight = x_max - x_min  # wider boxes = more reliable/important
                centers_x.append((center_x, weight))
        except:
            continue  # skip bad entry

    if not centers_x:
        print("  No valid x-centers → fallback to frame center")
        return [0.25, 0.0, 0.75, 1.0]  # mild center crop as safe default

    # Weighted average center (wider boxes count more)
    total_weight = sum(w for _, w in centers_x)
    avg_center_x = sum(cx * w for cx, w in centers_x) / total_weight

    print(f"  Average character x-center: {avg_center_x:.3f} (from {len(centers_x)} detections)")

    # Full-height square
    side = 1.0
    half = side / 2.0

    crop_x_min = avg_center_x - half
    crop_x_max = avg_center_x + half

    # Shift to stay in bounds
    if crop_x_min < 0:
        crop_x_max += -crop_x_min
        crop_x_min = 0.0
    if crop_x_max > 1.0:
        crop_x_min -= crop_x_max - 1.0
        crop_x_max = 1.0

    # If still almost full width → force a bit of centering/zoom
    final_width = crop_x_max - crop_x_min

    if final_width > 0.98:
        # Shrink slightly and re-center
        shrink_to = 0.85  # adjust: smaller = more zoom
        extra = (1.0 - shrink_to) / 2
        crop_x_min = extra
        crop_x_max = 1.0 - extra
        print(f"  Near-full width → forced shrink to {shrink_to:.2f} width")

    crop_bbox = [crop_x_min, 0.0, crop_x_max, 1.0]
    print(f"  Final crop bbox: {crop_bbox} (square, full height, horizontal shift only)")

    return crop_bbox

def crop_video(input_path: Path, output_path: Path, bbox: List[float]):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("  ✗ Failed to open video for cropping")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # bbox is [x_min_norm, 0.0, x_max_norm, 1.0]
    x1 = max(0, int(bbox[0] * width))
    x2 = min(width, int(bbox[2] * width))
    y1 = 0
    y2 = height

    crop_w = x2 - x1
    crop_h = y2 - y1  # = height

    print(f"  Horizontal crop: x {x1}→{x2} ({crop_w}px), full height {crop_h}px")

    if crop_w <= 0 or crop_h <= 0:
        print("  ✗ Invalid crop dimensions")
        cap.release()
        return False

    # Force square: use min(crop_w, crop_h) as side length
    side = min(crop_w, crop_h)

    # Re-center the square horizontally within the horizontal crop
    extra_left = (crop_w - side) // 2 if crop_w > side else 0
    x1 += extra_left
    x2 = x1 + side

    print(f"  Forced square crop: {side}x{side} px, from x {x1}→{x2}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (side, side))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y1:y2, x1:x2]
        # If cropped shape != (side, side), pad with black (rare here)
        if cropped.shape[1] != side or cropped.shape[0] != side:
            padded = np.zeros((side, side, 3), dtype=np.uint8)
            padded[:cropped.shape[0], :cropped.shape[1]] = cropped
            cropped = padded
        out.write(cropped)
        frame_count += 1

    cap.release()
    out.release()
    print(f"  ✓ Cropped {frame_count} frames to square {side}x{side}")
    return True

def process_single_video(video_path: Path, ref_images: List[Any], prompt: str, processor, debug: bool = False) -> Optional[Dict]:
    if not validate_video(video_path):
        print("  ✗ Invalid video: {video_path.name}")
        return None

    abs_video = str(video_path.absolute())

    valid_refs = [r for r in ref_images if r is not None]
    if len(valid_refs) == 0:
        print("  ✗ No valid reference images")
        return None

    content = []
    for ref in valid_refs:
        content.append({"type": "image", "image": ref})
    content.append({"type": "video", "video": abs_video})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    inputs = [prepare_inputs_for_vllm(messages, processor)]
    if not inputs or inputs[0] is None:
        return None

    try:
        sampling_params = SamplingParams(
            temperature=0.1,      # Lower for more consistent JSON
            max_tokens=4096,      # Increased to prevent cutoff
            top_p=0.7,
            repetition_penalty=1.0,
        )

        outputs = llm.generate(inputs, sampling_params)
        raw_text = outputs[0].outputs[0].text.strip()

        if debug:
            print(f"  Raw model output: {raw_text}")

        try:
            # Strip common markdown fences and whitespace
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:].strip()          # remove ```json
            if cleaned.startswith("```"):
                cleaned = cleaned[3:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            # Optional: find first { and last } to extract inner JSON if there's junk
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start != -1 and end != -1:
                cleaned = cleaned[start:end]

            result = json.loads(cleaned)
            print("  ✓ JSON parsed successfully (after cleaning)")

        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parsing failed: {e}")
            print(f"  Raw output snippet: {raw_text[:300]}...")  # for debugging
            result = {"raw_output": raw_text, "parse_error": True}

        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"  ✗ vLLM generation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Video Character Detector + Square Crop")
    parser.add_argument("--input-dir", type=Path, default=Path("input"), help="Directory containing input videos")
    parser.add_argument("--ref-dir", type=Path, required=True, help="Directory containing reference images of the character")
    parser.add_argument("--character-name", type=str, required=True, help="Name of the character")
    parser.add_argument("--anime-title", type=str, default="", help="Optional anime/series title")
    parser.add_argument("--ref-max-size", type=int, default=1024, help="Max side length for reference images (0 = no resizing)")
    parser.add_argument("--output-dir", type=Path, default=Path("detections"))
    parser.add_argument("--crop", action="store_true", help="Crop videos to static square focused on character (no audio)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        llm, processor = load_model_and_processor()

        # Load reference images
        ref_extensions = {"*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"}
        ref_paths = []
        for ext in ref_extensions:
            ref_paths.extend(args.ref_dir.glob(ext))
            ref_paths.extend(args.ref_dir.glob(ext.upper()))
        ref_paths = sorted(list(set(ref_paths)))

        if not ref_paths:
            print(f"✗ No reference images found in {args.ref_dir}")
            return

        if len(ref_paths) > 10:
            print(f"⚠ Using only first 10 refs (out of {len(ref_paths)})")
            ref_paths = ref_paths[:10]

        print(f"✓ Loading {len(ref_paths)} reference images for \"{args.character_name}\"")
        ref_images = [load_and_resize_image(p, args.ref_max_size) for p in ref_paths]

        prompt = build_prompt(len(ref_paths), args.character_name, args.anime_title if args.anime_title else None)
        if args.debug:
            print(f"Prompt:\n{prompt}")

        args.input_dir.mkdir(exist_ok=True)
        args.output_dir.mkdir(exist_ok=True)
        if args.crop:
            cropped_dir = args.output_dir / "cropped"
            cropped_dir.mkdir(exist_ok=True)

        video_extensions = {"*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"}
        video_files = []
        for ext in video_extensions:
            video_files.extend(args.input_dir.glob(ext))
            video_files.extend(args.input_dir.glob(ext.upper()))
        video_files = sorted(list(set(p for p in video_files if validate_video(p))))

        if args.limit:
            video_files = video_files[:args.limit]

        if not video_files:
            print(f"No valid videos in {args.input_dir}")
            return

        print(f"Found {len(video_files)} videos")

        success_count = 0
        for i, video in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing {video.name}")
            result = process_single_video(video, ref_images, prompt, processor, args.debug)

            if result:
                json_path = args.output_dir / f"{video.stem}_detection.json"
                if "parse_error" in result:
                    json_path.with_suffix(".txt").write_text(result.get("raw_output", ""))
                else:
                    detections = result.get("detections", [])
                    crop_bbox = compute_square_crop_bbox(detections) if detections else None
                    result["computed_square_crop_bbox"] = crop_bbox

                    json_path.write_text(json.dumps(result, indent=2))
                    success_count += 1

                    found = result.get("character_found", False)
                    conf = result.get("overall_confidence", "N/A")
                    num_dets = len(detections)
                    print(f"  ✓ Detection saved ({'Found' if found else 'Not found'}, {conf}, {num_dets} detections)")

                    if args.crop and found and crop_bbox:
                        cropped_path = args.output_dir / "cropped" / f"{video.stem}_cropped_square.mp4"
                        if crop_video(video, cropped_path, crop_bbox):
                            crop_size = int((crop_bbox[2] - crop_bbox[0]) * 100)  # rough % for log
                            print(f"  ✓ Square cropped video saved ({crop_size}% side): {cropped_path.name}")
                        else:
                            print(f"  ✗ Failed to crop (invalid bbox)")
            else:
                print(f"  ✗ Failed")

        print(f"\n🎉 Complete! {success_count}/{len(video_files)} successful detections")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
