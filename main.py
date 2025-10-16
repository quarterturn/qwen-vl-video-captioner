#!/usr/bin/env python3
"""
Official vLLM + Qwen3-VL FP8 Video Captioner
Adapted from HuggingFace example for local MP4 files
"""

import os
import argparse
import atexit
import gc
import torch
from pathlib import Path
from typing import List, Optional
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Set vLLM worker method
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Config
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

# Global instances
llm = None
processor = None

def load_model_and_processor():
    global llm, processor
    if llm is None:
        print("🚀 Loading Qwen3-VL-30B-A3B-Instruct-FP8 with official vLLM method...")
        
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=False,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=42,
            max_model_len=8192,
        )
        print("✓ vLLM FP8 engine loaded!")
    return llm, processor

def load_prompt() -> str:
    prompt_file = Path("prompt.txt")
    if not prompt_file.exists():
        default_prompt = "Analyze this 15-second video clip and generate a detailed caption."
        prompt_file.write_text(default_prompt)
        print("📝 Created default prompt.txt")
        return default_prompt
    return prompt_file.read_text().strip()

def validate_video(video_path: Path) -> bool:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    valid = cap.isOpened()
    cap.release()
    return valid

def prepare_inputs_for_vllm(messages: list, processor) -> Optional[dict]:
    """Official vLLM input preparation"""
    try:
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # OFFICIAL: Use return_video_metadata=True
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,  # Critical for local videos
        )
        
        print(f"  video_kwargs: {video_kwargs}")
        
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        else:
            print("  ⚠ No video_inputs - check file format")
            return None
        
        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
    except Exception as e:
        print(f"  ✗ prepare_inputs failed: {e}")
        return None

def process_single_video(video_path: Path, prompt: str, processor, debug: bool = False) -> Optional[str]:
    """Process single video using official vLLM method"""
    if not validate_video(video_path):
        print(f"  ✗ Invalid video: {video_path.name}")
        return None
    
    abs_path = video_path.absolute()
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(abs_path)},  # Local absolute path
            {"type": "text", "text": prompt},
        ]
    }]
    
    inputs = [prepare_inputs_for_vllm(messages, processor)]
    if not inputs or inputs[0] is None:
        return None
    
    try:
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            top_k=64,
            repetition_penalty=1.05,
        )
        
        outputs = llm.generate(inputs, sampling_params)
        caption = outputs[0].outputs[0].text.strip()
        
        if debug:
            print(f"  ✓ Generated: {len(caption.split())} words")
            print(f"  Preview: {caption[:100]}...")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return caption
        
    except Exception as e:
        print(f"  ✗ vLLM generation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Official vLLM Video Captioner")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    try:
        llm, processor = load_model_and_processor()
        prompt = load_prompt()
        
        args.input_dir.mkdir(exist_ok=True)
        args.output_dir.mkdir(exist_ok=True)
        
        video_files = [p for p in args.input_dir.glob("*.mp4") if validate_video(p)]
        if args.limit:
            video_files = video_files[:args.limit]
        
        if not video_files:
            print(f"No valid videos in {args.input_dir}")
            return
        
        print(f"Found {len(video_files)} videos")
        
        success_count = 0
        for i, video in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] {video.name}")
            caption = process_single_video(video, prompt, processor, args.debug)
            
            if caption:
                output_file = args.output_dir / f"{video.stem}_caption.txt"
                output_file.write_text(caption)
                print(f"  ✓ Saved: {output_file.name}")
                success_count += 1
            else:
                print(f"  ✗ Failed")
        
        print(f"\n🎉 Complete! {success_count}/{len(video_files)} successful")
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()