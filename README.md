---
license: cc-by-nc-4.0
---

Qwen3-VL-30B-A3B-Instruct-FP8 video captioner

A simple, super-fast, accurate video clip captioner, aimed at creating a dataset of annotated videos for AI training and LoRA creation.
Tested with CUDA 12.9.

SUPER thanks to the Qwen team!
https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8

Requirements:
At least 40 GB GPU VRAM (sorry, 5090 owners), since in testing the script uses about 37 GB.
Ada or newer GPU (Ampere might work, didn't test).
Enough hard drive space to store the model.

Install:
1. clone the repo
2. create a conda or python env for the project
3. activate the env
4. install the dependencies via ```pip install -r requirements.txt```
   optional: uncomment flashinfer-python in requirements.txt or ```pip install flashinfer-python``` for greater performance
   optional: If you find the script slow to pull the model the first time, download it manually first using ```huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct-FP8```
5. put the videos to be captioned into 'input' or specify whatever you want with ```--input-dir``` and ```--output-dir```

Use:
```python3 main.py --input-dir input --output-dir output```


BONUS TOOL!
charcter-cropper.py
 
usage: character-cropper.py [-h] [--input-dir INPUT_DIR] --ref-dir REF_DIR --character-name CHARACTER_NAME [--anime-title ANIME_TITLE]
                            [--ref-max-size REF_MAX_SIZE] [--output-dir OUTPUT_DIR] [--crop] [--debug] [--limit LIMIT]
character-cropper.py: error: the following arguments are required: --ref-dir, --character-name


How it works:
This tool uses qwen3-vl to find a character in a video clip using images in REF_DIR. It slides a square bounding box around in the image and crops the video to where the characters face is, if it can find it. This is useful for making a dataset of a specific character.
