#!/usr/bin/env python3
"""
Extract real 512x512 crops from test boards at the same positions as eval patches.
No resizing — direct crop from board images.

Usage:
    python eval_metrics/extract_real_crops.py \
        --eval_json /home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/eval_patches_full_test.json \
        --board_image_dir /home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/test \
        --output_dir eval_metrics/real_crops
"""
import json
import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/eval_patches_full_test.json")
    parser.add_argument("--board_image_dir", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/test")
    parser.add_argument("--output_dir", default="eval_metrics/real_crops")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eval_json) as f:
        eval_patches = json.load(f)
    print(f"Loaded {len(eval_patches)} patches")

    board_images = {}
    saved = 0
    skipped = 0

    for idx, patch in enumerate(eval_patches):
        patch_id = patch["patch_id"]
        board = patch["board"]
        cx = patch["crop_x"]
        cy = patch["crop_y"]
        crop_size = patch["crop_size"]

        # Load board image (cached)
        if board not in board_images:
            board_path = Path(args.board_image_dir) / f"{board}.png"
            if board_path.exists():
                board_images[board] = Image.open(board_path).convert("RGB")
            else:
                board_images[board] = None

        board_img = board_images[board]
        if board_img is None:
            skipped += 1
            continue

        crop = board_img.crop((cx, cy, cx + crop_size, cy + crop_size))
        crop.save(output_dir / f"{patch_id}.png")
        saved += 1

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{len(eval_patches)}] saved {saved}")

    print(f"\nDone! Saved {saved}, skipped {skipped} -> {output_dir}")


if __name__ == "__main__":
    main()
