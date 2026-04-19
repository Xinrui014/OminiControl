#!/usr/bin/env python3
"""
v3.1 inference — save at 1024 (no downscale).
Minimal fork of infer_full_eval.py for sharpness comparison.
"""
import json
import argparse
import random as _random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from omini.pipeline.flux_omini import Condition, generate, seed_everything
from lib.component_bank_v2 import ComponentBankV2, CAT_ID_TO_NAME
from lib.tiled_harmonize import load_model

PROMPT_TEMPLATES = {
    "green":  "A high-quality photograph of a printed circuit board with green soldermask, copper traces, and electronic components",
    "red":    "A high-quality photograph of a printed circuit board with red soldermask, copper traces, and electronic components",
    "blue":   "A high-quality photograph of a printed circuit board with blue soldermask, copper traces, and electronic components",
    "black":  "A high-quality photograph of a printed circuit board with black soldermask, copper traces, and electronic components",
    "yellow": "A high-quality photograph of a printed circuit board with yellow soldermask, copper traces, and electronic components",
}

BBOX_COLORS = {
    "RESISTOR": (255, 107, 107), "CAPACITOR": (78, 205, 196),
    "INDUCTOR": (69, 183, 209), "CONNECTOR": (150, 206, 180),
    "DIODE": (255, 234, 167), "LED": (221, 160, 221),
    "SWITCH": (240, 230, 140), "TRANSISTOR": (255, 179, 71),
    "IC": (135, 206, 235), "OSCILLATOR": (255, 160, 122),
    "FUSE": (200, 200, 200),
}

GEN_SIZE = 1024


def paste_patch_composite(annotations, bank, board_color, crop_size=512, top_k=10):
    img = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
    placed = 0
    for ann in annotations:
        cat = ann["category_name"]
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            continue
        match = bank.find_match(
            category=cat, target_w=w, target_h=h,
            board_color=board_color, top_k=top_k,
        )
        if match is None:
            continue
        crop = bank.load_crop(match, w, h, resize_jitter=0.0)
        if crop is None:
            continue
        px = max(0, min(x, crop_size - w))
        py = max(0, min(y, crop_size - h))
        img.paste(crop, (px, py))
        placed += 1
    return img, placed


def draw_patch_bboxes(annotations, crop_size=512):
    img = Image.new("RGB", (crop_size, crop_size), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for ann in annotations:
        cat = ann["category_name"]
        x, y, w, h = ann["bbox"]
        color = BBOX_COLORS.get(cat, (128, 128, 128))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((int(x) + 2, int(y) + 2), cat[:3], fill=color)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--anno_dir", default="/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/train")
    parser.add_argument("--image_dir", default="/home/xinrui/projects/data/ti_pcb/images_top")
    parser.add_argument("--board_image_dir", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/test")
    parser.add_argument("--omini_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eval_json) as f:
        eval_patches = json.load(f)
    total = len(eval_patches)
    if args.end > 0:
        eval_patches = eval_patches[args.start:args.end]
    elif args.start > 0:
        eval_patches = eval_patches[args.start:]
    print(f"Loaded {len(eval_patches)} patches [{args.start}:{args.end}] (total: {total})")

    print("\nBuilding ComponentBankV2...")
    bank = ComponentBankV2(
        anno_dir=args.anno_dir,
        image_dir=args.image_dir,
        edge_margin=5,
    )

    print(f"\nLoading model: {args.omini_ckpt}")
    pipe = load_model(args.omini_ckpt, device=args.device)

    board_images = {}
    results = []

    for idx, patch in enumerate(eval_patches):
        patch_id = patch["patch_id"]
        board = patch["board"]
        board_color = patch["board_color"]
        cx = patch["crop_x"]
        cy = patch["crop_y"]
        crop_size = patch["crop_size"]
        annotations = patch["annotations"]
        prompt = PROMPT_TEMPLATES.get(board_color, PROMPT_TEMPLATES["green"])

        print(f"\n[{idx+1}/{len(eval_patches)}] {patch_id} ({len(annotations)} components)")

        if board not in board_images:
            board_path = Path(args.board_image_dir) / f"{board}.png"
            if board_path.exists():
                board_images[board] = Image.open(board_path).convert("RGB")
            else:
                board_images[board] = None

        board_img = board_images[board]
        orig_patch = None
        if board_img is not None:
            orig_patch = board_img.crop((cx, cy, cx + crop_size, cy + crop_size))

        abs_idx = args.start + idx
        _random.seed(args.seed + abs_idx)

        composite_512, placed = paste_patch_composite(annotations, bank, board_color, crop_size, args.top_k)
        print(f"  Placed: {placed}/{len(annotations)}")

        composite_up = composite_512.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)
        condition = Condition(composite_up, "pcb_harmonize")
        seed_everything(args.seed + abs_idx)
        harmonized = generate(
            pipe, prompt=prompt, conditions=[condition],
            height=GEN_SIZE, width=GEN_SIZE,
        ).images[0]
        # NO downscale — keep at 1024

        bbox_img = draw_patch_bboxes(annotations, crop_size)

        # Upscale other panels to 1024 so 4-panel aligns
        composite_1024 = composite_up  # already 1024
        bbox_1024 = bbox_img.resize((GEN_SIZE, GEN_SIZE), Image.NEAREST)
        orig_1024 = orig_patch.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS) if orig_patch is not None else None

        panel = Image.new("RGB", (GEN_SIZE * 4, GEN_SIZE))
        panel.paste(bbox_1024, (0, 0))
        panel.paste(composite_1024, (GEN_SIZE, 0))
        panel.paste(harmonized, (GEN_SIZE * 2, 0))
        if orig_1024 is not None:
            panel.paste(orig_1024, (GEN_SIZE * 3, 0))

        panel.save(output_dir / f"{patch_id}_comparison.png")
        harmonized.save(output_dir / f"{patch_id}_harmonized.png")
        print(f"  -> {patch_id}_comparison.png (1024)")

        results.append({
            "patch_id": patch_id, "board": board, "color": board_color,
            "crop": [cx, cy], "n_components": len(annotations), "n_placed": placed,
        })

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! {len(results)} patches -> {output_dir}")


if __name__ == "__main__":
    main()
