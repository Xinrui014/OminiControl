#!/usr/bin/env python3
"""
Quick v2 patch-level harmonization test.
Takes a few test boards, crops 512×512 patches (with GT components),
builds composites, and runs harmonization at native 512×512.

Output: 3-panel per patch [Composite | Harmonized | Original patch]
"""
import os
import sys
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from omini.pipeline.flux_omini import Condition, generate, seed_everything
from lib.component_bank_v2 import ComponentBankV2, CAT_ID_TO_NAME, get_annotations_in_crop
from lib.tiled_harmonize import load_model

BOARD_W, BOARD_H = 1280, 720
CROP_SIZE = 512

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


def paste_patch_composite(annotations, bank, board_color, crop_size=512, top_k=10):
    """Create a composite for a single 512×512 patch from annotations."""
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
    parser.add_argument("--anno_dir", default="/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/train")
    parser.add_argument("--image_dir", default="/home/xinrui/projects/data/ti_pcb/images_top")
    parser.add_argument("--train_jsonl", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/train.jsonl")
    parser.add_argument("--test_jsonl", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/test.jsonl")
    parser.add_argument("--omini_ckpt", default="/home/xinrui/projects/OminiControl/runs/v2_pcb_harmonize/ckpt/12000")
    parser.add_argument("--output_dir", default="./patch_v2_output")
    parser.add_argument("--num_boards", type=int, default=5, help="Number of test boards")
    parser.add_argument("--patches_per_board", type=int, default=2, help="Random crops per board")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test metadata
    print("Loading test data...")
    with open(args.test_jsonl) as f:
        test_data = [json.loads(l) for l in f]
    random.shuffle(test_data)
    samples = test_data[:args.num_boards]

    # Build component bank (from training data only)
    print("\nBuilding ComponentBankV2...")
    bank = ComponentBankV2(
        anno_dir=args.anno_dir,
        image_dir=args.image_dir,
        
        edge_margin=5,
    )

    # Load model
    print(f"\nLoading harmonization model: {args.omini_ckpt}")
    pipe = load_model(args.omini_ckpt, device=args.device)

    results = []
    patch_idx = 0

    for si, sample in enumerate(samples):
        meta = sample.get("_meta", {})
        board_name = meta.get("image", f"board_{si}")
        board_color = meta.get("color", "green")

        # Load board image
        board_path = Path("/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/test") / f"{board_name}.png"
        if not board_path.exists():
            print(f"[{si+1}] {board_name}: image not found, skipping")
            continue
        board_img = Image.open(board_path).convert("RGB")
        bw, bh = board_img.size

        # Load annotations
        anno_path = Path(args.anno_dir.replace("annotation/train", "annotation/test")) / f"{board_name}.json"
        if not anno_path.exists():
            print(f"[{si+1}] {board_name}: annotations not found, skipping")
            continue
        with open(anno_path) as f:
            anno_data = json.load(f)
        all_annotations = anno_data.get("annotations", [])

        print(f"\n{'='*60}")
        print(f"[{si+1}/{len(samples)}] {board_name} ({board_color}, {bw}×{bh}, {len(all_annotations)} annotations)")

        prompt = PROMPT_TEMPLATES.get(board_color, PROMPT_TEMPLATES["green"])

        for pi in range(args.patches_per_board):
            # Random crop position
            max_x = max(0, bw - CROP_SIZE)
            max_y = max(0, bh - CROP_SIZE)
            cx = random.randint(0, max_x) if max_x > 0 else 0
            cy = random.randint(0, max_y) if max_y > 0 else 0

            # Get annotations in this crop
            patch_anns = get_annotations_in_crop(all_annotations, cx, cy, CROP_SIZE, min_visible_ratio=0.5)

            if len(patch_anns) < 2:
                print(f"  Patch {pi}: ({cx},{cy}) — only {len(patch_anns)} components, trying another position")
                # Try again with a different position
                for _ in range(10):
                    cx = random.randint(0, max_x) if max_x > 0 else 0
                    cy = random.randint(0, max_y) if max_y > 0 else 0
                    patch_anns = get_annotations_in_crop(all_annotations, cx, cy, CROP_SIZE, min_visible_ratio=0.5)
                    if len(patch_anns) >= 2:
                        break
                if len(patch_anns) < 2:
                    print(f"  Patch {pi}: gave up, skipping")
                    continue

            print(f"  Patch {pi}: crop({cx},{cy}) — {len(patch_anns)} components")

            # Original patch
            orig_patch = board_img.crop((cx, cy, cx + CROP_SIZE, cy + CROP_SIZE))

            # Bbox visualization
            bbox_img = draw_patch_bboxes(patch_anns)

            # Composite
            composite, placed = paste_patch_composite(patch_anns, bank, board_color, top_k=args.top_k)
            print(f"    Placed: {placed}/{len(patch_anns)}")

            # Harmonize (single 512×512 — no tiling!)
            condition = Condition(composite, "pcb_harmonize")
            seed_everything(args.seed + patch_idx)
            harmonized = generate(pipe, prompt=prompt, conditions=[condition]).images[0]

            # 4-panel: bbox | composite | harmonized | original
            panel = Image.new("RGB", (CROP_SIZE * 4, CROP_SIZE))
            panel.paste(bbox_img, (0, 0))
            panel.paste(composite, (CROP_SIZE, 0))
            panel.paste(harmonized, (CROP_SIZE * 2, 0))
            panel.paste(orig_patch, (CROP_SIZE * 3, 0))

            name = f"{board_name}_p{pi}_{cx}_{cy}"
            panel.save(output_dir / f"{name}_comparison.png")
            composite.save(output_dir / f"{name}_composite.png")
            harmonized.save(output_dir / f"{name}_harmonized.png")
            print(f"    ✓ {name}_comparison.png")

            results.append({
                "name": name, "board": board_name, "color": board_color,
                "crop": (cx, cy), "n_components": len(patch_anns), "n_placed": placed,
            })
            patch_idx += 1

    # Gallery
    rows = []
    for r in results:
        rows.append(f"""
        <div class="sample">
          <h3>{r['name']}</h3>
          <p class="meta">Board: {r['board']} | Color: {r['color']} | Crop: {r['crop']} | Components: {r['n_placed']}/{r['n_components']}</p>
          <img src="{r['name']}_comparison.png" style="width:100%;border-radius:8px;margin-top:8px">
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PCB v2 Patch-Level Test</title>
<style>
body{{background:#1a1a2e;color:#eee;font-family:sans-serif;max-width:1200px;margin:auto;padding:20px}}
h1{{color:#4ECDC4;text-align:center}}
.sample{{background:#16213e;border-radius:12px;padding:16px;margin:16px auto}}
h3{{color:#FF6B6B;margin:0 0 4px}}
.meta{{color:#4ECDC4;font-size:13px;margin:4px 0}}
</style></head><body>
<h1>PCB v2 — Patch-Level Harmonization (512×512, no tiling)</h1>
<p style="text-align:center;color:#aaa">Panels: bbox | composite | harmonized | original patch</p>
{"".join(rows)}
</body></html>"""
    (output_dir / "gallery.html").write_text(html)
    print(f"\n✅ Done! {len(results)} patches → {output_dir}/gallery.html")


if __name__ == "__main__":
    main()
