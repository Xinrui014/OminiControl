#!/usr/bin/env python3
"""
PCB v2 Pipeline Inference: Test Layout Pasting + Tiled Harmonization (v2 12k ckpt)

Uses GROUND TRUTH layouts from test.jsonl (v2 format: 1280×720 boards).
Parses v2 tokens → pastes components via ComponentBankV2 → tiled harmonization.

Output: 3-panel comparison [Composite | Harmonized | Original]
"""
import os
import sys
import json
import re
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# OminiControl imports
OMINI_ROOT = "/home/xinrui/projects/OminiControl"
sys.path.insert(0, OMINI_ROOT)
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from component_bank_v2 import ComponentBankV2, CAT_ID_TO_NAME, CAT_NAME_TO_ID

# Reuse tiling logic
from tiled_harmonize import (
    compute_tile_grid, stitch_tiles, load_model, tiled_harmonize
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOARD_W, BOARD_H = 1280, 720

CATEGORIES = set(CAT_ID_TO_NAME.values())

COLOR_MAP = {
    "COLOR_GREEN": "green", "COLOR_RED": "red", "COLOR_BLUE": "blue",
    "COLOR_BLACK": "black", "COLOR_YELLOW": "yellow",
}

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

# ---------------------------------------------------------------------------
# v2 Layout Parser
# ---------------------------------------------------------------------------
def parse_v2_layout(text: str):
    """
    Parse v2 layout tokens: [BOS] [COLOR_x] [Rx] [CAT] [x] [y] [w] [h] ... [EOS]
    Returns: (board_color, components_list)
    where components_list = [(category, x, y, w, h), ...]
    """
    tokens = re.findall(r"\[(\w+)\]", text)

    board_color = "green"  # default
    components = []

    i = 0
    while i < len(tokens):
        t = tokens[i]

        # Parse color token
        if t in COLOR_MAP:
            board_color = COLOR_MAP[t]
            i += 1
            continue

        # Skip BOS/EOS/resolution tokens
        if t in ("BOS", "EOS") or t.startswith("R") and len(t) <= 2 and t[1:].isdigit():
            i += 1
            continue

        # Parse component: [CAT] [x] [y] [w] [h]
        if t in CATEGORIES and i + 4 < len(tokens):
            try:
                x = int(tokens[i + 1])
                y = int(tokens[i + 2])
                w = int(tokens[i + 3])
                h = int(tokens[i + 4])
                if w > 0 and h > 0:
                    components.append((t, x, y, w, h))
                i += 5
                continue
            except ValueError:
                pass

        i += 1

    return board_color, components


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def draw_bboxes_v2(components, board_color="green", width=BOARD_W, height=BOARD_H):
    """Draw bboxes on a dark background."""
    bg_colors = {
        "green": (20, 60, 20), "red": (60, 20, 20), "blue": (20, 20, 60),
        "black": (30, 30, 30), "yellow": (60, 60, 20),
    }
    img = Image.new("RGB", (width, height), bg_colors.get(board_color, (30, 30, 30)))
    draw = ImageDraw.Draw(img)
    for cat, x, y, w, h in components:
        color = BBOX_COLORS.get(cat, (128, 128, 128))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        label = cat[:3]
        draw.text((x + 2, y + 2), label, fill=color)
    return img


def paste_components_v2(components, bank: ComponentBankV2, board_color="green",
                        width=BOARD_W, height=BOARD_H, top_k=10):
    """Paste matched components onto a white canvas at 1280×720."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    placed = 0
    for cat, x, y, w, h in components:
        match = bank.find_match(
            category=cat, target_w=w, target_h=h,
            board_color=board_color, top_k=top_k,
        )
        if match is None:
            continue
        crop = bank.load_crop(match, int(w), int(h), resize_jitter=0.0)
        if crop is None:
            continue
        px = max(0, min(x, width - w))
        py = max(0, min(y, height - h))
        img.paste(crop, (int(px), int(py)))
        placed += 1
    return img, placed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PCB v2 Pipeline Inference (GT layout → paste → harmonize)")
    parser.add_argument("--test_jsonl", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/test.jsonl")
    parser.add_argument("--anno_dir", default="/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/train")
    parser.add_argument("--image_dir", default="/home/xinrui/projects/data/ti_pcb/images_top")
    parser.add_argument("--train_jsonl", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/train.jsonl")
    parser.add_argument("--omini_ckpt", default="/home/xinrui/projects/OminiControl/runs/20260327-022340/ckpt/12000")
    parser.add_argument("--output_dir", default="./pipeline_v2_output")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--min_overlap", type=int, default=128)
    parser.add_argument("--skip_harmonize", action="store_true", help="Only do pasting, skip harmonization")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test data ──────────────────────────────────────────────────────
    print(f"Loading test data from {args.test_jsonl}...")
    with open(args.test_jsonl) as f:
        test_data = [json.loads(l) for l in f]

    if args.num_samples == -1:
        samples = test_data
    else:
        random.shuffle(test_data)
        samples = test_data[:args.num_samples]
    print(f"Selected {len(samples)} test samples")

    # ── Build component bank ────────────────────────────────────────────────
    print("\nBuilding ComponentBankV2...")
    bank = ComponentBankV2(
        anno_dir=args.anno_dir,
        image_dir=args.image_dir,
        v2_jsonl=args.train_jsonl,
        edge_margin=5,
    )

    # ── Load harmonization model ────────────────────────────────────────────
    if not args.skip_harmonize:
        print(f"\nLoading harmonization model from {args.omini_ckpt}...")
        pipe = load_model(args.omini_ckpt, device=args.device)
    else:
        pipe = None

    # ── Process samples ─────────────────────────────────────────────────────
    results = []
    for i, sample in enumerate(samples):
        meta = sample.get("_meta", {})
        img_name = meta.get("image", f"sample_{i:04d}")
        gt_text = sample["messages"][2]["content"]
        user_prompt = sample["messages"][1]["content"]

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(samples)}] {img_name}")
        print(f"  Prompt: {user_prompt[:100]}")

        # Parse v2 layout
        board_color, components = parse_v2_layout(gt_text)
        print(f"  Color: {board_color} | Components: {len(components)}")

        if not components:
            print("  ⚠ No components parsed, skipping")
            continue

        # Draw bbox visualization
        bbox_img = draw_bboxes_v2(components, board_color)

        # Paste components
        composite, placed = paste_components_v2(
            components, bank, board_color=board_color, top_k=args.top_k
        )
        print(f"  Pasted: {placed}/{len(components)} components")

        # Harmonize
        if pipe is not None:
            prompt = PROMPT_TEMPLATES.get(board_color, PROMPT_TEMPLATES["green"])
            harmonized = tiled_harmonize(
                pipe, composite, prompt,
                tile_size=args.tile_size,
                min_overlap=args.min_overlap,
                seed=args.seed + i,
            )
        else:
            harmonized = None

        # Load original board for reference
        orig_path = Path(args.image_dir) / f"{img_name}.png"
        if orig_path.exists():
            orig_img = Image.open(orig_path).convert("RGB")
        else:
            orig_img = Image.new("RGB", (BOARD_W, BOARD_H), (30, 30, 30))

        # Build comparison panel
        if harmonized is not None:
            # 4-panel: bbox | composite | harmonized | original
            panel = Image.new("RGB", (BOARD_W * 4, BOARD_H))
            panel.paste(bbox_img, (0, 0))
            panel.paste(composite, (BOARD_W, 0))
            panel.paste(harmonized, (BOARD_W * 2, 0))
            panel.paste(orig_img, (BOARD_W * 3, 0))
        else:
            # 3-panel: bbox | composite | original
            panel = Image.new("RGB", (BOARD_W * 3, BOARD_H))
            panel.paste(bbox_img, (0, 0))
            panel.paste(composite, (BOARD_W, 0))
            panel.paste(orig_img, (BOARD_W * 2, 0))

        panel.save(output_dir / f"{img_name}_comparison.png")
        composite.save(output_dir / f"{img_name}_composite.png")
        if harmonized is not None:
            harmonized.save(output_dir / f"{img_name}_harmonized.png")
        print(f"  ✓ Saved: {img_name}_comparison.png")

        results.append({
            "name": img_name,
            "color": board_color,
            "prompt": user_prompt,
            "num_components": len(components),
            "num_placed": placed,
        })

    # ── Build gallery ───────────────────────────────────────────────────────
    has_harmonized = pipe is not None
    panel_label = "bbox | composite | harmonized | original" if has_harmonized else "bbox | composite | original"
    rows = []
    for r in results:
        rows.append(f"""
        <div class="sample">
          <h3>{r['name']}</h3>
          <p class="meta">Color: {r['color']} | Components: {r['num_placed']}/{r['num_components']} placed</p>
          <p class="prompt">{r['prompt']}</p>
          <img src="{r['name']}_comparison.png" style="width:100%;border-radius:8px;margin-top:8px">
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PCB v2 Pipeline</title>
<style>
body{{background:#1a1a2e;color:#eee;font-family:sans-serif;max-width:1800px;margin:auto;padding:20px}}
h1{{color:#4ECDC4;text-align:center}}
.sample{{background:#16213e;border-radius:12px;padding:16px;margin:16px auto}}
h3{{color:#FF6B6B;margin:0 0 4px}}
.meta{{color:#4ECDC4;font-size:13px;margin:4px 0}}
.prompt{{color:#888;font-size:13px;margin:4px 0}}
</style></head><body>
<h1>PCB v2 Pipeline — GT Layout → Paste → Harmonize (v2 ckpt 12k)</h1>
<p style="text-align:center;color:#aaa;font-size:13px">Panel: {panel_label}</p>
{"".join(rows)}
</body></html>"""
    (output_dir / "gallery.html").write_text(html)
    print(f"\n✅ Done! {len(results)} samples → {output_dir}/gallery.html")


if __name__ == "__main__":
    main()
