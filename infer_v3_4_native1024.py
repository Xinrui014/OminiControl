#!/usr/bin/env python3
"""
v3.4 inference pipeline B — native 1024 with PIL auto-black-pad, matching v3.4
training's non-zoom branch exactly.

v3.4 non-zoom training (60% of batches):
  - crop_size = 1024 on a 1280x720 board
  - cx in [0, 256], cy = 0 (degenerate range, always top-anchored)
  - PIL board.crop((cx, 0, cx+1024, 1024)) auto-fills the bottom 304 rows BLACK
  - composite canvas init = white, components pasted on it
  - self-fallback = board.crop(...) at full bbox (also auto-black-pads OOB)

This script mirrors that exactly:
  1. Load 1280x720 test board (no pre-padding).
  2. Place a 1024x1024 window at cy=0, cx chosen to center on the original 512
     eval patch's center, clamped to [0, 256].
  3. Crop real_patch and build composite at native 1024 (no upscale).
  4. Generate at 1024. Save 1024.

Usage:
    CUDA_VISIBLE_DEVICES=0 python infer_v3_4_native1024.py \
        --eval_json /projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json \
        --omini_ckpt runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000 \
        --output_dir runs/v3.4_resumed_from8k_1024/20260421-060156/eval/ckpt14k_native1024_sanity5 \
        --start 0 --end 5
"""
import json
import argparse
import random as _random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import FluxPipeline

from omini.pipeline.flux_omini import Condition, generate, seed_everything
from lib.component_bank_v2_1 import (
    ComponentBankV2_1, CAT_ID_TO_NAME, parse_resolution_class,
)

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
GEN_SIZE = 1024
ORIG_PATCH_SIZE = 512  # eval json patches are 512x512

PROMPT_TEMPLATES = {
    "green":  "A high-quality photograph of a printed circuit board with green soldermask, copper traces, and electronic components",
    "red":    "A high-quality photograph of a printed circuit board with red soldermask, copper traces, and electronic components",
    "blue":   "A high-quality photograph of a printed circuit board with blue soldermask, copper traces, and electronic components",
    "black":  "A high-quality photograph of a printed circuit board with black soldermask, copper traces, and electronic components",
    "white":  "A high-quality photograph of a printed circuit board with white soldermask, copper traces, and electronic components",
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


def load_model(checkpoint_path, device="cuda"):
    print(f"Loading FLUX.1-dev from {FLUX_PATH} ...")
    pipe = FluxPipeline.from_pretrained(
        FLUX_PATH, torch_dtype=torch.bfloat16,
    ).to(device)
    print(f"Loading LoRA: {checkpoint_path}")
    pipe.load_lora_weights(
        checkpoint_path,
        weight_name="default.safetensors",
        adapter_name="pcb_harmonize",
    )
    pipe.set_adapters(["pcb_harmonize"])
    print("Model ready.")
    return pipe


def load_board(board_image_dir, board_name):
    """Load 1280x720 board as RGB, no padding (matches v3.4 training input)."""
    board_path = Path(board_image_dir) / f"{board_name}.png"
    if not board_path.exists():
        return None
    img = Image.open(board_path)
    if img.mode in ("RGBA", "PA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.convert("RGBA").split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    return img


def compute_1024_window(patch_crop_x, patch_crop_y, img_w, img_h):
    """Place a 1024x1024 window on the 1280x720 board matching v3.4 training:
      - cy = 0 always (board is 720 tall, PIL auto-black-pads bottom 304 rows)
      - cx chosen to center on the original 512 eval patch; clamped to [0, 256]
    """
    patch_center_x = patch_crop_x + ORIG_PATCH_SIZE // 2
    win_x = patch_center_x - GEN_SIZE // 2
    max_x = max(0, img_w - GEN_SIZE)  # 1280 - 1024 = 256
    win_x = max(0, min(max_x, win_x))
    win_y = 0  # top-anchored; img_h=720 < GEN_SIZE so PIL will black-pad
    return win_x, win_y


def annotations_in_window(full_annos, win_x, win_y, win_size=1024,
                          min_visible_ratio=0.3):
    """Filter full-board annotations to those intersecting the 1024 window.
    Returns a list with crop-relative bbox + original board-space bbox."""
    out = []
    for ann in full_annos:
        cat_id = ann.get("category_id")
        if cat_id not in CAT_ID_TO_NAME:
            continue
        ax, ay, aw, ah = ann["bbox"]
        x1 = max(ax, win_x); y1 = max(ay, win_y)
        x2 = min(ax + aw, win_x + win_size); y2 = min(ay + ah, win_y + win_size)
        if x2 <= x1 or y2 <= y1:
            continue
        visible = ((x2 - x1) * (y2 - y1)) / max(1.0, aw * ah)
        if visible < min_visible_ratio:
            continue
        out.append({
            "category_id": cat_id,
            "category_name": CAT_ID_TO_NAME[cat_id],
            "bbox": (x1 - win_x, y1 - win_y, x2 - x1, y2 - y1),
            "original_bbox": (ax, ay, aw, ah),
            "orientation": ann.get("orientation") or 0,
            "sub_class": ann.get("sub_class") if ann.get("sub_class") is not None else -1,
            "visible_ratio": visible,
        })
    return out


def paste_composite_native(annos, bank, board_color, resolution_class,
                           exclude_board, board_img, crop_x, crop_y,
                           canvas_size=1024, top_k=10):
    """Build composite at native 1024, mirroring v3.4 _build_composite exactly:
      - canvas init white (255,255,255)
      - match on FULL original bbox (ow, oh)
      - load_crop at full size, then PIL-crop to visible portion
      - self-fallback = board_img.crop at absolute bbox (auto-black-pads OOB)
    """
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    placed = 0
    for ann in annos:
        cat = ann["category_name"]
        rx, ry, rw, rh = ann["bbox"]
        ox, oy, ow, oh = ann["original_bbox"]
        rw_int, rh_int = int(rw), int(rh)
        if rw_int < 3 or rh_int < 3:
            continue

        result = bank.find_match(
            category=cat, sub_class=ann.get("sub_class", -1),
            target_w=ow, target_h=oh,
            board_color=board_color, resolution_class=resolution_class,
            orientation=ann.get("orientation", 0),
            exclude_board=exclude_board, top_k=top_k,
        )

        crop_to_paste = None
        if result is not None:
            entry, rotation = result
            full_crop = bank.load_crop(
                entry, int(ow), int(oh),
                rotation=rotation, resize_jitter=0.0,
            )
            if full_crop is not None:
                fw, fh = full_crop.size
                off_x = int(rx - (ox - crop_x))
                off_y = int(ry - (oy - crop_y))
                off_x = max(0, min(off_x, max(fw - 1, 0)))
                off_y = max(0, min(off_y, max(fh - 1, 0)))
                vis_w = max(1, min(rw_int, fw - off_x))
                vis_h = max(1, min(rh_int, fh - off_y))
                crop_to_paste = full_crop.crop(
                    (off_x, off_y, off_x + vis_w, off_y + vis_h)
                )

        if crop_to_paste is None:
            abs_x = crop_x + int(rx)
            abs_y = crop_y + int(ry)
            crop_to_paste = board_img.crop(
                (abs_x, abs_y, abs_x + rw_int, abs_y + rh_int)
            )

        px = max(0, min(int(rx), canvas_size - crop_to_paste.width))
        py = max(0, min(int(ry), canvas_size - crop_to_paste.height))
        canvas.paste(crop_to_paste, (px, py))
        placed += 1
    return canvas, placed


def draw_bboxes_native(annos, size=GEN_SIZE):
    img = Image.new("RGB", (size, size), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for ann in annos:
        cat = ann["category_name"]
        x, y, w, h = ann["bbox"]
        color = BBOX_COLORS.get(cat, (128, 128, 128))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((int(x) + 2, int(y) + 2), cat[:3], fill=color)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--anno_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/annotation/train")
    parser.add_argument("--image_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/image/train")
    parser.add_argument("--board_image_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/image/test")
    parser.add_argument("--anno_test_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/annotation/test")
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

    print("\nBuilding ComponentBankV2.1...")
    bank = ComponentBankV2_1(
        anno_dir=args.anno_dir, image_dir=args.image_dir, edge_margin=5,
    )

    test_board_meta = {}
    test_board_annos = {}
    for anno_path in Path(args.anno_test_dir).glob("*.json"):
        with open(anno_path) as f:
            data = json.load(f)
        test_board_meta[anno_path.stem] = {
            "color": data.get("board_color") or "green",
            "resolution_class": parse_resolution_class(
                data.get("resolution_class") or "R3"
            ),
        }
        test_board_annos[anno_path.stem] = data.get("annotations", [])
    print(f"Loaded metadata + annotations for {len(test_board_meta)} test boards")

    print(f"\nLoading model: {args.omini_ckpt}")
    pipe = load_model(args.omini_ckpt, device=args.device)

    boards = {}
    results = []

    for idx, patch in enumerate(eval_patches):
        patch_id = patch["patch_id"]
        board = patch["board"]
        board_color = patch.get("board_color") or "green"
        cx_512 = patch["crop_x"]
        cy_512 = patch["crop_y"]

        if board not in boards:
            boards[board] = load_board(args.board_image_dir, board)
        board_img = boards[board]
        if board_img is None:
            print(f"[{idx+1}/{len(eval_patches)}] {patch_id} — SKIP (no board image)")
            continue
        img_w, img_h = board_img.size  # expected 1280 x 720

        win_x, win_y = compute_1024_window(cx_512, cy_512, img_w, img_h)

        # Real patch at native 1024 — PIL auto-black-pads bottom rows where y > img_h
        real_patch = board_img.crop((win_x, win_y, win_x + GEN_SIZE, win_y + GEN_SIZE))

        full_annos = test_board_annos.get(board, [])
        crop_annos = annotations_in_window(
            full_annos, win_x, win_y, win_size=GEN_SIZE, min_visible_ratio=0.3,
        )

        meta = test_board_meta.get(board, {})
        resolution_class = meta.get("resolution_class", "R3")
        if not board_color or board_color not in PROMPT_TEMPLATES:
            board_color = meta.get("color", "green")
        prompt = PROMPT_TEMPLATES.get(board_color) or PROMPT_TEMPLATES["green"]

        print(f"\n[{idx+1}/{len(eval_patches)}] {patch_id} "
              f"({len(crop_annos)} components, win=({win_x},{win_y}))")

        if (output_dir / f"{patch_id}_harmonized.png").exists():
            print(f"  SKIP (exists)")
            continue

        abs_idx = args.start + idx
        _random.seed(args.seed + abs_idx)

        composite, placed = paste_composite_native(
            crop_annos, bank, board_color, resolution_class,
            exclude_board=board, board_img=board_img,
            crop_x=win_x, crop_y=win_y,
            canvas_size=GEN_SIZE, top_k=args.top_k,
        )
        print(f"  Placed: {placed}/{len(crop_annos)}")

        condition = Condition(composite, "pcb_harmonize")
        seed_everything(args.seed + abs_idx)
        harmonized = generate(
            pipe, prompt=prompt, conditions=[condition],
            height=GEN_SIZE, width=GEN_SIZE,
        ).images[0]

        bbox_img = draw_bboxes_native(crop_annos, size=GEN_SIZE)
        panel = Image.new("RGB", (GEN_SIZE * 4, GEN_SIZE))
        panel.paste(bbox_img, (0, 0))
        panel.paste(composite, (GEN_SIZE, 0))
        panel.paste(harmonized, (GEN_SIZE * 2, 0))
        panel.paste(real_patch, (GEN_SIZE * 3, 0))
        panel.save(output_dir / f"{patch_id}_comparison.png")
        harmonized.save(output_dir / f"{patch_id}_harmonized.png")

        results.append({
            "patch_id": patch_id, "board": board, "color": board_color,
            "crop": [win_x, win_y], "n_components": len(crop_annos),
            "n_placed": placed,
        })

    with open(output_dir / f"results_{args.start}_{args.end}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone! {len(results)} patches -> {output_dir}")


if __name__ == "__main__":
    main()
