#!/usr/bin/env python3
"""
v3.3 inference pipeline B — native 1024 with white padding (matches v3.3 training
40% branch).

Pipeline:
  1. Load full test board (1280×720)
  2. Pad vertically to 1280×1280 (white, fixed center y_offset=280)
  3. Compute a 1024×1024 crop centered on the original eval patch's center
  4. Build composite at native 1024 using full-board test annotations
     (shifted by y_offset, clipped to the 1024 crop window)
  5. Generate at 1024 (no upscale step)
  6. Save at 1024

Usage:
    CUDA_VISIBLE_DEVICES=0 python infer_v3_3_native1024.py \
        --eval_json /projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json \
        --omini_ckpt runs/v3.3_refined_1024/20260419-121405/ckpt/8000 \
        --output_dir runs/v3.3_refined_1024/eval/ckpt8k_B_sanity20 \
        --start 0 --end 20
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


def load_model(checkpoint_path, device="cuda"):
    """Load FLUX.1-dev from the local model dir + LoRA adapter."""
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
PAD_SIZE = 1280     # match training native_pad_size
PAD_COLOR = (255, 255, 255)
FIXED_Y_OFFSET = 280   # center-pad a 720-tall board to 1280

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


def load_board_padded(board_image_dir, board_name):
    """Load 1280x720 board and pad to 1280x1280 (white, fixed center y_offset)."""
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
    w, h = img.size
    pad_w = max(PAD_SIZE, w, GEN_SIZE)
    pad_h = max(PAD_SIZE, h, GEN_SIZE)
    padded = Image.new("RGB", (pad_w, pad_h), PAD_COLOR)
    x_off = (pad_w - w) // 2
    y_off = FIXED_Y_OFFSET if pad_h >= h + FIXED_Y_OFFSET else (pad_h - h) // 2
    padded.paste(img, (x_off, y_off))
    return padded, x_off, y_off, pad_w, pad_h


def annotations_in_window(full_annos, x_off, y_off, win_x, win_y,
                          win_size=1024, min_visible_ratio=0.3):
    """Return annotations that fall inside the 1024 crop window, with bboxes
    translated to crop-relative coords."""
    out = []
    for ann in full_annos:
        cat_id = ann.get("category_id")
        if cat_id not in CAT_ID_TO_NAME:
            continue
        ax, ay, aw, ah = ann["bbox"]
        # Shift board coords → padded image coords
        ax += x_off
        ay += y_off
        # Intersect with crop window
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
            # crop-relative clipped bbox
            "bbox": (x1 - win_x, y1 - win_y, x2 - x1, y2 - y1),
            # full bbox in padded image coords (for bank match on full size)
            "original_bbox": (ax, ay, aw, ah),
            "orientation": ann.get("orientation") or 0,
            "sub_class": ann.get("sub_class") if ann.get("sub_class") is not None else -1,
            "visible_ratio": visible,
        })
    return out


def paste_composite_native(annos, bank, board_color, resolution_class,
                            exclude_board, padded_img, crop_x, crop_y,
                            canvas_size=1024, top_k=10):
    """Build composite at native 1024 by matching each annotation's full bbox
    and cropping to visible portion. Self-fallback = pixels from padded_img."""
    canvas = Image.new("RGB", (canvas_size, canvas_size), PAD_COLOR)
    placed = 0
    for ann in annos:
        cat = ann["category_name"]
        rx, ry, rw, rh = ann["bbox"]              # clipped, crop-relative
        ox, oy, ow, oh = ann["original_bbox"]     # full, padded-img coords
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

        # Self-fallback from padded_img
        if crop_to_paste is None:
            abs_x = crop_x + int(rx)
            abs_y = crop_y + int(ry)
            crop_to_paste = padded_img.crop(
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

    # Preload test board annotations + metadata
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

    padded_boards = {}   # board_name -> (padded_img, x_off, y_off, pad_w, pad_h)
    results = []

    for idx, patch in enumerate(eval_patches):
        patch_id = patch["patch_id"]
        board = patch["board"]
        board_color = patch.get("board_color") or "green"
        cx = patch["crop_x"]
        cy = patch["crop_y"]
        crop_size_orig = patch["crop_size"]  # 512

        # Pad board once and cache
        if board not in padded_boards:
            padded_boards[board] = load_board_padded(args.board_image_dir, board)
        pb = padded_boards[board]
        if pb is None:
            print(f"[{idx+1}/{len(eval_patches)}] {patch_id} — SKIP (no board image)")
            continue
        padded_img, x_off, y_off, pad_w, pad_h = pb

        # Center of the 512 eval patch on the padded image
        patch_center_x = cx + crop_size_orig // 2 + x_off
        patch_center_y = cy + crop_size_orig // 2 + y_off
        # 1024 crop centered on that, clamped to padded bounds
        win_x = max(0, min(pad_w - GEN_SIZE, patch_center_x - GEN_SIZE // 2))
        win_y = max(0, min(pad_h - GEN_SIZE, patch_center_y - GEN_SIZE // 2))

        # Real patch at 1024 native
        real_patch = padded_img.crop((win_x, win_y, win_x + GEN_SIZE, win_y + GEN_SIZE))

        # Annotations inside the 1024 window (from full test board annos, shifted)
        full_annos = test_board_annos.get(board, [])
        crop_annos = annotations_in_window(
            full_annos, x_off, y_off, win_x, win_y, win_size=GEN_SIZE,
            min_visible_ratio=0.3,
        )

        # Metadata (color / res) from full test board
        meta = test_board_meta.get(board, {})
        resolution_class = meta.get("resolution_class", "R3")
        # Prefer patch-level color; fallback to board meta; final default green
        if not board_color or board_color not in PROMPT_TEMPLATES:
            board_color = meta.get("color", "green")
        prompt = PROMPT_TEMPLATES.get(board_color) or PROMPT_TEMPLATES["green"]

        print(f"\n[{idx+1}/{len(eval_patches)}] {patch_id} "
              f"({len(crop_annos)} components in 1024 window, "
              f"win=({win_x},{win_y}))")

        # Skip if already done
        if (output_dir / f"{patch_id}_harmonized.png").exists():
            print(f"  SKIP (exists)")
            continue

        abs_idx = args.start + idx
        _random.seed(args.seed + abs_idx)

        # Build native 1024 composite
        composite, placed = paste_composite_native(
            crop_annos, bank, board_color, resolution_class,
            exclude_board=board, padded_img=padded_img,
            crop_x=win_x, crop_y=win_y,
            canvas_size=GEN_SIZE, top_k=args.top_k,
        )
        print(f"  Placed: {placed}/{len(crop_annos)}")

        # Generate at 1024 native (NO upscale step)
        condition = Condition(composite, "pcb_harmonize")
        seed_everything(args.seed + abs_idx)
        harmonized = generate(
            pipe, prompt=prompt, conditions=[condition],
            height=GEN_SIZE, width=GEN_SIZE,
        ).images[0]

        # Save: 4-panel bbox | composite | harmonized | real (all 1024×1024)
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
