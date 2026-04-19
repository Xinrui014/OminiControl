#!/usr/bin/env python3
"""
Full test set inference for v3.2 — native 1024x1024.

Builds composite at 1024, generates at 1024, saves at 1024.
Eval patches are defined at 512 crop coords — we scale annotations to 1024.

Usage:
    CUDA_VISIBLE_DEVICES=0 python infer_v3_2_full_eval.py \
        --eval_json eval_patches_full_test_v2.1.json \
        --omini_ckpt runs/v3.2_subclass_1024/ckpt/10000 \
        --output_dir runs/v3.2_subclass_1024/eval_full/ckpt10k \
        --start 0 --end 500
"""
import json
import argparse
import random as _random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import FluxPipeline

from omini.pipeline.flux_omini import Condition, generate, seed_everything
from lib.component_bank_v2_1 import ComponentBankV2_1, CAT_ID_TO_NAME, parse_resolution_class

FLUX_PATH = "/home/xinrui/projects/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
GEN_SIZE = 1024  # native generation resolution

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


def load_model(checkpoint_path: str, device: str = "cuda") -> FluxPipeline:
    print(f"Loading FLUX.1-dev base model from {FLUX_PATH}...")
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=torch.bfloat16).to(device)
    print(f"Loading LoRA: {checkpoint_path}")
    pipe.load_lora_weights(
        checkpoint_path,
        weight_name="default.safetensors",
        adapter_name="pcb_harmonize",
    )
    pipe.set_adapters(["pcb_harmonize"])
    print("Model ready.")
    return pipe


def scale_annotations(annotations, scale):
    """Scale annotation bboxes by a factor."""
    scaled = []
    for ann in annotations:
        ann_s = dict(ann)
        x, y, w, h = ann["bbox"]
        ann_s["bbox"] = [x * scale, y * scale, w * scale, h * scale]
        scaled.append(ann_s)
    return scaled


def paste_patch_composite(annotations, bank, board_color, resolution_class,
                          exclude_board, canvas_size=1024, top_k=10):
    """Build composite at canvas_size with annotations already in canvas_size coords."""
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    placed = 0
    for ann in annotations:
        cat = ann.get("category_name") or CAT_ID_TO_NAME.get(ann.get("category_id"), "unknown")
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 2 or h <= 2:
            continue
        sub_class = ann.get("sub_class", -1)
        orientation = ann.get("orientation", 0)

        result = bank.find_match(
            category=cat, sub_class=sub_class,
            target_w=w, target_h=h,
            board_color=board_color, resolution_class=resolution_class,
            orientation=orientation, exclude_board=exclude_board,
            top_k=top_k,
        )
        if result is not None:
            entry, rotation = result
            crop = bank.load_crop(entry, w, h, rotation=rotation, resize_jitter=0.0)
        else:
            orig_bbox = ann.get("original_bbox")
            if orig_bbox is not None:
                crop = bank.load_self_crop(exclude_board, orig_bbox, w, h)
            else:
                crop = None
        if crop is None:
            continue
        px = max(0, min(x, canvas_size - crop.width))
        py = max(0, min(y, canvas_size - crop.height))
        img.paste(crop, (px, py))
        placed += 1
    return img, placed


def draw_patch_bboxes(annotations, canvas_size=1024):
    """Draw bbox visualization at canvas_size with annotations already scaled."""
    img = Image.new("RGB", (canvas_size, canvas_size), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for ann in annotations:
        cat = ann.get("category_name") or CAT_ID_TO_NAME.get(ann.get("category_id"), "unknown")
        x, y, w, h = ann["bbox"]
        color = BBOX_COLORS.get(cat, (128, 128, 128))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((int(x) + 2, int(y) + 2), cat[:3], fill=color)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--anno_dir",
        default="/home/xinrui/projects/data/ti_pcb/layout_data/v2.1_subclass/annotation/train")
    parser.add_argument("--image_dir",
        default="/home/xinrui/projects/data/ti_pcb/layout_data/v2.1_subclass/image/train")
    parser.add_argument("--board_image_dir",
        default="/home/xinrui/projects/data/ti_pcb/layout_data/v2.1_subclass/image/test")
    parser.add_argument("--anno_test_dir",
        default="/home/xinrui/projects/data/ti_pcb/layout_data/v2.1_subclass/annotation/test")
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
        anno_dir=args.anno_dir,
        image_dir=args.image_dir,
        edge_margin=5,
    )

    # Preload test board annotations for resolution_class lookup
    test_board_meta = {}
    for anno_path in Path(args.anno_test_dir).glob("*.json"):
        with open(anno_path) as f:
            data = json.load(f)
        test_board_meta[anno_path.stem] = {
            "color": data.get("board_color", "green"),
            "resolution_class": parse_resolution_class(data.get("resolution_class", "R3")),
        }

    print(f"Loaded metadata for {len(test_board_meta)} test boards")

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
        crop_size = patch["crop_size"]  # 512 in eval JSON
        annotations = patch["annotations"]  # bbox in 512 space

        # Scale factor from eval crop (512) to generation resolution (1024)
        scale = GEN_SIZE / crop_size

        # Get resolution_class from test board metadata
        meta = test_board_meta.get(board, {})
        resolution_class = meta.get("resolution_class", "R3")

        prompt = PROMPT_TEMPLATES.get(board_color, PROMPT_TEMPLATES["green"])

        # Skip if already done
        if (output_dir / f"{patch_id}_harmonized.png").exists():
            print(f"[{idx+1}/{len(eval_patches)}] {patch_id} — SKIP (exists)")
            continue

        print(f"\n[{idx+1}/{len(eval_patches)}] {patch_id} ({len(annotations)} components)")

        # Ensure annotations have category_name
        for ann in annotations:
            if "category_name" not in ann:
                ann["category_name"] = CAT_ID_TO_NAME.get(ann.get("category_id"), "unknown")

        # Scale annotations to 1024 space
        annotations_1024 = scale_annotations(annotations, scale)

        # Load board image (cached)
        if board not in board_images:
            board_path = Path(args.board_image_dir) / f"{board}.png"
            if board_path.exists():
                img = Image.open(board_path)
                if img.mode in ("RGBA", "PA", "P"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.convert("RGBA").split()[3])
                    img = bg
                else:
                    img = img.convert("RGB")
                board_images[board] = img
            else:
                board_images[board] = None

        board_img = board_images[board]
        orig_patch = None
        if board_img is not None:
            # Crop at original coords, then upscale to 1024
            orig_patch_512 = board_img.crop((cx, cy, cx + crop_size, cy + crop_size))
            orig_patch = orig_patch_512.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)

        # Deterministic seeding
        abs_idx = args.start + idx
        _random.seed(args.seed + abs_idx)

        # Build composite directly at 1024 with scaled annotations
        composite, placed = paste_patch_composite(
            annotations_1024, bank, board_color, resolution_class,
            exclude_board=board, canvas_size=GEN_SIZE, top_k=args.top_k,
        )
        print(f"  Placed: {placed}/{len(annotations)}")

        # Generate at 1024 — no upscale/downscale needed
        condition = Condition(composite, "pcb_harmonize")
        seed_everything(args.seed + abs_idx)
        harmonized = generate(
            pipe, prompt=prompt, conditions=[condition],
            height=GEN_SIZE, width=GEN_SIZE,
        ).images[0]

        # Bbox visualization at 1024
        bbox_img = draw_patch_bboxes(annotations_1024, GEN_SIZE)

        # 4-panel comparison at 1024: bbox | composite | harmonized | original
        panel = Image.new("RGB", (GEN_SIZE * 4, GEN_SIZE))
        panel.paste(bbox_img, (0, 0))
        panel.paste(composite, (GEN_SIZE, 0))
        panel.paste(harmonized, (GEN_SIZE * 2, 0))
        if orig_patch is not None:
            panel.paste(orig_patch, (GEN_SIZE * 3, 0))

        panel.save(output_dir / f"{patch_id}_comparison.png")
        harmonized.save(output_dir / f"{patch_id}_harmonized.png")

        results.append({
            "patch_id": patch_id, "board": board, "color": board_color,
            "crop": [cx, cy], "n_components": len(annotations), "n_placed": placed,
        })

    with open(output_dir / f"results_{args.start}_{args.end}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! {len(results)} patches -> {output_dir}")


if __name__ == "__main__":
    main()
