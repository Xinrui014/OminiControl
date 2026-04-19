#!/usr/bin/env python3
"""
Generate eval JSON for ALL 512x512 patches from the full test set.
Tiles each board with a fixed stride (25% overlap by default).

Usage:
    python scripts/generate_full_eval_patches.py \
        --anno_dir /home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/annotation/test \
        --image_dir /home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/test \
        --output /home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/eval_patches_full_test.json \
        --crop_size 512 --stride 384 --min_components 2
"""
import json
import argparse
import os
from pathlib import Path
from PIL import Image

CAT_ID_TO_NAME = {
    1: "RESISTOR", 2: "CAPACITOR", 3: "INDUCTOR", 4: "CONNECTOR",
    5: "DIODE", 6: "LED", 7: "SWITCH", 8: "TRANSISTOR", 9: "IC",
    10: "OSCILLATOR", 11: "FUSE",
}


def get_annotations_in_crop(annotations, crop_x, crop_y, crop_size, min_visible_ratio=0.5):
    """Extract annotations that fall within a crop, converted to crop-local coords."""
    cx2 = crop_x + crop_size
    cy2 = crop_y + crop_size
    result = []
    for ann in annotations:
        cat_id = ann["category_id"]
        if cat_id not in CAT_ID_TO_NAME:
            continue
        ax, ay, aw, ah = ann["bbox"]
        if aw <= 0 or ah <= 0:
            continue

        # Clip to crop bounds
        ox1 = max(ax, crop_x)
        oy1 = max(ay, crop_y)
        ox2 = min(ax + aw, cx2)
        oy2 = min(ay + ah, cy2)

        if ox2 <= ox1 or oy2 <= oy1:
            continue

        visible_ratio = ((ox2 - ox1) * (oy2 - oy1)) / (aw * ah)
        if visible_ratio < min_visible_ratio:
            continue

        result.append({
            "category_id": cat_id,
            "category_name": CAT_ID_TO_NAME[cat_id],
            "bbox": [
                round(ox1 - crop_x, 1),
                round(oy1 - crop_y, 1),
                round(ox2 - ox1, 1),
                round(oy2 - oy1, 1),
            ],
        })
    return result


def tile_positions(total, crop_size, stride):
    """Generate 1D tile start positions to cover total pixels."""
    positions = []
    pos = 0
    while pos + crop_size <= total:
        positions.append(pos)
        pos += stride
    # Ensure last tile covers the edge
    if positions[-1] + crop_size < total:
        positions.append(total - crop_size)
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_dir", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/annotation/test")
    parser.add_argument("--image_dir", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/test")
    parser.add_argument("--output", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/eval_patches_full_test.json")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--min_components", type=int, default=2)
    parser.add_argument("--min_visible_ratio", type=float, default=0.5)
    args = parser.parse_args()

    anno_files = sorted([f for f in os.listdir(args.anno_dir) if f.endswith(".json")])
    print(f"Found {len(anno_files)} test boards")

    all_patches = []
    skipped_small_board = 0
    skipped_few_comps = 0

    for ai, anno_file in enumerate(anno_files):
        board_name = anno_file.replace(".json", "")

        with open(os.path.join(args.anno_dir, anno_file)) as f:
            data = json.load(f)

        board_color = data.get("board_color", "green")
        annotations = data.get("annotations", [])

        # Get board dimensions from image
        img_path = os.path.join(args.image_dir, f"{board_name}.png")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        img_w, img_h = img.size
        img.close()

        # Skip boards smaller than crop_size
        if img_w < args.crop_size or img_h < args.crop_size:
            skipped_small_board += 1
            continue

        # Generate tile positions
        xs = tile_positions(img_w, args.crop_size, args.stride)
        ys = tile_positions(img_h, args.crop_size, args.stride)

        patch_idx = 0
        for cy in ys:
            for cx in xs:
                crop_anns = get_annotations_in_crop(
                    annotations, cx, cy, args.crop_size, args.min_visible_ratio
                )
                if len(crop_anns) < args.min_components:
                    skipped_few_comps += 1
                    continue

                patch_id = f"{board_name}_p{patch_idx}_{cx}_{cy}"
                all_patches.append({
                    "patch_id": patch_id,
                    "board": board_name,
                    "board_color": board_color,
                    "crop_x": cx,
                    "crop_y": cy,
                    "crop_size": args.crop_size,
                    "n_components": len(crop_anns),
                    "annotations": crop_anns,
                })
                patch_idx += 1

        if (ai + 1) % 50 == 0:
            print(f"  [{ai+1}/{len(anno_files)}] {len(all_patches)} patches so far")

    print(f"\nDone!")
    print(f"  Boards: {len(anno_files)}")
    print(f"  Skipped (too small): {skipped_small_board}")
    print(f"  Skipped patches (<{args.min_components} components): {skipped_few_comps}")
    print(f"  Total patches: {len(all_patches)}")
    print(f"  Avg components/patch: {sum(p['n_components'] for p in all_patches) / len(all_patches):.1f}")

    with open(args.output, "w") as f:
        json.dump(all_patches, f, indent=2)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
