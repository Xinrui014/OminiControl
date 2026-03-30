#!/usr/bin/env python3
"""
Generate eval_patches_small.json from the existing v2.1 inference outputs.
Extracts exact board/crop positions from filenames, then loads annotations
to build a complete, reproducible eval set.
"""
import json
import re
from pathlib import Path

CROP_SIZE = 512
REFERENCE_DIR = Path("output_v2_1/ckpt_6000")
ANNO_DIR = Path("/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/test")
TEST_JSONL = Path("/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/test.jsonl")

CAT_ID_TO_NAME = {
    1: "RESISTOR", 2: "CAPACITOR", 3: "INDUCTOR", 4: "CONNECTOR",
    5: "DIODE", 6: "LED", 7: "SWITCH", 8: "TRANSISTOR", 9: "IC",
    10: "OSCILLATOR", 11: "FUSE",
}


def get_annotations_in_crop(annotations, crop_x, crop_y, crop_size=512, min_visible_ratio=0.5):
    result = []
    cx2 = crop_x + crop_size
    cy2 = crop_y + crop_size
    for ann in annotations:
        cat_id = ann.get("category_id")
        if cat_id not in CAT_ID_TO_NAME:
            continue
        ax, ay, aw, ah = ann["bbox"]
        if aw <= 0 or ah <= 0:
            continue
        vx1 = max(ax, crop_x)
        vy1 = max(ay, crop_y)
        vx2 = min(ax + aw, cx2)
        vy2 = min(ay + ah, cy2)
        if vx2 <= vx1 or vy2 <= vy1:
            continue
        vis_area = (vx2 - vx1) * (vy2 - vy1)
        full_area = aw * ah
        if vis_area / full_area < min_visible_ratio:
            continue
        result.append({
            "category_id": cat_id,
            "category_name": CAT_ID_TO_NAME[cat_id],
            "bbox": [round(vx1 - crop_x, 1), round(vy1 - crop_y, 1),
                     round(vx2 - vx1, 1), round(vy2 - vy1, 1)],
        })
    return result


def main():
    # Load board colors from test jsonl
    board_colors = {}
    with open(TEST_JSONL) as f:
        for line in f:
            d = json.loads(line)
            meta = d.get("_meta", {})
            name = meta.get("image", "")
            color = meta.get("color", "green")
            if name:
                board_colors[name] = color

    # Parse patch info from existing output filenames
    comparison_files = sorted(REFERENCE_DIR.glob("*_comparison.png"))
    # filename pattern: {board}_p{idx}_{cx}_{cy}_comparison.png
    patch_re = re.compile(r"^(.+)_p(\d+)_(\d+)_(\d+)_comparison\.png$")

    eval_patches = []
    anno_cache = {}

    for f in comparison_files:
        m = patch_re.match(f.name)
        if not m:
            print(f"WARN: could not parse {f.name}")
            continue

        board = m.group(1)
        pidx = int(m.group(2))
        cx = int(m.group(3))
        cy = int(m.group(4))
        patch_id = f"{board}_p{pidx}_{cx}_{cy}"
        board_color = board_colors.get(board, "green")

        # Load annotations (cached per board)
        if board not in anno_cache:
            anno_path = ANNO_DIR / f"{board}.json"
            if anno_path.exists():
                with open(anno_path) as af:
                    anno_cache[board] = json.load(af).get("annotations", [])
            else:
                print(f"WARN: no annotations for {board}")
                anno_cache[board] = []

        patch_anns = get_annotations_in_crop(anno_cache[board], cx, cy, CROP_SIZE)

        eval_patches.append({
            "patch_id": patch_id,
            "board": board,
            "board_color": board_color,
            "crop_x": cx,
            "crop_y": cy,
            "crop_size": CROP_SIZE,
            "n_components": len(patch_anns),
            "annotations": patch_anns,
        })
        print(f"  {patch_id}: {len(patch_anns)} components")

    out_path = Path("config/eval_patches_small.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(eval_patches, f, indent=2)
    print(f"\nSaved {len(eval_patches)} patches → {out_path}")


if __name__ == "__main__":
    main()
