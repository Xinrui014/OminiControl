#!/usr/bin/env python3
"""
Generate eval_patches_small_components.json — patches with many small (<30px) components.
Scans all test boards, scores every possible crop by small component density, picks the best.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

CROP_SIZE = 512
SMALL_THRESH = 30  # components with max(w,h) < 30px
NUM_PATCHES = 30
ANNO_DIR = Path("/home/xinrui/projects/data/ti_pcb/COCO_label/annotation/test")
TEST_JSONL = Path("/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/test.jsonl")
BOARD_IMG_DIR = Path("/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/test")

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


def count_small(annotations):
    """Count components with max(w,h) < SMALL_THRESH."""
    n = 0
    for ann in annotations:
        _, _, w, h = ann["bbox"]
        if max(w, h) < SMALL_THRESH:
            n += 1
    return n


def main():
    random.seed(123)

    # Load board colors
    board_colors = {}
    with open(TEST_JSONL) as f:
        for line in f:
            d = json.loads(line)
            meta = d.get("_meta", {})
            name = meta.get("image", "")
            color = meta.get("color", "green")
            if name:
                board_colors[name] = color

    # Scan all test boards for small-component-dense crops
    candidates = []
    anno_files = sorted(ANNO_DIR.glob("*.json"))

    for anno_path in anno_files:
        board = anno_path.stem
        board_img_path = BOARD_IMG_DIR / f"{board}.png"
        if not board_img_path.exists():
            continue

        from PIL import Image
        img = Image.open(board_img_path)
        bw, bh = img.size
        img.close()

        with open(anno_path) as f:
            anno_data = json.load(f)
        all_anns = anno_data.get("annotations", [])
        if len(all_anns) < 5:
            continue

        board_color = board_colors.get(board, "green")

        # Sample crop positions (grid + random)
        positions = set()
        step = 128
        for cx in range(0, max(1, bw - CROP_SIZE + 1), step):
            for cy in range(0, max(1, bh - CROP_SIZE + 1), step):
                positions.add((cx, cy))
        # Add some random positions
        for _ in range(50):
            cx = random.randint(0, max(0, bw - CROP_SIZE))
            cy = random.randint(0, max(0, bh - CROP_SIZE))
            positions.add((cx, cy))

        for cx, cy in positions:
            patch_anns = get_annotations_in_crop(all_anns, cx, cy, CROP_SIZE)
            if len(patch_anns) < 3:
                continue
            n_small = count_small(patch_anns)
            if n_small < 3:
                continue
            candidates.append({
                "board": board,
                "board_color": board_color,
                "crop_x": cx,
                "crop_y": cy,
                "n_components": len(patch_anns),
                "n_small": n_small,
                "small_ratio": n_small / len(patch_anns),
                "annotations": patch_anns,
            })

    print(f"Found {len(candidates)} candidate crops with >= 3 small components")

    # Sort by number of small components (descending), then by total components
    candidates.sort(key=lambda c: (c["n_small"], c["n_components"]), reverse=True)

    # Deduplicate: don't pick overlapping crops from same board
    selected = []
    used = defaultdict(list)  # board -> list of (cx, cy)

    for c in candidates:
        board = c["board"]
        cx, cy = c["crop_x"], c["crop_y"]

        # Skip if too close to an already-selected crop on the same board
        too_close = False
        for px, py in used[board]:
            if abs(cx - px) < 256 and abs(cy - py) < 256:
                too_close = True
                break
        if too_close:
            continue

        # Limit patches per board
        if len(used[board]) >= 3:
            continue

        patch_id = f"{board}_p{len(used[board])}_{cx}_{cy}"
        selected.append({
            "patch_id": patch_id,
            "board": c["board"],
            "board_color": c["board_color"],
            "crop_x": cx,
            "crop_y": cy,
            "crop_size": CROP_SIZE,
            "n_components": c["n_components"],
            "n_small": c["n_small"],
            "annotations": c["annotations"],
        })
        used[board].append((cx, cy))

        print(f"  {patch_id}: {c['n_components']} total, {c['n_small']} small ({c['small_ratio']:.0%})")

        if len(selected) >= NUM_PATCHES:
            break

    out_path = Path("config/eval_patches_small_components.json")
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved {len(selected)} patches → {out_path}")


if __name__ == "__main__":
    main()
