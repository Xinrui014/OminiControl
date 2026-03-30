"""
ComponentBankV2 — Component matching + pasting for v2 PCB harmonization.

Loads components from per-board COCO annotation JSONs (float coordinates,
1280×720 boards). Supports color-matched pasting and resize jitter.

Data layout:
    anno_dir/         — per-board COCO JSON files (<board>.json)
    image_dir/        — 1280×720 board PNG images (<board>.png)

Each annotation JSON has:
    - images: board metadata (width, height)
    - categories: component types
    - annotations: bboxes with category_id
    - board_color: PCB color (green/blue/red/black/white) — NEW in v2
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Category ID → token name mapping
CAT_ID_TO_NAME = {
    1: "RESISTOR", 2: "CAPACITOR", 3: "INDUCTOR", 4: "CONNECTOR",
    5: "DIODE", 6: "LED", 7: "SWITCH", 8: "TRANSISTOR", 9: "IC",
    10: "OSCILLATOR", 11: "FUSE",
}
# Also accept "Integrated Circuit" as IC (from some annotation files)
CAT_NAME_TO_ID = {v: k for k, v in CAT_ID_TO_NAME.items()}
CAT_NAME_TO_ID["Integrated Circuit"] = 9
CAT_NAME_TO_ID["Integrated_Circuit"] = 9


class ComponentEntry:
    """A single component crop reference in the pool."""
    __slots__ = ("board_name", "bbox", "category", "area", "ar")

    def __init__(self, board_name: str, bbox: Tuple[float, float, float, float], category: str):
        self.board_name = board_name
        x, y, w, h = bbox
        self.bbox = (x, y, w, h)
        self.category = category
        self.area = w * h
        self.ar = w / h if h > 0 else 1.0


class ComponentBankV2:
    """
    Component pool built from per-board COCO annotation JSON files + 1280×720 board images.

    Loads annotations from `anno_dir/<board>.json` (COCO format with float bboxes).
    Board colors are read directly from the JSON's `board_color` field.

    Supports:
    - Category matching
    - Size/AR matching (top-K nearest)
    - Board color filtering (optional)
    - Random resize jitter
    """

    def __init__(
        self,
        anno_dir: str,
        image_dir: str,
        edge_margin: int = 5,
        max_cache: int = 300,
    ):
        """
        Args:
            anno_dir:      Directory with per-board COCO JSONs (<board>.json)
            image_dir:     Directory with 1280×720 board PNG images (<board>.png)
            edge_margin:   Skip components within this many pixels of image edge
            max_cache:     Max board images to cache in memory
        """
        self.image_dir = image_dir
        self.max_cache = max_cache
        self._img_cache: Dict[str, Image.Image] = {}

        # Build component pool
        self.by_category: Dict[str, List[ComponentEntry]] = defaultdict(list)
        self.by_cat_color: Dict[Tuple[str, str], List[ComponentEntry]] = defaultdict(list)

        skipped_edge = 0
        total = 0
        boards_loaded = 0
        boards_missing_image = 0
        boards_empty = 0
        color_counts = defaultdict(int)

        anno_files = sorted(Path(anno_dir).glob("*.json"))
        print(f"[ComponentBankV2] Loading from {anno_dir} ({len(anno_files)} boards)...")

        for anno_path in anno_files:
            board_name = anno_path.stem

            # Check board image exists
            img_path = os.path.join(image_dir, f"{board_name}.png")
            if not os.path.exists(img_path):
                boards_missing_image += 1
                continue

            with open(anno_path) as f:
                data = json.load(f)

            # Get board color from embedded field (now standard in v2 JSONs)
            board_color = data.get("board_color", "green")

            # Get image dimensions
            img_info = data["images"][0] if data.get("images") else None
            img_w = img_info["width"] if img_info else 1280
            img_h = img_info["height"] if img_info else 720

            annotations = data.get("annotations", [])
            if not annotations:
                boards_empty += 1
                continue

            board_components = 0
            for ann in annotations:
                # Resolve category
                cat_id = ann.get("category_id")
                cat_name = CAT_ID_TO_NAME.get(cat_id)
                if cat_name is None:
                    continue

                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue

                # Skip edge components (likely truncated)
                if (x < edge_margin or y < edge_margin or
                        x + w > img_w - edge_margin or
                        y + h > img_h - edge_margin):
                    skipped_edge += 1
                    continue

                entry = ComponentEntry(board_name, (x, y, w, h), cat_name)
                self.by_category[cat_name].append(entry)
                self.by_cat_color[(cat_name, board_color)].append(entry)
                total += 1
                board_components += 1

            if board_components > 0:
                boards_loaded += 1
                color_counts[board_color] += board_components

        print(f"[ComponentBankV2] Loaded {total} components from {boards_loaded} boards")
        if boards_missing_image:
            print(f"  Skipped {boards_missing_image} boards (image not found)")
        if boards_empty:
            print(f"  Skipped {boards_empty} boards (0 annotations)")
        print(f"  Skipped {skipped_edge} edge components (margin={edge_margin}px)")
        for cat in sorted(self.by_category.keys()):
            print(f"  {cat}: {len(self.by_category[cat])}")
        print("  Color breakdown:")
        for color, count in sorted(color_counts.items()):
            print(f"    {color}: {count}")

    def _get_board_image(self, board_name: str) -> Optional[Image.Image]:
        if board_name not in self._img_cache:
            path = os.path.join(self.image_dir, f"{board_name}.png")
            if not os.path.exists(path):
                return None
            self._img_cache[board_name] = Image.open(path).convert("RGB")
            if len(self._img_cache) > self.max_cache:
                oldest = next(iter(self._img_cache))
                del self._img_cache[oldest]
        return self._img_cache.get(board_name)

    def find_match(
        self,
        category: str,
        target_w: float,
        target_h: float,
        board_color: Optional[str] = None,
        top_k: int = 10,
        size_thresh: float = 0.5,
        exclude_board: Optional[str] = None,
    ) -> Optional[ComponentEntry]:
        """
        Find a matching component from the pool.

        Args:
            category:     Component category name (e.g., "RESISTOR")
            target_w/h:   Target component dimensions
            board_color:  If provided, prefer same-colored boards
            top_k:        Pick randomly from top-K matches by AR similarity
            size_thresh:  Min ratio of min(area)/max(area) to consider
            exclude_board: Don't pick from this board (avoid self-matching)
        """
        # Try color-matched first, fall back to all
        if board_color:
            candidates = self.by_cat_color.get((category, board_color), [])
            if len(candidates) < top_k:
                candidates = self.by_category.get(category, [])
        else:
            candidates = self.by_category.get(category, [])

        if not candidates:
            return None

        target_area = target_w * target_h
        target_ar = target_w / target_h if target_h > 0 else 1.0
        target_horiz = target_w >= target_h

        # Filter by size + exclude board
        filtered = [
            e for e in candidates
            if (min(target_area, e.area) / max(target_area, e.area) >= size_thresh
                and (exclude_board is None or e.board_name != exclude_board))
        ]

        if len(filtered) < top_k:
            filtered = [e for e in candidates
                        if exclude_board is None or e.board_name != exclude_board]
            filtered.sort(key=lambda e: abs(target_area - e.area))
            filtered = filtered[:max(top_k * 5, 50)]

        # Prefer same orientation, rank by AR similarity
        same_orient = [e for e in filtered if (e.bbox[2] >= e.bbox[3]) == target_horiz]
        pool = same_orient if len(same_orient) >= top_k else filtered
        pool.sort(key=lambda e: abs(target_ar - e.ar))

        return random.choice(pool[:top_k]) if pool else None

    def load_crop(
        self,
        entry: ComponentEntry,
        target_w: int,
        target_h: int,
        resize_jitter: float = 0.0,
    ) -> Optional[Image.Image]:
        """
        Load and resize a component crop from its source board.

        Args:
            entry:         ComponentEntry to crop
            target_w/h:    Desired output size
            resize_jitter: Random scale factor range (e.g., 0.15 = ±15%)
        """
        board = self._get_board_image(entry.board_name)
        if board is None:
            return None

        x, y, w, h = entry.bbox
        crop = board.crop((int(x), int(y), int(x + w), int(y + h)))

        if resize_jitter > 0:
            scale = 1.0 + random.uniform(-resize_jitter, resize_jitter)
            target_w = max(1, int(target_w * scale))
            target_h = max(1, int(target_h * scale))

        return crop.resize((max(target_w, 1), max(target_h, 1)), Image.LANCZOS)


def get_annotations_in_crop(
    annotations: List[dict],
    crop_x: int,
    crop_y: int,
    crop_size: int = 512,
    min_visible_ratio: float = 0.5,
) -> List[dict]:
    """
    Filter and clip annotations to a 512×512 crop window.

    Args:
        annotations:       COCO annotation dicts with 'bbox' and 'category_id'
        crop_x/y:          Top-left corner of crop in original image
        crop_size:         Size of the square crop
        min_visible_ratio: Include component only if this fraction is visible

    Returns:
        Annotation dicts with bbox coords relative to the crop.
    """
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
            "bbox": (ox1 - crop_x, oy1 - crop_y, ox2 - ox1, oy2 - oy1),
            "original_bbox": (ax, ay, aw, ah),
            "visible_ratio": visible_ratio,
        })

    return result
