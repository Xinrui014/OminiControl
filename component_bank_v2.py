"""
ComponentBankV2 — On-the-fly component matching + pasting for v2 harmonization training.

Loads all v2 training annotations (3,515 boards × 1280×720) as the component pool.
Supports color-matched, resolution-matched pasting with random resize jitter.

Used by train_pcb_v2.py for on-the-fly composite generation during training.
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Category ID → token name mapping (COCO annotation IDs)
CAT_ID_TO_NAME = {
    1: "RESISTOR", 2: "CAPACITOR", 3: "INDUCTOR", 4: "CONNECTOR",
    5: "DIODE", 7: "SWITCH", 8: "TRANSISTOR", 9: "IC", 10: "OSCILLATOR",
    11: "FUSE",
}
CAT_NAME_TO_ID = {v: k for k, v in CAT_ID_TO_NAME.items()}


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
    Component pool built from v2 COCO annotations + full 1280×720 board images.

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
        v2_jsonl: str,
        edge_margin: int = 5,
        max_cache: int = 300,
    ):
        """
        Args:
            anno_dir: Directory with per-board COCO JSON annotation files
            image_dir: Directory with 1280×720 board images (PNG)
            v2_jsonl: v2 train.jsonl — used to get board color/resolution metadata
            edge_margin: Skip components within this many pixels of image edge
            max_cache: Max board images to cache in memory
        """
        self.image_dir = image_dir
        self.max_cache = max_cache
        self._img_cache = {}

        # Load board metadata from v2 jsonl
        self.board_meta = {}  # board_name -> {color, resolution}
        with open(v2_jsonl) as f:
            for line in f:
                entry = json.loads(line)
                meta = entry["_meta"]
                self.board_meta[meta["image"]] = {
                    "color": meta.get("color", "green"),
                    "resolution": meta.get("resolution", "R3"),
                }

        # Build component pool from annotations
        self.by_category = defaultdict(list)          # category -> [ComponentEntry]
        self.by_cat_color = defaultdict(list)          # (category, color) -> [ComponentEntry]

        skipped_edge = 0
        total = 0

        for board_name, meta in self.board_meta.items():
            anno_path = os.path.join(anno_dir, f"{board_name}.json")
            if not os.path.exists(anno_path):
                continue

            with open(anno_path) as f:
                data = json.load(f)

            # Get image dimensions
            img_info = data["images"][0] if data.get("images") else None
            img_w = img_info["width"] if img_info else 1280
            img_h = img_info["height"] if img_info else 720

            for ann in data.get("annotations", []):
                cat_id = ann["category_id"]
                cat_name = CAT_ID_TO_NAME.get(cat_id)
                if cat_name is None:
                    continue

                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue

                # Skip edge components (likely truncated)
                if (x < edge_margin or y < edge_margin or
                        x + w > img_w - edge_margin or y + h > img_h - edge_margin):
                    skipped_edge += 1
                    continue

                entry = ComponentEntry(board_name, (x, y, w, h), cat_name)
                self.by_category[cat_name].append(entry)
                self.by_cat_color[(cat_name, meta["color"])].append(entry)
                total += 1

        print(f"[ComponentBankV2] Loaded {total} components from {len(self.board_meta)} boards")
        print(f"  Skipped {skipped_edge} edge components (margin={edge_margin}px)")
        for cat in sorted(self.by_category.keys()):
            print(f"  {cat}: {len(self.by_category[cat])}")
        print(f"  Color breakdown:")
        color_counts = defaultdict(int)
        for (cat, color), entries in self.by_cat_color.items():
            color_counts[color] += len(entries)
        for color, count in sorted(color_counts.items()):
            print(f"    {color}: {count}")

    def _get_board_image(self, board_name: str) -> Optional[Image.Image]:
        if board_name not in self._img_cache:
            path = os.path.join(self.image_dir, f"{board_name}.png")
            if not os.path.exists(path):
                return None
            self._img_cache[board_name] = Image.open(path).convert("RGB")
            if len(self._img_cache) > self.max_cache:
                # Evict oldest
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
            category: Component category name (e.g., "RESISTOR")
            target_w, target_h: Target component dimensions
            board_color: If provided, prefer components from same-colored boards
            top_k: Pick randomly from top-K matches by AR similarity
            size_thresh: Min ratio of min(area)/max(area) to consider
            exclude_board: Don't pick from this board (avoid self-matching)

        Returns:
            ComponentEntry or None
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

        # Filter by size
        filtered = [
            e for e in candidates
            if (min(target_area, e.area) / max(target_area, e.area) >= size_thresh
                and (exclude_board is None or e.board_name != exclude_board))
        ]

        if len(filtered) < top_k:
            # Relax: just exclude self-board, sort by area similarity
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
            entry: ComponentEntry to crop
            target_w, target_h: Desired output size
            resize_jitter: Random scale factor range (e.g., 0.15 = ±15%)

        Returns:
            Resized PIL Image or None
        """
        board = self._get_board_image(entry.board_name)
        if board is None:
            return None

        x, y, w, h = entry.bbox
        crop = board.crop((int(x), int(y), int(x + w), int(y + h)))

        # Apply resize jitter
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
    Filter and clip annotations to a crop window.

    Args:
        annotations: List of COCO annotation dicts with 'bbox' and 'category_id'
        crop_x, crop_y: Top-left corner of the crop in the original image
        crop_size: Size of the square crop
        min_visible_ratio: Include component only if this fraction is visible

    Returns:
        List of annotation dicts with bbox coordinates relative to the crop
    """
    result = []
    cx2 = crop_x + crop_size
    cy2 = crop_y + crop_size

    for ann in annotations:
        cat_id = ann["category_id"]
        if cat_id not in CAT_ID_TO_NAME:
            continue

        ax, ay, aw, ah = ann["bbox"]
        if aw <= 0 or ah <= 0:
            continue

        # Compute overlap with crop window
        ox1 = max(ax, crop_x)
        oy1 = max(ay, crop_y)
        ox2 = min(ax + aw, cx2)
        oy2 = min(ay + ah, cy2)

        if ox2 <= ox1 or oy2 <= oy1:
            continue  # No overlap

        overlap_area = (ox2 - ox1) * (oy2 - oy1)
        total_area = aw * ah
        visible_ratio = overlap_area / total_area

        if visible_ratio < min_visible_ratio:
            continue

        # Clip bbox to crop and convert to crop-relative coordinates
        rel_x = ox1 - crop_x
        rel_y = oy1 - crop_y
        rel_w = ox2 - ox1
        rel_h = oy2 - oy1

        result.append({
            "category_id": cat_id,
            "category_name": CAT_ID_TO_NAME[cat_id],
            "bbox": (rel_x, rel_y, rel_w, rel_h),
            "original_bbox": (ax, ay, aw, ah),
            "visible_ratio": visible_ratio,
        })

    return result
