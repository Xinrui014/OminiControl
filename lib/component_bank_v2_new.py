"""
ComponentBankV2_new — Component matching with full metadata: color, resolution class, orientation.

Loads from v2 annotations (v2_Color_Res_Class_xywh/annotation/) which include:
  - Per board: board_color, resolution_class
  - Per component: orientation (0/90/180/270/45/135/225/315)

Matching uses cascading fallback:
  L0: category + color + resolution_class + orientation_bucket
  L1: category + color + resolution_class
  L2: category + color + adjacent resolution (±1)
  L3: category + color (any resolution)
  L4: category + resolution_class (any color)
  L5: category only (last resort)
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

CAT_ID_TO_NAME = {
    1: "RESISTOR", 2: "CAPACITOR", 3: "INDUCTOR", 4: "CONNECTOR",
    5: "DIODE", 6: "LED", 7: "SWITCH", 8: "TRANSISTOR", 9: "IC",
    10: "OSCILLATOR", 11: "FUSE",
}
CAT_NAME_TO_ID = {v: k for k, v in CAT_ID_TO_NAME.items()}
CAT_NAME_TO_ID["Integrated Circuit"] = 9
CAT_NAME_TO_ID["Integrated_Circuit"] = 9

# Resolution class ordering for adjacency lookup
RES_ORDER = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
RES_IDX = {r: i for i, r in enumerate(RES_ORDER)}


def orient_bucket(angle: int) -> str:
    """Map orientation angle to bucket: '0', '90', or 'other'."""
    if angle == 0 or angle == 180:
        return "0"
    elif angle == 90 or angle == 270:
        return "90"
    else:
        return "other"


def adjacent_res_classes(res_class: str) -> List[str]:
    """Return ±1 adjacent resolution classes."""
    idx = RES_IDX.get(res_class)
    if idx is None:
        return []
    adj = []
    if idx > 0:
        adj.append(RES_ORDER[idx - 1])
    if idx < len(RES_ORDER) - 1:
        adj.append(RES_ORDER[idx + 1])
    return adj


class ComponentEntry:
    """A single component crop reference in the pool."""
    __slots__ = ("board_name", "bbox", "category", "area", "ar",
                 "board_color", "resolution_class", "orientation")

    def __init__(self, board_name: str, bbox: Tuple[float, float, float, float],
                 category: str, board_color: str, resolution_class: str, orientation: int):
        self.board_name = board_name
        x, y, w, h = bbox
        self.bbox = (x, y, w, h)
        self.category = category
        self.area = w * h
        self.ar = w / h if h > 0 else 1.0
        self.board_color = board_color
        self.resolution_class = resolution_class
        self.orientation = orientation


class ComponentBankV2_new:
    """
    Component pool with full metadata matching: color, resolution class, orientation.

    Uses cascading fallback to find best-matching components while handling
    sparse combinations gracefully.
    """

    def __init__(
        self,
        anno_dir: str,
        image_dir: str,
        edge_margin: int = 5,
        max_cache: int = 300,
    ):
        self.image_dir = image_dir
        self.max_cache = max_cache
        self._img_cache: Dict[str, Image.Image] = {}

        # Multi-level indices
        self.by_cat: Dict[str, List[ComponentEntry]] = defaultdict(list)
        self.by_cat_color: Dict[Tuple[str, str], List[ComponentEntry]] = defaultdict(list)
        self.by_cat_res: Dict[Tuple[str, str], List[ComponentEntry]] = defaultdict(list)
        self.by_cat_color_res: Dict[Tuple[str, str, str], List[ComponentEntry]] = defaultdict(list)
        self.by_cat_color_res_orient: Dict[Tuple[str, str, str, str], List[ComponentEntry]] = defaultdict(list)

        skipped_edge = 0
        total = 0
        boards_loaded = 0
        boards_missing_image = 0
        boards_empty = 0
        color_counts = defaultdict(int)
        res_counts = defaultdict(int)

        anno_files = sorted(Path(anno_dir).glob("*.json"))
        print(f"[ComponentBankV2_new] Loading from {anno_dir} ({len(anno_files)} boards)...")

        for anno_path in anno_files:
            board_name = anno_path.stem

            img_path = os.path.join(image_dir, f"{board_name}.png")
            if not os.path.exists(img_path):
                boards_missing_image += 1
                continue

            with open(anno_path) as f:
                data = json.load(f)

            board_color = data.get("board_color", "green")
            resolution_class = data.get("resolution_class", "R3")

            img_info = data["images"][0] if data.get("images") else None
            img_w = img_info["width"] if img_info else 1280
            img_h = img_info["height"] if img_info else 720

            annotations = data.get("annotations", [])
            if not annotations:
                boards_empty += 1
                continue

            board_components = 0
            for ann in annotations:
                cat_id = ann.get("category_id")
                cat_name = CAT_ID_TO_NAME.get(cat_id)
                if cat_name is None:
                    continue

                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue

                if (x < edge_margin or y < edge_margin or
                        x + w > img_w - edge_margin or
                        y + h > img_h - edge_margin):
                    skipped_edge += 1
                    continue

                orientation = ann.get("orientation", 0)
                entry = ComponentEntry(
                    board_name, (x, y, w, h), cat_name,
                    board_color, resolution_class, orientation,
                )

                ob = orient_bucket(orientation)
                self.by_cat[cat_name].append(entry)
                self.by_cat_color[(cat_name, board_color)].append(entry)
                self.by_cat_res[(cat_name, resolution_class)].append(entry)
                self.by_cat_color_res[(cat_name, board_color, resolution_class)].append(entry)
                self.by_cat_color_res_orient[(cat_name, board_color, resolution_class, ob)].append(entry)

                total += 1
                board_components += 1

            if board_components > 0:
                boards_loaded += 1
                color_counts[board_color] += board_components
                res_counts[resolution_class] += board_components

        print(f"[ComponentBankV2_new] Loaded {total} components from {boards_loaded} boards")
        if boards_missing_image:
            print(f"  Skipped {boards_missing_image} boards (image not found)")
        if boards_empty:
            print(f"  Skipped {boards_empty} boards (0 annotations)")
        print(f"  Skipped {skipped_edge} edge components (margin={edge_margin}px)")
        for cat in sorted(self.by_cat.keys()):
            print(f"  {cat}: {len(self.by_cat[cat])}")
        print("  Color breakdown:")
        for color, count in sorted(color_counts.items()):
            print(f"    {color}: {count}")
        print("  Resolution breakdown:")
        for rc in RES_ORDER:
            if rc in res_counts:
                print(f"    {rc}: {res_counts[rc]}")

    def _get_board_image(self, board_name: str) -> Optional[Image.Image]:
        if board_name not in self._img_cache:
            path = os.path.join(self.image_dir, f"{board_name}.png")
            if not os.path.exists(path):
                return None
            img = Image.open(path)
            if img.mode in ("RGBA", "PA", "P"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.convert("RGBA").split()[3])
                img = bg
            else:
                img = img.convert("RGB")
            self._img_cache[board_name] = img
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
        resolution_class: Optional[str] = None,
        orientation: Optional[int] = None,
        top_k: int = 10,
        size_thresh: float = 0.5,
        exclude_board: Optional[str] = None,
    ) -> Optional[ComponentEntry]:
        """
        Find a matching component using cascading fallback.

        Priority: orientation (never dropped) > color > resolution.
        For rare orientations (not 0/90), fall back to 0 or 90 bucket.
        """
        ob = orient_bucket(orientation) if orientation is not None else None

        # Build candidate list via cascading fallback
        # Orientation is NEVER dropped — color and resolution can be relaxed
        candidates = None

        if ob:
            # L0: cat + color + res + orient (best)
            if board_color and resolution_class:
                pool = self.by_cat_color_res_orient.get((category, board_color, resolution_class, ob), [])
                if len(pool) >= top_k:
                    candidates = pool

            # L1: cat + color + orient (drop res)
            if candidates is None and board_color:
                pool = [e for e in self.by_cat_color.get((category, board_color), [])
                        if orient_bucket(e.orientation) == ob]
                if len(pool) >= top_k:
                    candidates = pool

            # L2: cat + res + orient (drop color)
            if candidates is None and resolution_class:
                pool = [e for e in self.by_cat_res.get((category, resolution_class), [])
                        if orient_bucket(e.orientation) == ob]
                if len(pool) >= top_k:
                    candidates = pool

            # L3: cat + orient (drop color + res)
            if candidates is None:
                pool = [e for e in self.by_cat.get(category, [])
                        if orient_bucket(e.orientation) == ob]
                if len(pool) >= top_k:
                    candidates = pool

            # L4-L6: rare orient fallback — try "0" or "90" instead
            if candidates is None and ob == "other":
                for fallback_ob in ("0", "90"):
                    if board_color and resolution_class:
                        pool = self.by_cat_color_res_orient.get(
                            (category, board_color, resolution_class, fallback_ob), [])
                        if len(pool) >= top_k:
                            candidates = pool
                            break
                    if candidates is None and board_color:
                        pool = [e for e in self.by_cat_color.get((category, board_color), [])
                                if orient_bucket(e.orientation) == fallback_ob]
                        if len(pool) >= top_k:
                            candidates = pool
                            break
                    if candidates is None:
                        pool = [e for e in self.by_cat.get(category, [])
                                if orient_bucket(e.orientation) == fallback_ob]
                        if len(pool) >= top_k:
                            candidates = pool
                            break

        # No orientation info — fall back to cat + color + res without orient
        if candidates is None:
            if board_color and resolution_class:
                pool = self.by_cat_color_res.get((category, board_color, resolution_class), [])
                if len(pool) >= top_k:
                    candidates = pool
            if candidates is None and board_color:
                pool = self.by_cat_color.get((category, board_color), [])
                if len(pool) >= top_k:
                    candidates = pool
            if candidates is None:
                candidates = self.by_cat.get(category, [])

        if not candidates:
            return None

        target_area = target_w * target_h
        target_ar = target_w / target_h if target_h > 0 else 1.0

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

        # Rank by AR similarity
        filtered.sort(key=lambda e: abs(target_ar - e.ar))

        return random.choice(filtered[:top_k]) if filtered else None

    def load_crop(
        self,
        entry: ComponentEntry,
        target_w: int,
        target_h: int,
        resize_jitter: float = 0.0,
    ) -> Optional[Image.Image]:
        """Load and resize a component crop from its source board."""
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
    """Filter and clip annotations to a square crop window."""
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
            "orientation": ann.get("orientation", 0),
        })

    return result
