"""
ComponentBankV2.1 — Subclass-aware component matching.

Loads from v2.1_subclass annotations which include per-component `sub_class` field.

Matching rules:
  - (sub_class, color) must match — sub_class implies category
  - Resolution: soft match within ±2 classes (e.g. R3 matches R1–R5)
  - Orientation: rotate all candidates in same group (cardinal/diagonal) to
    target orientation, then size-match on post-rotation dimensions
  - Size: filter area_ratio >= 0.3, rank by AR similarity
  - No match found → use the original component itself (self-fallback)

Orientation groups (CCW rotation):
  Cardinal: {0, 90, 180, 270}
  Diagonal: {45, 135, 225, 315}

Dry-run stats (20k sample on v2.1_subclass):
  99.85% matched, 0.15% self-fallback
  55.4% exact orientation, 44.5% rotated to match
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

CAT_ID_TO_NAME = {
    1: "RESISTOR", 2: "CAPACITOR", 3: "INDUCTOR", 4: "CONNECTOR",
    5: "DIODE", 6: "LED", 7: "SWITCH", 8: "TRANSISTOR", 9: "IC",
    10: "OSCILLATOR", 11: "FUSE",
}
CAT_NAME_TO_ID = {v: k for k, v in CAT_ID_TO_NAME.items()}
CAT_NAME_TO_ID["Integrated Circuit"] = 9
CAT_NAME_TO_ID["Integrated_Circuit"] = 9

# Resolution class ordering for ±2 soft match
RES_ORDER = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
RES_IDX = {r: i for i, r in enumerate(RES_ORDER)}

# Orientation groups — within each group, components can be rotated CCW to match
CARDINAL = frozenset({0, 90, 180, 270})
DIAGONAL = frozenset({45, 135, 225, 315})


def orient_group(angle: int) -> str:
    """Return which rotation group this angle belongs to."""
    if angle in CARDINAL:
        return "cardinal"
    elif angle in DIAGONAL:
        return "diagonal"
    else:
        return "other"


def parse_resolution_class(val) -> str:
    """Handle resolution_class being either a string or a dict."""
    if isinstance(val, dict):
        return val.get("class", "R3")
    if isinstance(val, str):
        return val
    return "R3"


def res_within_range(res_a: str, res_b: str, tolerance: int = 2) -> bool:
    """Check if two resolution classes are within ±tolerance of each other."""
    idx_a = RES_IDX.get(res_a)
    idx_b = RES_IDX.get(res_b)
    if idx_a is None or idx_b is None:
        return True  # unknown res, allow match
    return abs(idx_a - idx_b) <= tolerance


class ComponentEntry:
    """A single component crop reference in the pool."""
    __slots__ = ("board_name", "bbox", "category", "sub_class", "area", "ar",
                 "board_color", "resolution_class", "orientation")

    def __init__(self, board_name: str, bbox: Tuple[float, float, float, float],
                 category: str, sub_class: int, board_color: str,
                 resolution_class: str, orientation: int):
        self.board_name = board_name
        x, y, w, h = bbox
        self.bbox = (x, y, w, h)
        self.category = category
        self.sub_class = sub_class
        self.area = w * h
        self.ar = w / h if h > 0 else 1.0
        self.board_color = board_color
        self.resolution_class = resolution_class
        self.orientation = orientation


class ComponentBankV2_1:
    """
    Component pool with subclass-aware matching.

    Matching:
      1. (sub_class, color) — mandatory
      2. Exclude same board
      3. Resolution — soft, ±2 classes
      4. Rotate all candidates to target orientation (within group)
      5. Size — filter area_ratio >= 0.3, rank by post-rotation AR similarity
      6. Pick random from top_k
      7. No match → return None (caller uses self-fallback)
    """

    def __init__(
        self,
        anno_dir: str,
        image_dir: str,
        edge_margin: int = 5,
        max_cache: int = 300,
        res_tolerance: int = 2,
    ):
        self.image_dir = image_dir
        self.max_cache = max_cache
        self.res_tolerance = res_tolerance
        self._img_cache: Dict[str, Image.Image] = {}

        # Primary index: (category, sub_class, color) → list of entries
        self.by_subclass_color: Dict[Tuple[str, int, str], List[ComponentEntry]] = defaultdict(list)

        skipped_edge = 0
        total = 0
        boards_loaded = 0
        boards_missing_image = 0
        boards_empty = 0
        color_counts = defaultdict(int)
        res_counts = defaultdict(int)
        subclass_counts = defaultdict(int)

        anno_files = sorted(Path(anno_dir).glob("*.json"))
        print(f"[ComponentBankV2.1] Loading from {anno_dir} ({len(anno_files)} boards)...")

        for anno_path in anno_files:
            board_name = anno_path.stem

            img_path = os.path.join(image_dir, f"{board_name}.png")
            if not os.path.exists(img_path):
                boards_missing_image += 1
                continue

            with open(anno_path) as f:
                data = json.load(f)

            # v2.2 has some JSONs with explicit `board_color: null` — coerce None → default
            board_color = data.get("board_color") or "green"
            resolution_class = parse_resolution_class(data.get("resolution_class") or "R3")

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
                sub_class = ann.get("sub_class", -1)

                entry = ComponentEntry(
                    board_name, (x, y, w, h), cat_name, sub_class,
                    board_color, resolution_class, orientation,
                )

                key = (cat_name, sub_class, board_color)
                self.by_subclass_color[key].append(entry)

                total += 1
                board_components += 1
                subclass_counts[(cat_name, sub_class)] += 1

            if board_components > 0:
                boards_loaded += 1
                color_counts[board_color] += board_components
                res_counts[resolution_class] += board_components

        print(f"[ComponentBankV2.1] Loaded {total} components from {boards_loaded} boards")
        if boards_missing_image:
            print(f"  Skipped {boards_missing_image} boards (image not found)")
        if boards_empty:
            print(f"  Skipped {boards_empty} boards (0 annotations)")
        print(f"  Skipped {skipped_edge} edge components (margin={edge_margin}px)")

        cat_totals = defaultdict(int)
        cat_subclasses = defaultdict(set)
        for (cat, sc), count in subclass_counts.items():
            cat_totals[cat] += count
            cat_subclasses[cat].add(sc)
        for cat in sorted(cat_totals.keys()):
            print(f"  {cat}: {cat_totals[cat]} ({len(cat_subclasses[cat])} subclasses)")

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
        sub_class: int,
        target_w: float,
        target_h: float,
        board_color: str,
        resolution_class: str,
        orientation: int = 0,
        top_k: int = 10,
        size_thresh: float = 0.3,
        exclude_board: Optional[str] = None,
    ) -> Optional[Tuple[ComponentEntry, int]]:
        """
        Find a matching component. Returns (entry, rotation_degrees) or None.

        Logic:
          1. Pool = (category, sub_class, color) match, exclude same board
          2. Filter resolution ±2 (relax to full pool if empty)
          3. Rotate all candidates to target orientation (within group),
             compute post-rotation w/h
          4. Filter area_ratio >= size_thresh on post-rotation area
          5. Sort by AR diff (post-rotation AR vs target AR)
          6. Pick random from top_k
          7. No valid candidate → return None (caller does self-fallback)
        """
        key = (category, sub_class, board_color)
        pool = self.by_subclass_color.get(key, [])
        if not pool:
            return None

        # Exclude same board
        candidates = [e for e in pool if e.board_name != exclude_board]
        if not candidates:
            return None

        # Filter resolution ±2
        res_filtered = [e for e in candidates
                        if res_within_range(e.resolution_class, resolution_class,
                                            self.res_tolerance)]
        if not res_filtered:
            res_filtered = candidates

        target_area = target_w * target_h
        target_ar = target_w / target_h if target_h > 0 else 1.0
        tgt_group = orient_group(orientation)

        # Rotate all candidates to target orientation, compute post-rotation size
        scored = []  # (entry, delta, post_ar, area_ratio)
        for e in res_filtered:
            src_group = orient_group(e.orientation)
            if src_group == tgt_group and tgt_group != "other":
                delta = (orientation - e.orientation) % 360
                if delta in (90, 270):
                    post_w, post_h = e.bbox[3], e.bbox[2]  # swap w,h
                else:
                    post_w, post_h = e.bbox[2], e.bbox[3]
            else:
                # Different group or "other" — can't rotate, use as-is
                delta = 0
                post_w, post_h = e.bbox[2], e.bbox[3]

            post_ar = post_w / post_h if post_h > 0 else 1.0
            post_area = post_w * post_h
            area_ratio = min(target_area, post_area) / max(target_area, post_area)
            ar_diff = abs(target_ar - post_ar)
            scored.append((e, delta, ar_diff, area_ratio))

        # Filter by area_ratio >= threshold
        valid = [s for s in scored if s[3] >= size_thresh]
        if not valid:
            return None

        # Sort by AR diff, pick random from top_k
        valid.sort(key=lambda s: s[2])
        chosen = random.choice(valid[:top_k])
        return (chosen[0], chosen[1])

    # Cardinal CCW rotations — exact pixel permutation via Image.transpose.
    # Avoids BILINEAR blur that crop.rotate(...) applies even for 90/180/270.
    _CARDINAL_ROTATE = {
        90: Image.ROTATE_90,
        180: Image.ROTATE_180,
        270: Image.ROTATE_270,
    }

    def load_crop(
        self,
        entry: ComponentEntry,
        target_w: int,
        target_h: int,
        rotation: int = 0,
        resize_jitter: float = 0.0,
    ) -> Optional[Image.Image]:
        """
        Load a component crop, optionally rotate CCW, then resize to target.

        rotation: CCW degrees (0, 90, 180, 270 are exact; diagonals use BILINEAR).
        """
        board = self._get_board_image(entry.board_name)
        if board is None:
            return None

        x, y, w, h = entry.bbox
        crop = board.crop((int(x), int(y), int(x + w), int(y + h)))

        # Apply CCW rotation. For cardinal 90/180/270, use Image.transpose —
        # it is a pure memory rearrangement (bit-exact, no resampling). For
        # diagonal angles (45/135/…), fall back to BILINEAR rotate.
        if rotation in self._CARDINAL_ROTATE:
            crop = crop.transpose(self._CARDINAL_ROTATE[rotation])
        elif rotation != 0:
            crop = crop.rotate(rotation, expand=True, resample=Image.BILINEAR)

        if resize_jitter > 0:
            scale = 1.0 + random.uniform(-resize_jitter, resize_jitter)
            target_w = max(1, int(target_w * scale))
            target_h = max(1, int(target_h * scale))

        return crop.resize((max(target_w, 1), max(target_h, 1)), Image.LANCZOS)

    def load_self_crop(
        self,
        board_name: str,
        bbox: Tuple[float, float, float, float],
        target_w: int,
        target_h: int,
        resize_jitter: float = 0.0,
    ) -> Optional[Image.Image]:
        """Load the original component from its own board (self-fallback)."""
        board = self._get_board_image(board_name)
        if board is None:
            return None

        x, y, w, h = bbox
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
            "sub_class": ann.get("sub_class", -1),
        })

    return result
