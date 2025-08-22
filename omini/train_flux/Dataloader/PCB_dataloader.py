#!/usr/bin/env python3
# pcb_dataset.py
"""
Turn board-level CSV + image pairs into the same dict format used by HICO.
"""

import os, glob, csv, json, math, random, re, multiprocessing, argparse
from pathlib import Path
from typing import List, Tuple
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image as PImage, Image

VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']

# pcb_classes.py  (or paste into your dataset file)

import re
from typing import Optional


# Build quick look-ups:   keyword → canonical label
_KEYWORDS = {
    "resistor"          : "Resistor",
    "capacitor"         : "Capacitor",
    "inductor"          : "Inductor",
    "connector"         : "Connector",
    "diode"             : "Diode",
    "led"               : "LED",
    "switch"            : "Switch",
    "transistor"        : "Transistor",
    "integrated circuit": "Integrated Circuit",
    "ic"                : "Integrated Circuit",   # special case
    "cat"               : "Cat",  # special case, not a component
}


def normalise_type(raw: str) -> Optional[str]:
    s = raw.strip().lower()
    if not s:
        return None
    # exact match
    if s in _KEYWORDS:
        return _KEYWORDS[s]
    # substring match
    for key, label in _KEYWORDS.items():
        if key in s:
            return label
    return None




# ═══════════════════════════════════════════════════════════════════════
class TIPCBDataset(torch.utils.data.Dataset):
    """
    Returns
    -------
    {
        'id':            board filename stem,
        'image':         FloatTensor   (3, 512, 512)   values ∈ [-1, 1],
        'caption':       "<sob> … <eob>.",
        'subject_boxes': (max_boxes, 4)  normalised to [0,1]  (0-padded),
        'object_boxes':  (max_boxes, 4)  **all zeros** (placeholder),
        'masks':         (max_boxes,)   1 = valid row
    }
    """

    # ──────────────────────────────────────────────────────────────────
    def __init__(
            self,
            img_root: str,
            csv_root: str,
            image_size: int = 1024,
            max_boxes_per_data: int = 100,
            max_images: Optional[int] = None,
            random_crop: bool = False,
            random_flip: bool = True,
            grid_size: int = 16,
            use_patches: bool = True,       # turn on 256×256 patch mode
            patch_size: int = 256,
            save_debug: bool = False,       # dumps test.png
    ):
        super().__init__()
        self.img_root = Path(img_root)
        self.csv_root = Path(csv_root)
        self.image_size = image_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.grid_size = grid_size
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.save_debug = save_debug

        # discover *.csv that have a matching image
        self.items: List[Tuple[Path, Path]] = []
        for csv_path in self.csv_root.glob("*.csv"):
            stem = csv_path.stem
            for ext in VALID_IMAGE_TYPES:
                img_path = self.img_root / f"{stem}{ext}"
                if img_path.exists():
                    self.items.append((img_path, csv_path))
                    break
        if not self.items:
            raise RuntimeError(f"No *.csv + image pairs in {self.csv_root}")

        if self.max_images:
            self.items = self.items[: self.max_images]

        self.start_token = "<sob>"
        self.end_token = "<eob>"

        print(
            f"[PCBDataset] {len(self.items)} boards | "
            f"base={image_size} | patching={use_patches} (patch={patch_size}) | grid={grid_size}×{grid_size}"
        )

    def _crop_to_patch(self, image_tensor, boxes_list, types_list):
        """
        image_tensor: (3, H, W) in [-1,1], H=W=self.image_size
        boxes_list:   list[Tensor(4)] normalized to [0,1] wrt H/W
        types_list:   list[str], same length
        returns: (patch_tensor, new_boxes_list, new_types_list, (px, py))
        """
        C, H, W = image_tensor.shape
        ps = self.patch_size
        assert ps <= H and ps <= W, "patch_size must be <= image_size"

        # choose a random top-left inside the full image
        px = random.randrange(0, W - ps + 1)
        py = random.randrange(0, H - ps + 1)

        # crop image
        patch = image_tensor[:, py:py+ps, px:px+ps]  # (3, ps, ps)

        # remap & clip boxes to patch
        new_boxes, new_types = [], []
        for b, t in zip(boxes_list, types_list):
            # to pixel coords in full image
            x0 = float(b[0] * W); y0 = float(b[1] * H)
            x1 = float(b[2] * W); y1 = float(b[3] * H)

            # intersect with patch [px,px+ps)×[py,py+ps)
            ix0 = max(x0, px);  iy0 = max(y0, py)
            ix1 = min(x1, px+ps); iy1 = min(y1, py+ps)

            if ix1 <= ix0 or iy1 <= iy0:
                continue  # no overlap

            # convert into patch-local normalized coords
            nx0 = (ix0 - px) / ps
            ny0 = (iy0 - py) / ps
            nx1 = (ix1 - px) / ps
            ny1 = (iy1 - py) / ps

            new_boxes.append(torch.tensor([nx0, ny0, nx1, ny1], dtype=torch.float32))
            new_types.append(t)

        return patch, new_boxes, new_types, (px, py)


    def center_crop_arr(self, pil_image, image_size):
        WW, HH = pil_image.size
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=PImage.BOX
            )
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=PImage.BICUBIC
        )
        performed_scale = image_size / min(WW, HH)
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        info = {"performed_scale": performed_scale, 'crop_y': crop_y, 'crop_x': crop_x, "WW": WW, 'HH': HH}
        return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size], info

    def random_crop_arr(self, pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
        min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
        max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
        smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
        while min(*pil_image.size) >= 2 * smaller_dim_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=PImage.BOX
            )
        scale = smaller_dim_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=PImage.BICUBIC
        )
        arr = np.array(pil_image)
        crop_y = random.randrange(arr.shape[0] - image_size + 1)
        crop_x = random.randrange(arr.shape[1] - image_size + 1)
        return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    def transform_image(self, pil_image):
        if self.random_crop:
            arr, info = self.random_crop_arr(pil_image, self.image_size)
        else:
            arr, info = self.center_crop_arr(pil_image, self.image_size)

        info["scale"] = info["performed_scale"]
        info["flip"] = info.get("performed_flip", False)

        info["performed_flip"] = False
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            info["performed_flip"] = True
            info["flip"] = True


        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, (2,0,1))
        return torch.tensor(arr), info

    def recalculate_box_and_verify_if_valid(self, x, y, w, h, tinfo):
        # straight-line version: apply scale + optional flip, clip to frame
        # x0 = x * tinfo["scale"]
        # y0 = y * tinfo["scale"]
        # x1 = (x + w) * tinfo["scale"]
        # y1 = (y + h) * tinfo["scale"]
        x0 = x * (self.image_size / 1024.0)
        y0 = y * (self.image_size / 1024.0)
        x1 = (x + w) * (self.image_size / 1024.0)
        y1 = (y + h) * (self.image_size / 1024.0)

        # 2) subtract the crop offset (so 0,0 in the cropped image lines up)
        x0 -= tinfo["crop_x"]
        x1 -= tinfo["crop_x"]
        y0 -= tinfo["crop_y"]
        y1 -= tinfo["crop_y"]

        if tinfo["flip"]:
            x0, x1 = self.image_size - x1, self.image_size - x0

        # clip
        x0, y0, x1, y1 = map(
            lambda v: max(0, min(self.image_size, v)),
            (x0, y0, x1, y1)
        )
        if (x1 - x0) * (y1 - y0) <= 0:
            return False, (None,)*4
        return True, (x0, y0, x1, y1)

    def _make_natural_caption(self, types_list):
        """
        Given list of component types (strings) in this patch,
        return a natural language description.
        """
        from collections import Counter
        if not types_list:
            return "An empty PCB patch."

        counts = Counter(types_list)

        # map class labels to natural words (optional)
        nice_names = {
            "Resistor": "resistor",
            "Capacitor": "capacitor",
            "Diode": "diode",
            "LED": "LED light",
            "Connector": "connector",
            "Switch": "switch",
            "Integrated Circuit": "integrated circuit chip",
        }

        parts = []
        for comp, n in counts.items():
            word = nice_names.get(comp, comp.lower())
            if n == 1:
                parts.append(f"one {word}")
            else:
                parts.append(f"{n} {word}s")

        obj_str = ", ".join(parts[:-1]) + (" and " + parts[-1] if len(parts) > 1 else parts[0])
        caption = f"A PCB patch containing {obj_str}."
        return caption

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.items)
        # return 1

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        # Load the image and apply crop/flip + record all info in trans_info
        img_path, csv_path = self.items[idx]

        # because the coordinates are stored when image resize to 1024, so resize to 1024 first
        pil_image_rgba = PImage.open(img_path).convert("RGBA")
        ow, oh = pil_image_rgba.size
        scale = 1024 / min(ow, oh)
        nw, nh = int(ow * scale), int(oh * scale)
        image_resized = pil_image_rgba.resize((nw, nh), Image.BILINEAR)
        # 丢弃 alpha 通道
        pil_image = image_resized.convert("RGB")

        image_tensor, trans_info = self.transform_image(pil_image)
        # trans_info contains:
        #   'performed_scale', 'crop_x', 'crop_y', 'WW', 'HH', 'performed_flip'

        # Read the CSV and project every box through trans_info
        boxes, areas, types = [], [], []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[0] != "patch_name":
                f.seek(0);
                reader = csv.reader(f)

            for patch, comp_raw, _ in reader:
                comp_type = normalise_type(comp_raw)
                if comp_type is None:
                    continue

                # parse original coords from filename
                x0, y0, x1, y1 = map(int, re.findall(r"\d+", patch))


                ok, (bx0, by0, bx1, by1) = self.recalculate_box_and_verify_if_valid(
                    x0, y0, x1 - x0, y1 - y0, trans_info
                )
                if not ok:
                    continue

                # store normalised box + area + type
                boxes.append(torch.tensor([bx0, by0, bx1, by1]) / self.image_size)
                areas.append((bx1 - bx0) * (by1 - by0))
                types.append(comp_type)

        # Keep top-K largest by area
        if areas:
            order = torch.tensor(areas).sort(descending=True)[1].tolist()
        else:
            order = []
        order = order[: self.max_boxes_per_data]

        kept_boxes = [boxes[i] for i in order]
        kept_types = [types[i] for i in order]

        # Optional 256×256 patching
        if self.use_patches:
            image_tensor, kept_boxes, kept_types, _ = self._crop_to_patch(image_tensor, kept_boxes, kept_types)

        # Build the outputs
        max_k = self.max_boxes_per_data
        subject_boxes = torch.zeros(max_k, 4)
        object_boxes = torch.zeros(max_k, 4)  # placeholder
        masks = torch.zeros(max_k)
        # cap_tokens = []
        # quant = lambda v: min(int(v * self.grid_size), self.grid_size - 1)

        # Re-rank by area after patch (optional but consistent)
        if kept_boxes:
            areas2 = [float((b[2] - b[0]) * (b[3] - b[1])) for b in kept_boxes]
            order2 = torch.tensor(areas2).sort(descending=True)[1].tolist()
            kept_boxes = [kept_boxes[i] for i in order2[:max_k]]
            kept_types = [kept_types[i] for i in order2[:max_k]]

        for i, b in enumerate(kept_boxes):
            subject_boxes[i] = b
            masks[i] = 1

        # Natural-language caption for this patch
        caption = self._make_natural_caption(kept_types)

        valid_k = int(masks.sum().item())
        boxes_out = subject_boxes[:valid_k].clone()            # (K,4), normalized [0,1]
        box_prompts_out = kept_types[:valid_k]                 # List[str], len K

        # Draw boxes on the image (for debugging purposes)
        if self.save_debug:
            import cv2
            H = image_tensor.shape[1]
            img = image_tensor.numpy().copy()
            img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            img = np.ascontiguousarray(img.transpose(1, 2, 0))  # (H,W,C) RGB

            for box, label in zip(boxes_out, box_prompts_out):
                if box.sum() == 0:
                    continue
                x_min = int(box[0] * H); y_min = int(box[1] * H)
                x_max = int(box[2] * H); y_max = int(box[3] * H)

                # Draw box (red)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                # Draw label (above the box)
                cv2.putText(
                    img,
                    str(label),
                    (x_min, max(y_min - 5, 0)),  # position slightly above the box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,              # font scale
                    (0, 0, 255),      # color (red, in BGR)
                    1,                # thickness
                    cv2.LINE_AA,
                )

            # Convert RGB->BGR before saving
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("test.png", img)


        return {
            # existing fields (kept for backward-compatibility)
            "id": img_path.stem,
            "image": image_tensor,               # 3×IMAGE_SIZE×IMAGE_SIZE, in [-1,1]
            "caption": caption,                  # original caption string
            "subject_boxes": subject_boxes,      # (max_k,4) padded
            "object_boxes": object_boxes,        # (max_k,4) placeholder
            "masks": masks,                      # (max_k,)

            # NEW fields used by the new trainer:
            "description": caption,              # alias so trainer reads a global prompt
            "boxes": boxes_out,                  # (K,4) only valid boxes
            "box_prompts": box_prompts_out,      # List[str], one per box
        }


# ═══════════════════════════════════════════════════════════════════════
#                             testing stub
# ═══════════════════════════════════════════════════════════════════════
def _show_sample(d):
    print(f"ID        : {d['id']}")
    print(f"image     : {tuple(d['image'].shape)}  "
          f"min={float(d['image'].min()):.2f}  max={float(d['image'].max()):.2f}")
    print(f"caption   : {d['caption']}")
    print(f"boxes(K)  : {len(d['boxes'])}  box_prompts(K): {len(d['box_prompts'])}")
    if len(d["boxes"]):
        print(f" first box: {d['boxes'][0].tolist()}, prompt='{d['box_prompts'][0]}'")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser(description="Sanity-check PCBDataset loader")
    ap.add_argument("--img_root", default="/home/xinrui/projects/data/ti_pcb/images_top",
                    help="Folder containing board images (e.g. …/ti_eval_boards)")
    ap.add_argument("--csv_root", default="/home/xinrui/projects/data/ti_pcb/ann_csv_top",
                    help="Folder with annotation CSVs (e.g. …/filtered_masks_annotation_4o)")
    ap.add_argument("-n", type=int, default=1, help="how many samples to show")
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--max_boxes", type=int, default=100)
    ap.add_argument("--no_patches", action="store_true", help="Disable 256×256 patching")
    ap.add_argument("--save_debug", action="store_true", help="Save test.png with boxes")
    args = ap.parse_args()

    ds = TIPCBDataset(img_root=args.img_root,
                      csv_root=args.csv_root,
                      image_size=args.image_size,
                      max_boxes_per_data=args.max_boxes,
                      random_crop=False,
                      random_flip=True,
                      grid_size=16,
                      use_patches=not args.no_patches,
                      patch_size=args.patch_size,
                      save_debug=args.save_debug,)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    for i, batch in enumerate(dl):
        sample = {k: v[0] if isinstance(v, torch.Tensor) else v[0]
                  for k, v in batch.items()}
        _show_sample(sample)
        if i + 1 >= args.n:
            break


if __name__ == "__main__":
    main()