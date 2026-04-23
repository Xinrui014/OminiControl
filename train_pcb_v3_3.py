"""
PCB Harmonization v3.3 — Refined v3.2 (ComponentBankV2.1) with bank + dataloader bug fixes.

Refinements over v3.2 (v4_subclass):
- Crop scheme: 60% 512-native → LANCZOS → 1024 (matches inference distribution),
  40% 1024-native crop on a 1280x1280 WHITE-padded board (random pad placement,
  preserves native PCB detail with no scaling).
- ComponentBank matching uses the FULL component bbox (original_bbox), not the
  crop-clipped bbox — edge-clipped components no longer get aspect-distorted pastes.
- On find_match == None OR load_crop == None: paste real pixels from source image
  at the clipped bbox (no LANCZOS squeeze, no hallucination training).
- ComponentBankV2.1 cardinal rotations (90/180/270) use Image.transpose (bit-exact)
  instead of BILINEAR (see lib/component_bank_v2_1.py).

Upstream trainer (omini/train_flux/trainer.py) must be at the refined version:
- Weighted loss is sum/sum-normalized so component_loss_weight only controls
  relative weighting.
- DataLoader has worker_init_fn that seeds Python/numpy RNG per worker.
"""
import json
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from omini.train_flux.trainer import OminiModel, get_config, train
from omini.pipeline.flux_omini import Condition, generate

from lib.component_bank_v2_1 import (
    ComponentBankV2_1, get_annotations_in_crop, CAT_ID_TO_NAME, parse_resolution_class,
)


# ---------------------------------------------------------------------------
# Prompt templates — same as v3.1
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    "A high-quality photograph of a printed circuit board with {color} soldermask, copper traces, and electronic components",
    "A realistic PCB with {color} soldermask showing {n} electronic components including {comp_list}",
    "A {color} printed circuit board with visible solder joints, traces, and {n} components",
    "An electronic circuit board with {color} substrate, copper routing, and surface-mount components",
    "A close-up of a {color} PCB populated with {comp_list}",
    "A detailed view of a printed circuit board featuring {color} soldermask and {n} components",
    "A {color} circuit board with electronic components, solder pads, and copper traces",
    "PCB board with {color} soldermask, {n} components including {comp_list}",
]

COLOR_NAMES = {
    "green": "green",
    "red": "red",
    "blue": "blue",
    "black": "black",
    "white": "white",
}


def make_prompt(board_color: str, annotations: list) -> str:
    """Generate a diverse prompt from board metadata and annotations."""
    color = COLOR_NAMES.get(board_color, "green")
    n = len(annotations)

    cat_counts = {}
    for ann in annotations:
        name = ann["category_name"].lower()
        cat_counts[name] = cat_counts.get(name, 0) + 1

    parts = []
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        if count > 1:
            plural = cat + "s" if not cat.endswith("s") else cat + "es"
            parts.append(f"{count} {plural}")
        else:
            parts.append(f"1 {cat}")
    comp_list = ", ".join(parts[:4])

    template = random.choice(PROMPT_TEMPLATES)
    return template.format(color=color, n=n, comp_list=comp_list)


class PCBHarmonizeDatasetV3_3(Dataset):
    """
    On-the-fly composite pasting dataset with ComponentBankV2.1 (v3.3 refined).

    Two-branch crop scheme (matches inference + preserves native detail):
    - native_crop_prob (default 0.4): pad board to native_pad_size×native_pad_size
      with WHITE, random pad placement, random crop at target_size (1024 native).
      The top ~70% of each crop is real PCB detail at native resolution; the
      remaining ~30% is white padding at a randomized position.
    - 1 - native_crop_prob (default 0.6): crop at upscale_crop_size (default 512)
      directly from the board, LANCZOS → target_size (1024). Exactly matches the
      inference composite pipeline (composite@512 → upscale → 1024 gen).
    """

    def __init__(
        self,
        anno_dir: str,
        image_dir: str,
        condition_size: Tuple[int, int] = (1024, 1024),
        target_size: Tuple[int, int] = (1024, 1024),
        crops_per_board: int = 10,
        resize_jitter: float = 0.15,
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        min_visible_ratio: float = 0.5,
        min_components: int = 2,
        component_bank: ComponentBankV2_1 = None,
        native_crop_prob: float = 0.4,
        upscale_crop_size: int = 512,
        native_pad_size: int = 1280,
        pad_color: Tuple[int, int, int] = (255, 255, 255),
        component_loss_weight: float = 3.0,
    ):
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.condition_size = condition_size
        self.target_size = target_size
        self.resize_jitter = resize_jitter
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.min_visible_ratio = min_visible_ratio
        self.min_components = min_components
        self.bank = component_bank
        self.to_tensor = T.ToTensor()
        self.native_crop_prob = native_crop_prob
        self.upscale_crop_size = upscale_crop_size
        self.native_pad_size = native_pad_size
        self.pad_color = pad_color
        self.component_loss_weight = component_loss_weight

        # Load board list from annotation dir
        self.boards = []
        self._annotations = {}

        anno_files = sorted(Path(anno_dir).glob("*.json"))
        print(f"[DatasetV3_3] Loading boards from {anno_dir} ({len(anno_files)} files)...")

        for anno_path in anno_files:
            board_name = anno_path.stem
            img_path = os.path.join(image_dir, f"{board_name}.png")
            if not os.path.exists(img_path):
                continue

            with open(anno_path) as f:
                data = json.load(f)

            annotations = data.get("annotations", [])
            if not annotations:
                continue

            # Resolve category names
            for ann in annotations:
                if "category_name" not in ann:
                    ann["category_name"] = CAT_ID_TO_NAME.get(ann.get("category_id"), "unknown")

            # v2.2 has some JSONs with explicit `board_color: null` — coerce None → default
            self.boards.append({
                "name": board_name,
                "color": data.get("board_color") or "green",
                "resolution_class": parse_resolution_class(
                    data.get("resolution_class") or "R3"
                ),
            })
            self._annotations[board_name] = annotations

        # Build sample list: (board_idx, crop_idx) pairs
        self.samples = []
        for i, board in enumerate(self.boards):
            for j in range(crops_per_board):
                self.samples.append((i, j))

        print(f"[DatasetV3_3] {len(self.boards)} boards, {len(self.samples)} samples "
              f"({crops_per_board} crops/board)")
        print(f"[DatasetV3_3] native_crop_prob={self.native_crop_prob} "
              f"upscale_crop_size={self.upscale_crop_size} "
              f"native_pad_size={self.native_pad_size}")

    def __len__(self):
        return len(self.samples)

    def _random_crop_position(self, img_w: int, img_h: int,
                              crop_w: int, crop_h: int) -> Tuple[int, int]:
        x = random.randint(0, max(0, img_w - crop_w))
        y = random.randint(0, max(0, img_h - crop_h))
        return x, y

    def _build_latent_weight_mask(self, annotations, crop_w, crop_h):
        """Build weight mask in FLUX packed-latent space for component-aware loss."""
        token_grid_h = crop_h // 16
        token_grid_w = crop_w // 16
        mask = np.ones((token_grid_h, token_grid_w), dtype=np.float32)

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            tx1 = max(0, int(x / 16))
            ty1 = max(0, int(y / 16))
            tx2 = min(token_grid_w, int((x + w) / 16) + 1)
            ty2 = min(token_grid_h, int((y + h) / 16) + 1)
            mask[ty1:ty2, tx1:tx2] = self.component_loss_weight

        return mask.flatten()

    def _pad_board(self, board_img, all_annotations):
        """Pad a 1280x720ish board with white to a square big enough for a full
        target-size crop with random placement. Returns (padded_img, shifted_anns,
        pad_w, pad_h). Annotations' bbox coords are shifted by (x_offset, y_offset)
        so they refer to positions in the padded image."""
        img_w, img_h = board_img.size
        tgt_w, tgt_h = self.target_size
        pad_w = max(self.native_pad_size, img_w, tgt_w)
        pad_h = max(self.native_pad_size, img_h, tgt_h)

        x_offset = random.randint(0, pad_w - img_w)
        y_offset = random.randint(0, pad_h - img_h)

        padded = Image.new("RGB", (pad_w, pad_h), self.pad_color)
        padded.paste(board_img, (x_offset, y_offset))

        shifted = []
        for ann in all_annotations:
            bx, by, bw, bh = ann["bbox"]
            shifted.append({**ann, "bbox": (bx + x_offset, by + y_offset, bw, bh)})

        return padded, shifted, pad_w, pad_h

    def __getitem__(self, idx):
        board_idx, crop_idx = self.samples[idx]
        board = self.boards[board_idx]
        board_name = board["name"]
        board_color = board["color"]
        resolution_class = board["resolution_class"]

        target_w, target_h = self.target_size

        # Load raw board (RGBA-safe)
        img_path = os.path.join(self.image_dir, f"{board_name}.png")
        raw_img = Image.open(img_path)
        if raw_img.mode in ("RGBA", "PA", "P"):
            bg = Image.new("RGB", raw_img.size, (255, 255, 255))
            bg.paste(raw_img, mask=raw_img.convert("RGBA").split()[3])
            raw_img = bg
        else:
            raw_img = raw_img.convert("RGB")

        all_annotations = self._annotations.get(board_name, [])

        # Decide branch — 40% native-pad 1024 / 60% upscale 512→1024
        use_native = random.random() < self.native_crop_prob
        if use_native:
            # Native branch: pad board so target-size fits with random placement
            src_img, src_annotations, img_w, img_h = self._pad_board(
                raw_img, all_annotations,
            )
            crop_size = target_w
        else:
            src_img = raw_img
            src_annotations = all_annotations
            img_w, img_h = src_img.size
            crop_size = self.upscale_crop_size

        # Random crop position with min_components retry (up to 5 attempts)
        for attempt in range(5):
            cx, cy = self._random_crop_position(img_w, img_h, crop_size, crop_size)
            crop_annotations = get_annotations_in_crop(
                src_annotations, cx, cy, crop_size, self.min_visible_ratio,
            )
            if len(crop_annotations) >= self.min_components:
                break

        # Real patch (ground truth target) at native crop resolution
        real_patch = src_img.crop((cx, cy, cx + crop_size, cy + crop_size))

        # Build composite at native crop resolution
        if random.random() < self.drop_image_prob:
            composite = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
        else:
            composite = self._build_composite(
                crop_annotations, board_name, board_color, resolution_class,
                crop_size, crop_size, src_img, cx, cy,
            )

        # Upscale to target size if we cropped at upscale_crop_size (< target)
        if crop_size != target_w:
            real_patch = real_patch.resize((target_w, target_h), Image.LANCZOS)
            composite = composite.resize((target_w, target_h), Image.LANCZOS)
            scale = target_w / crop_size
            crop_annotations = [
                {**ann, "bbox": (ann["bbox"][0] * scale, ann["bbox"][1] * scale,
                                 ann["bbox"][2] * scale, ann["bbox"][3] * scale)}
                for ann in crop_annotations
            ]

        # Component-aware loss weight mask in target-size space
        weight_mask = self._build_latent_weight_mask(
            crop_annotations, target_w, target_h,
        )

        # Prompt
        if random.random() < self.drop_text_prob:
            description = ""
        else:
            description = make_prompt(board_color, crop_annotations)

        return {
            "image": self.to_tensor(real_patch),
            "condition_0": self.to_tensor(composite),
            "condition_type_0": "pcb_harmonize",
            "position_delta_0": np.array([0, 0]),
            "description": description,
            "loss_weight_mask": weight_mask,
        }

    def _build_composite(
        self,
        annotations: list,
        exclude_board: str,
        board_color: str,
        resolution_class: str,
        width: int,
        height: int,
        board_img: Image.Image,
        crop_x: int,
        crop_y: int,
    ) -> Image.Image:
        """Build composite with ComponentBankV2.1 (subclass + orientation rotation).

        Per-annotation flow:
          1. Match bank using FULL component dims (ann['original_bbox']), not the
             crop-clipped dims. Bank sees the true component size; size_thresh is
             a hard floor in the bank (returns None if no size-valid candidate).
          2. If matched: load the reference at full size, then PIL-crop the portion
             corresponding to the visible (clipped) region of the target. Paste
             that partial crop at the clipped location. No aspect squeeze.
          3. If no match (or load failed): fall back to the real pixels from the
             source image at the clipped bbox location. This avoids training on
             blank composites with upweighted loss (hallucination pressure).
        """
        canvas = Image.new("RGB", (width, height), (255, 255, 255))

        for ann in annotations:
            cat_name = ann["category_name"]
            rx, ry, rw, rh = ann["bbox"]
            rw_int, rh_int = int(rw), int(rh)
            if rw_int < 3 or rh_int < 3:
                continue

            # Full bbox (in src image coords) — set by get_annotations_in_crop
            orig_bbox = ann.get("original_bbox")
            if orig_bbox is None:
                # Shouldn't happen in the current pipeline, but guard anyway
                ox, oy, ow, oh = crop_x + rx, crop_y + ry, rw, rh
            else:
                ox, oy, ow, oh = orig_bbox

            orientation = ann.get("orientation", 0)
            sub_class = ann.get("sub_class", -1)

            # Bug 12: match on the FULL component size, not the clipped size
            result = self.bank.find_match(
                category=cat_name,
                sub_class=sub_class,
                target_w=ow,
                target_h=oh,
                board_color=board_color,
                resolution_class=resolution_class,
                orientation=orientation,
                exclude_board=exclude_board,
            )

            crop_to_paste = None

            if result is not None:
                entry, rotation = result
                # Load reference at its FULL target size (no resize_jitter here —
                # we want exact-size output so the visible-portion offset math is clean)
                full_crop = self.bank.load_crop(
                    entry, int(ow), int(oh),
                    rotation=rotation,
                    resize_jitter=0.0,
                )
                if full_crop is not None:
                    # Offset of the visible portion within the full reference.
                    # Full bbox top-left in crop coords: (ox - crop_x, oy - crop_y)
                    # Visible top-left in crop coords:   (rx, ry)
                    fw, fh = full_crop.size
                    off_x = int(rx - (ox - crop_x))
                    off_y = int(ry - (oy - crop_y))
                    off_x = max(0, min(off_x, max(fw - 1, 0)))
                    off_y = max(0, min(off_y, max(fh - 1, 0)))
                    vis_w = max(1, min(rw_int, fw - off_x))
                    vis_h = max(1, min(rh_int, fh - off_y))
                    visible = full_crop.crop(
                        (off_x, off_y, off_x + vis_w, off_y + vis_h)
                    )
                    # Apply resize_jitter on the visible portion (not the full ref),
                    # so size variation augmentation is preserved.
                    if self.resize_jitter > 0:
                        scale = 1.0 + random.uniform(
                            -self.resize_jitter, self.resize_jitter
                        )
                        vis_w_j = max(1, int(vis_w * scale))
                        vis_h_j = max(1, int(vis_h * scale))
                        visible = visible.resize(
                            (vis_w_j, vis_h_j), Image.LANCZOS,
                        )
                    crop_to_paste = visible

            # Bugs 8 + 11 unified self-fallback: real pixels from source image,
            # no LANCZOS squeeze. Used when find_match returns None OR load fails.
            if crop_to_paste is None:
                abs_x = crop_x + int(rx)
                abs_y = crop_y + int(ry)
                crop_to_paste = board_img.crop(
                    (abs_x, abs_y, abs_x + rw_int, abs_y + rh_int)
                )

            if crop_to_paste is None:
                continue

            px = max(0, min(int(rx), width - crop_to_paste.width))
            py = max(0, min(int(ry), height - crop_to_paste.height))
            canvas.paste(crop_to_paste, (px, py))

        return canvas


@torch.no_grad()
def test_function(model, save_path, file_name):
    """Generate val samples for visual tracking during training."""
    os.makedirs(save_path, exist_ok=True)

    config = model.training_config
    data_root = config["dataset"]["image_dir"]
    anno_dir = config["dataset"]["anno_dir"]
    target_size = tuple(config["dataset"]["target_size"])
    adapter = model.adapter_names[2]

    val_anno_dir = config["dataset"].get("val_anno_dir", anno_dir)
    val_files = sorted(Path(val_anno_dir).glob("*.json"))[:8]

    if not val_files:
        print("  No val data for sampling, skipping.")
        return

    for i, anno_path in enumerate(val_files[:4]):
        board_name = anno_path.stem
        img_path = os.path.join(data_root, f"{board_name}.png")
        if not os.path.exists(img_path):
            continue

        board_img = Image.open(img_path).convert("RGB")

        # Fixed crop position for consistent tracking
        random.seed(42 + i)
        cx = random.randint(0, max(0, board_img.width - target_size[0]))
        cy = random.randint(0, max(0, board_img.height - target_size[1]))

        real_patch = board_img.crop((cx, cy, cx + target_size[0], cy + target_size[1]))

        with open(anno_path) as f:
            data = json.load(f)
        crop_anns = get_annotations_in_crop(data.get("annotations", []), cx, cy, target_size[0])

        prompt = make_prompt(data.get("board_color", "green"), crop_anns)
        condition = Condition(real_patch, adapter, position_delta=np.array([0, 0]))

        generator = torch.Generator(device=model.device)
        generator.manual_seed(42 + i)

        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
        )
        gen_img = res.images[0]

        W, H = target_size
        canvas = Image.new("RGB", (W * 3, H))
        canvas.paste(real_patch, (0, 0))
        canvas.paste(gen_img, (W, 0))
        canvas.paste(real_patch, (W * 2, 0))
        out_path = os.path.join(save_path, f"{file_name}_sample{i}.jpg")
        canvas.save(out_path)
        print(f"  Saved sample: {out_path}")


