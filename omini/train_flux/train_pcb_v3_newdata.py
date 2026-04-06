"""
PCB Harmonization v3_newdata — Same as v2.1 but with ComponentBankV2_new.

Changes from v2.1:
- ComponentBankV2_new: orientation-first matching, resolution class, RGBA fix
- Paste at native crop resolution (256 for zoom), resize composite after
- v2 annotations with orientation/resolution_class metadata
- Board list loaded from annotation dir (no v2_jsonl needed)
"""
import json
import os
import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, generate

from lib.component_bank_v2_new import ComponentBankV2_new, get_annotations_in_crop, CAT_ID_TO_NAME


# ---------------------------------------------------------------------------
# Prompt templates — same as v2.1 (descriptive, no layout coords)
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

    # Count categories
    cat_counts = {}
    for ann in annotations:
        name = ann["category_name"].lower()
        cat_counts[name] = cat_counts.get(name, 0) + 1

    # Build component list string
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


class PCBHarmonizeDatasetV3(Dataset):
    """
    On-the-fly composite pasting dataset with ComponentBankV2_new.

    Same as v2.1 except:
    - Uses ComponentBankV2_new (orientation-first matching, resolution class, RGBA fix)
    - Paste at native crop resolution, resize after
    - Board list from annotation dir (no v2_jsonl)
    """

    def __init__(
        self,
        anno_dir: str,
        image_dir: str,
        condition_size: Tuple[int, int] = (512, 512),
        target_size: Tuple[int, int] = (512, 512),
        crops_per_board: int = 10,
        resize_jitter: float = 0.15,
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        min_visible_ratio: float = 0.5,
        min_components: int = 2,
        component_bank: ComponentBankV2_new = None,
        zoom_prob: float = 0.4,
        zoom_crop_size: int = 256,
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
        self.zoom_prob = zoom_prob
        self.zoom_crop_size = zoom_crop_size
        self.component_loss_weight = component_loss_weight

        # Load board list from annotation dir (no v2_jsonl needed)
        self.boards = []
        self._annotations = {}

        from pathlib import Path
        anno_files = sorted(Path(anno_dir).glob("*.json"))
        print(f"[DatasetV3] Loading boards from {anno_dir} ({len(anno_files)} files)...")

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
            cat_map = {}
            for cat in data.get("categories", []):
                cat_map[cat["id"]] = cat["name"]
            for ann in annotations:
                if "category_name" not in ann:
                    ann["category_name"] = CAT_ID_TO_NAME.get(ann.get("category_id"), "unknown")

            self.boards.append({
                "name": board_name,
                "color": data.get("board_color", "green"),
                "resolution_class": data.get("resolution_class", "R3"),
            })
            self._annotations[board_name] = annotations

        # Build sample list: (board_idx, crop_idx) pairs
        self.samples = []
        for i, board in enumerate(self.boards):
            for j in range(crops_per_board):
                self.samples.append((i, j))

        print(f"[DatasetV3] {len(self.boards)} boards, {len(self.samples)} samples "
              f"({crops_per_board} crops/board)")

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

    def __getitem__(self, idx):
        board_idx, crop_idx = self.samples[idx]
        board = self.boards[board_idx]
        board_name = board["name"]
        board_color = board["color"]
        resolution_class = board["resolution_class"]

        target_w, target_h = self.target_size

        # Load board image (RGBA-safe)
        img_path = os.path.join(self.image_dir, f"{board_name}.png")
        board_img = Image.open(img_path)
        if board_img.mode in ("RGBA", "PA", "P"):
            bg = Image.new("RGB", board_img.size, (255, 255, 255))
            bg.paste(board_img, mask=board_img.convert("RGBA").split()[3])
            board_img = bg
        else:
            board_img = board_img.convert("RGB")
        img_w, img_h = board_img.size

        # Get annotations
        all_annotations = self._annotations.get(board_name, [])

        # Decide crop size: zoom (256) or normal (512)
        use_zoom = random.random() < self.zoom_prob
        crop_size = self.zoom_crop_size if use_zoom else target_w

        # Try up to 5 random crop positions to find one with enough components
        for attempt in range(5):
            cx, cy = self._random_crop_position(img_w, img_h, crop_size, crop_size)
            crop_annotations = get_annotations_in_crop(
                all_annotations, cx, cy, crop_size, self.min_visible_ratio
            )
            if len(crop_annotations) >= self.min_components:
                break

        # Real patch (ground truth target)
        real_patch = board_img.crop((cx, cy, cx + crop_size, cy + crop_size))

        # Build composite at NATIVE crop resolution (before any resize)
        if random.random() < self.drop_image_prob:
            composite = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
        else:
            composite = self._build_composite(
                crop_annotations, board_name, board_color, resolution_class,
                crop_size, crop_size
            )

        # Resize to target size after pasting
        if crop_size != target_w:
            real_patch = real_patch.resize((target_w, target_h), Image.LANCZOS)
            composite = composite.resize((target_w, target_h), Image.LANCZOS)
            # Scale annotations for weight mask (needs to be in 512 space)
            scale = target_w / crop_size
            crop_annotations = [
                {**ann, "bbox": (ann["bbox"][0] * scale, ann["bbox"][1] * scale,
                                 ann["bbox"][2] * scale, ann["bbox"][3] * scale)}
                for ann in crop_annotations
            ]

        # Build component-aware loss weight mask (in 512 space)
        weight_mask = self._build_latent_weight_mask(crop_annotations, target_w, target_h)

        # Generate prompt
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
    ) -> Image.Image:
        """Build composite with ComponentBankV2_new (orientation + resolution matching)."""
        canvas = Image.new("RGB", (width, height), (255, 255, 255))

        for ann in annotations:
            cat_name = ann["category_name"]
            rx, ry, rw, rh = ann["bbox"]
            rw, rh = int(rw), int(rh)

            if rw < 3 or rh < 3:
                continue

            orientation = ann.get("orientation", 0)

            match = self.bank.find_match(
                category=cat_name,
                target_w=rw,
                target_h=rh,
                board_color=board_color,
                resolution_class=resolution_class,
                orientation=orientation,
                exclude_board=exclude_board,
            )

            if match is None:
                continue

            crop = self.bank.load_crop(
                match, rw, rh, resize_jitter=self.resize_jitter
            )

            if crop is None:
                continue

            px = max(0, min(int(rx), width - crop.width))
            py = max(0, min(int(ry), height - crop.height))
            canvas.paste(crop, (px, py))

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

    # Load a few val boards from annotation dir
    from pathlib import Path
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


def main():
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    dataset_cfg = training_config["dataset"]

    # Build component bank (shared across workers via fork)
    print("Building ComponentBankV2_new...")
    bank = ComponentBankV2_new(
        anno_dir=dataset_cfg["anno_dir"],
        image_dir=dataset_cfg["image_dir"],
        edge_margin=dataset_cfg.get("edge_margin", 5),
    )

    # Build dataset
    dataset = PCBHarmonizeDatasetV3(
        anno_dir=dataset_cfg["anno_dir"],
        image_dir=dataset_cfg["image_dir"],
        condition_size=tuple(dataset_cfg["condition_size"]),
        target_size=tuple(dataset_cfg["target_size"]),
        crops_per_board=dataset_cfg.get("crops_per_board", 10),
        resize_jitter=dataset_cfg.get("resize_jitter", 0.15),
        drop_text_prob=dataset_cfg.get("drop_text_prob", 0.1),
        drop_image_prob=dataset_cfg.get("drop_image_prob", 0.1),
        min_visible_ratio=dataset_cfg.get("min_visible_ratio", 0.5),
        min_components=dataset_cfg.get("min_components", 2),
        component_bank=bank,
        zoom_prob=dataset_cfg.get("zoom_prob", 0.4),
        zoom_crop_size=dataset_cfg.get("zoom_crop_size", 256),
        component_loss_weight=dataset_cfg.get("component_loss_weight", 3.0),
    )

    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config.get("lora_config"),
        lora_path=training_config.get("lora_path"),
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
