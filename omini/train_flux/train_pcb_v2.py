"""
PCB Harmonization v2 Training — On-the-fly composite pasting.

Instead of loading pre-built composite images, this dataset:
1. Picks a random 512×512 crop from a 1280×720 board
2. Builds a composite on-the-fly by matching + pasting components from the pool
3. Trains OminiControl to map composite → real PCB patch

Key improvements over v1:
- Random crops (not fixed 2×2 grid) → massive augmentation
- Color-matched component pasting
- Random resize jitter on pasted components
- Color-conditioned prompts ("green/red/blue soldermask")
- Multiple prompt templates for diversity
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

# Import from our component bank module (placed in OminiControl root)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from component_bank_v2 import ComponentBankV2, get_annotations_in_crop, CAT_ID_TO_NAME


# ---------------------------------------------------------------------------
# Prompt templates — diverse descriptions for generalizability
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
            # Simple pluralization
            plural = cat + "s" if not cat.endswith("s") else cat + "es"
            parts.append(f"{count} {plural}")
        else:
            parts.append(f"1 {cat}")
    comp_list = ", ".join(parts[:4])  # Limit to top 4 categories

    template = random.choice(PROMPT_TEMPLATES)
    return template.format(color=color, n=n, comp_list=comp_list)


class PCBHarmonizeDatasetV2(Dataset):
    """
    On-the-fly composite pasting dataset for v2.1 harmonization training.

    Each sample:
    1. Pick a board from v2 train set
    2. Sample a random crop location (512×512 or zoom crop for small components)
    3. Build composite by pasting matched components from the pool
    4. Return (composite, real_patch, prompt, component_weight_mask)

    v2.1 additions:
    - Multi-scale crops: 40% chance of 256×256 zoom crop (upscaled to 512×512)
      to improve small component detail
    - Component-aware loss mask: higher weight on component regions in latent space
    """

    def __init__(
        self,
        v2_jsonl: str,
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
        component_bank: ComponentBankV2 = None,
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

        # Load v2 board list with metadata
        self.boards = []
        with open(v2_jsonl) as f:
            for line in f:
                entry = json.loads(line)
                meta = entry["_meta"]
                self.boards.append({
                    "name": meta["image"],
                    "color": meta.get("color", "green"),
                    "resolution": meta.get("resolution", "R3"),
                    "num_components": meta.get("num_components", 0),
                })

        # Pre-load all annotations
        self._annotations = {}
        for board in self.boards:
            anno_path = os.path.join(anno_dir, f"{board['name']}.json")
            if os.path.exists(anno_path):
                with open(anno_path) as f:
                    data = json.load(f)
                self._annotations[board["name"]] = data.get("annotations", [])

        # Build sample list: (board_idx, crop_idx) pairs
        # We pre-compute crop_idx but actual crop positions are random each epoch
        self.samples = []
        for i, board in enumerate(self.boards):
            if board["name"] in self._annotations:
                for j in range(crops_per_board):
                    self.samples.append((i, j))

        print(f"[DatasetV2] {len(self.boards)} boards, {len(self.samples)} samples "
              f"({crops_per_board} crops/board)")

    def __len__(self):
        return len(self.samples)

    def _random_crop_position(self, img_w: int = 1280, img_h: int = 720,
                              crop_w: int = 512, crop_h: int = 512) -> Tuple[int, int]:
        """Random top-left corner for a crop within the image."""
        x = random.randint(0, max(0, img_w - crop_w))
        y = random.randint(0, max(0, img_h - crop_h))
        return x, y

    def _build_latent_weight_mask(self, annotations, crop_w, crop_h):
        """Build a weight mask in FLUX packed-latent space for component-aware loss.

        FLUX VAE: 512×512 → 64×64 latents, then packed into 2×2 patches → 32×32 tokens.
        Each token covers a 16×16 pixel region in the original image.
        For zoom crops (256→512), coordinates are already in the upscaled 512 space.
        """
        # Latent grid before packing: 64×64 (each cell = 8×8 pixels)
        # After 2×2 packing: 32×32 tokens (each token = 16×16 pixels)
        token_grid_h = crop_h // 16
        token_grid_w = crop_w // 16
        mask = np.ones((token_grid_h, token_grid_w), dtype=np.float32)

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            # Map pixel coords to token grid
            tx1 = max(0, int(x / 16))
            ty1 = max(0, int(y / 16))
            tx2 = min(token_grid_w, int((x + w) / 16) + 1)
            ty2 = min(token_grid_h, int((y + h) / 16) + 1)
            mask[ty1:ty2, tx1:tx2] = self.component_loss_weight

        # Flatten to match packed latent sequence: (32*32,) = (1024,)
        return mask.flatten()

    def __getitem__(self, idx):
        board_idx, crop_idx = self.samples[idx]
        board = self.boards[board_idx]
        board_name = board["name"]
        board_color = board["color"]

        target_w, target_h = self.target_size

        # Load board image
        img_path = os.path.join(self.image_dir, f"{board_name}.png")
        board_img = Image.open(img_path).convert("RGB")
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

        if use_zoom:
            # Upscale both real patch and annotations to 512×512
            scale = target_w / crop_size  # 512/256 = 2.0
            real_patch = real_patch.resize((target_w, target_h), Image.LANCZOS)
            # Scale annotation bboxes to match upscaled coordinates
            crop_annotations = [
                {**ann, "bbox": (ann["bbox"][0] * scale, ann["bbox"][1] * scale,
                                 ann["bbox"][2] * scale, ann["bbox"][3] * scale)}
                for ann in crop_annotations
            ]

        # Build composite on-the-fly
        if random.random() < self.drop_image_prob:
            # Condition dropout — blank white canvas
            composite = Image.new("RGB", self.condition_size, (255, 255, 255))
        else:
            composite = self._build_composite(
                crop_annotations, board_name, board_color, target_w, target_h
            )

        # Build component-aware loss weight mask
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
        width: int,
        height: int,
    ) -> Image.Image:
        """Build a composite image by pasting matched components onto white canvas."""
        canvas = Image.new("RGB", (width, height), (255, 255, 255))

        for ann in annotations:
            cat_name = ann["category_name"]
            rx, ry, rw, rh = ann["bbox"]
            rw, rh = int(rw), int(rh)

            if rw < 3 or rh < 3:
                continue

            # Find a matching component from the pool
            match = self.bank.find_match(
                category=cat_name,
                target_w=rw,
                target_h=rh,
                board_color=board_color,
                exclude_board=exclude_board,
            )

            if match is None:
                continue

            # Load and resize crop (with jitter)
            crop = self.bank.load_crop(
                match, rw, rh, resize_jitter=self.resize_jitter
            )

            if crop is None:
                continue

            # Paste at annotation position (clip to canvas)
            px = max(0, min(int(rx), width - crop.width))
            py = max(0, min(int(ry), height - crop.height))
            canvas.paste(crop, (px, py))

        return canvas


@torch.no_grad()
def test_function(model, save_path, file_name):
    """
    Generate val samples for visual tracking during training.
    Shows: composite | generated | real (side-by-side).
    """
    os.makedirs(save_path, exist_ok=True)

    config = model.training_config
    data_root = config["dataset"]["image_dir"]
    anno_dir = config["dataset"]["anno_dir"]
    v2_jsonl = config["dataset"]["v2_jsonl"]
    target_size = tuple(config["dataset"]["target_size"])
    condition_size = tuple(config["dataset"]["condition_size"])
    adapter = model.adapter_names[2]

    # Load a few val boards for consistent tracking
    val_boards = []
    v2_test = config["dataset"].get("v2_test_jsonl")
    if v2_test and os.path.exists(v2_test):
        with open(v2_test) as f:
            for i, line in enumerate(f):
                if i >= 8:
                    break
                entry = json.loads(line)
                val_boards.append(entry["_meta"])

    if not val_boards:
        print("  No val data for sampling, skipping.")
        return

    for i, meta in enumerate(val_boards[:4]):
        board_name = meta["image"]
        img_path = os.path.join(data_root, f"{board_name}.png")
        if not os.path.exists(img_path):
            continue

        board_img = Image.open(img_path).convert("RGB")

        # Fixed crop position for consistent tracking
        random.seed(42 + i)
        cx = random.randint(0, max(0, board_img.width - target_size[0]))
        cy = random.randint(0, max(0, board_img.height - target_size[1]))

        real_patch = board_img.crop((cx, cy, cx + target_size[0], cy + target_size[1]))

        # Load annotations and build composite
        anno_path = os.path.join(anno_dir, f"{board_name}.json")
        if not os.path.exists(anno_path):
            continue
        with open(anno_path) as f:
            data = json.load(f)
        crop_anns = get_annotations_in_crop(data.get("annotations", []), cx, cy, target_size[0])

        # Simple composite for val (no bank needed — just use the real patch as condition)
        prompt = make_prompt(meta.get("color", "green"), crop_anns)
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
    print("Building ComponentBankV2...")
    bank = ComponentBankV2(
        anno_dir=dataset_cfg["anno_dir"],
        image_dir=dataset_cfg["image_dir"],
        edge_margin=dataset_cfg.get("edge_margin", 5),
    )

    # Build dataset
    dataset = PCBHarmonizeDatasetV2(
        v2_jsonl=dataset_cfg["v2_jsonl"],
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
