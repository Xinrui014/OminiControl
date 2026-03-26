import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, generate


class PCBHarmonizeDataset(Dataset):
    """
    Dataset for PCB harmonization (image-conditional generation).

    condition_0 : composite image  — component crops pasted on a white canvas
    image       : real PCB image   — ground-truth target
    description : per-sample component-count prompt from the manifest
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        condition_size=(512, 512),
        target_size=(512, 512),
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
    ):
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.data_root = data_root
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _abs(self, rel_path: str) -> str:
        """Resolve path relative to data_root if not already absolute."""
        return rel_path if os.path.isabs(rel_path) else os.path.join(self.data_root, rel_path)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        real_img = Image.open(self._abs(sample["real"])).convert("RGB").resize(self.target_size)
        comp_img = Image.open(self._abs(sample["composite"])).convert("RGB").resize(self.condition_size)

        # Per-sample prompt from manifest (e.g. "A PCB board with 10 capacitors, ...")
        description = sample.get("prompt", "PCB board, realistic, green soldermask, electronic components")

        # Classifier-free guidance dropout
        if random.random() < self.drop_text_prob:
            description = ""
        if random.random() < self.drop_image_prob:
            comp_img = Image.new("RGB", self.condition_size, (255, 255, 255))

        return {
            "image": self.to_tensor(real_img),
            "condition_0": self.to_tensor(comp_img),
            "condition_type_0": "pcb_harmonize",
            "position_delta_0": np.array([0, 0]),   # condition aligned with target
            "description": description,
        }


@torch.no_grad()
def test_function(model, save_path, file_name):
    """
    Generates val samples: composite | generated | real  (side-by-side).
    Called by TrainingCallback at sample_interval steps.
    """
    os.makedirs(save_path, exist_ok=True)

    data_root = model.training_config["dataset"]["data_root"]
    val_manifest = model.training_config["dataset"]["val_manifest"]
    target_size = tuple(model.training_config["dataset"]["target_size"])
    condition_size = tuple(model.training_config["dataset"]["condition_size"])
    adapter = model.adapter_names[2]

    with open(val_manifest) as f:
        val_samples = json.load(f)

    # Use 4 fixed val samples for consistent visual tracking
    test_indices = [0, 50, 100, 200]

    for i, idx in enumerate(test_indices):
        if idx >= len(val_samples):
            break
        sample = val_samples[idx]

        def _abs(p):
            return p if os.path.isabs(p) else os.path.join(data_root, p)

        comp_img = Image.open(_abs(sample["composite"])).convert("RGB").resize(condition_size)
        real_img = Image.open(_abs(sample["real"])).convert("RGB").resize(target_size)
        prompt = sample.get("prompt", "PCB board, realistic, green soldermask, electronic components")

        condition = Condition(comp_img, adapter, position_delta=np.array([0, 0]))

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

        # Side-by-side: composite | generated | real
        W, H = target_size
        canvas = Image.new("RGB", (W * 3, H))
        canvas.paste(comp_img.resize((W, H)), (0, 0))
        canvas.paste(gen_img, (W, 0))
        canvas.paste(real_img, (W * 2, 0))

        out_path = os.path.join(save_path, f"{file_name}_sample{i}.jpg")
        canvas.save(out_path)
        print(f"  Saved sample: {out_path}")


def main():
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    dataset = PCBHarmonizeDataset(
        manifest_path=training_config["dataset"]["train_manifest"],
        data_root=training_config["dataset"]["data_root"],
        condition_size=tuple(training_config["dataset"]["condition_size"]),
        target_size=tuple(training_config["dataset"]["target_size"]),
        drop_text_prob=training_config["dataset"]["drop_text_prob"],
        drop_image_prob=training_config["dataset"]["drop_image_prob"],
    )

    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
