"""Evaluate current model state on fixed val set.

For each val sample: sample I_fine (no grad, delta=1), compute reward.
Average reward over noise seeds for stable estimate.

Returns dict of metrics for logging / ckpt selection.
Optionally saves a few generated images for visual inspection.
"""
import os, sys
from typing import Dict, List, Optional
import numpy as np
import torch

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from omini.pipeline.flux_omini import Condition
from omini.train_flux.flux_sample_with_grad import (
    flux_sample_with_grad, prepare_condition_data
)
from omini.train_flux.reward_dino import CAT_NAMES
from omini.train_flux.alignprop_step import _set_adapter_scale


def _save_image_png(image_tensor, path):
    """image_tensor: (1, 3, H, W) in [-1, 1] bf16/fp32 → PNG."""
    from torchvision.utils import save_image
    img = (image_tensor.float() + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    save_image(img, path)


def eval_on_val(
    pipe, val_samples, reward_model,
    delta_adapter_name="delta",
    height=1024, width=1024,
    num_inference_steps=10,
    eval_noise_seeds=(42, 43, 44),
    image_out_dir: Optional[str] = None,
    n_images_to_save: int = 0,
) -> Dict[str, float]:
    """Evaluate current delta LoRA on val set. Returns avg reward + per-class.

    If image_out_dir given and n_images_to_save > 0: save that many generated
    images (first n samples × first seed) as PNGs for visual inspection.
    """
    device = next(pipe.transformer.parameters()).device
    _set_adapter_scale(pipe, delta_adapter_name, 1.0)

    all_rewards = []
    per_class_all: Dict[str, List[float]] = {}

    if image_out_dir:
        os.makedirs(image_out_dir, exist_ok=True)

    for si, sample in enumerate(val_samples):
        composite = sample["composite"]
        prompt = sample["prompt"]
        bboxes = sample["bboxes"]
        classes = sample["classes"]

        pipe.text_encoder.to(device); pipe.text_encoder_2.to(device)
        cond = Condition(composite, adapter_setting="pcb_harmonize")
        cond_data = prepare_condition_data(pipe, [cond])
        pe, pool, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=device,
                                         num_images_per_prompt=1, max_sequence_length=512)
        pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

        sample_rewards = []
        for seed_idx, seed in enumerate(eval_noise_seeds):
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                image = flux_sample_with_grad(
                    pipe, prompt_embeds=pe, pooled_prompt_embeds=pool,
                    condition_data=cond_data, main_adapter=delta_adapter_name,
                    height=height, width=width,
                    num_inference_steps=num_inference_steps, k_grad_steps=1,
                    generator=gen, vae_checkpoint=False,
                )
                r, pc, _pc2 = reward_model(
                    image, bboxes, classes,
                    return_per_class=True, return_per_component=False,
                )
            sample_rewards.append(r.item())
            for k, v in pc.items():
                per_class_all.setdefault(k, []).append(v)

            # Save sample image (first seed, first n samples)
            if image_out_dir and si < n_images_to_save and seed_idx == 0:
                _save_image_png(image, os.path.join(image_out_dir, f"sample_{si:02d}_gen.png"))
                # Also save composite for reference (once per sample)
                comp_path = os.path.join(image_out_dir, f"sample_{si:02d}_composite.png")
                if not os.path.exists(comp_path):
                    composite.save(comp_path)

        all_rewards.append(float(np.mean(sample_rewards)))

    result = {
        "val_reward_mean": float(np.mean(all_rewards)),
        "val_reward_std":  float(np.std(all_rewards)),
        "val_n_samples":   len(all_rewards),
    }
    for k, vs in per_class_all.items():
        result[f"val_reward_{k}"] = float(np.mean(vs))
    return result
