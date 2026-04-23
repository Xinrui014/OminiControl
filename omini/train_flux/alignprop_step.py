"""One AlignProp training step: I_fine + I_base + DINO reward + Focus-N-Fix preservation.

Supports gradient accumulation over multiple noise seeds (num_accum > 1) to reduce
single-sample variance in the reward gradient estimate.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from omini.train_flux.flux_sample_with_grad import flux_sample_with_grad
from omini.train_flux.mask_utils import bboxes_to_mask


def _set_adapter_scale(pipe, adapter_name: str, scale: float):
    """Set LoRA scaling for a named adapter across all LoRA-wrapped modules."""
    from peft.tuners.lora.layer import BaseTunerLayer
    for m in pipe.transformer.modules():
        if isinstance(m, BaseTunerLayer):
            if adapter_name in m.scaling:
                m.scaling[adapter_name] = scale


def _one_backward_pass(
    pipe, prompt_embeds, pooled_prompt_embeds, condition_data,
    bboxes, classes, reward_model, delta_adapter_name,
    height, width, num_inference_steps, k_grad_steps,
    lambda_preserve, mask_dilate_px, guidance_scale,
    seed, loss_divisor, device,
    class_weights=None,
):
    """Compute one (I_base, I_fine) -> loss -> backward at a single noise seed.

    Gradients accumulate in LoRA param.grad.  Returns scalar logs.
    """
    # I_base: delta OFF, no_grad. main_adapter=None → specify_lora zeros delta
    # (and v3.4, since this is main branch not cond) on wrapped modules.
    # Explicit _set_adapter_scale handles ff.net.0.proj (unwrapped by specify_lora).
    _set_adapter_scale(pipe, delta_adapter_name, 0.0)
    gen_b = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        image_base = flux_sample_with_grad(
            pipe, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
            condition_data=condition_data, main_adapter=None,
            height=height, width=width,
            num_inference_steps=num_inference_steps, k_grad_steps=1,
            generator=gen_b, guidance_scale=guidance_scale,
            vae_checkpoint=False,
        )
    image_base = image_base.detach()

    # I_fine: delta ON, grad
    _set_adapter_scale(pipe, delta_adapter_name, 1.0)
    gen_f = torch.Generator(device=device).manual_seed(seed)
    image_fine = flux_sample_with_grad(
        pipe, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        condition_data=condition_data, main_adapter=delta_adapter_name,
        height=height, width=width,
        num_inference_steps=num_inference_steps, k_grad_steps=k_grad_steps,
        generator=gen_f, guidance_scale=guidance_scale,
        vae_checkpoint=True,
    )

    reward, per_class, _per_comp = reward_model(
        image_fine, bboxes, classes,
        class_weights=class_weights, return_per_class=True, return_per_component=False,
    )
    mask = bboxes_to_mask(bboxes, H=height, W=width, dilate_px=mask_dilate_px,
                          device=device, dtype=image_fine.dtype)
    delta = image_base - image_fine
    preserv = ((1.0 - mask) * delta).pow(2).mean()

    loss = (-reward + lambda_preserve * preserv) / loss_divisor
    loss.backward()

    return {
        "reward": reward.item(),
        "preserv": preserv.item(),
        "per_class": per_class,
    }


def alignprop_step(
    pipe,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    condition_data: Dict,
    bboxes: List[Tuple[int, int, int, int]],
    classes: List[int],
    reward_model,
    delta_adapter_name: str = "delta",
    height: int = 1024, width: int = 1024,
    num_inference_steps: int = 10,
    k_grad_steps: int = 3,
    lambda_preserve: float = 1.0,
    mask_dilate_px: int = 4,
    guidance_scale: float = 3.5,
    generator: Optional[torch.Generator] = None,
    num_accum: int = 1,
    class_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """One training step (possibly with gradient accumulation).

    If num_accum > 1: runs num_accum separate forward/backward passes with
    different noise seeds. Each backward adds to param.grad; caller does ONE
    opt.step() after. Reduces reward-gradient variance by √N.

    Caller must run opt.zero_grad() BEFORE this and opt.step() AFTER.
    """
    device = next(pipe.transformer.parameters()).device

    # Derive num_accum seeds from generator
    if generator is not None:
        base_seed = int(generator.initial_seed())
    else:
        base_seed = 42
    seeds = [(base_seed + i * 997) & 0xFFFFFFFF for i in range(num_accum)]

    all_rewards, all_preservs = [], []
    all_per_class: Dict[str, List[float]] = {}

    for i, seed in enumerate(seeds):
        log_i = _one_backward_pass(
            pipe, prompt_embeds, pooled_prompt_embeds, condition_data,
            bboxes, classes, reward_model, delta_adapter_name,
            height, width, num_inference_steps, k_grad_steps,
            lambda_preserve, mask_dilate_px, guidance_scale,
            seed, loss_divisor=num_accum, device=device,
            class_weights=class_weights,
        )
        all_rewards.append(log_i["reward"])
        all_preservs.append(log_i["preserv"])
        for k, v in log_i["per_class"].items():
            all_per_class.setdefault(k, []).append(v)

    mean_reward = float(np.mean(all_rewards))
    mean_preserv = float(np.mean(all_preservs))
    mean_loss = -mean_reward + lambda_preserve * mean_preserv
    mean_per_class = {k: float(np.mean(v)) for k, v in all_per_class.items()}

    return {
        "loss": mean_loss,
        "reward": mean_reward,
        "preserv": mean_preserv,
        "per_class_reward": mean_per_class,
        "num_accum": num_accum,
        "reward_stdev": float(np.std(all_rewards)) if num_accum > 1 else 0.0,
    }
