"""Sampling with gradient enabled on last K denoising steps.

For AlignProp / DRaFT-K training on FLUX:
  - First T-K steps run under torch.no_grad() (cheap, no activation memory)
  - Last K steps run with gradient tracking → grad flows back through transformer
    to trainable LoRA weights
  - VAE decode is gradient-checkpointed to save memory
  - Returns raw pixel tensor in [-1, 1]

Simplifications vs omini.pipeline.flux_omini.generate:
  - No CFG (assume image_guidance_scale = 1.0)
  - No KV cache
  - No latent_mask / complement conditions
  - No callbacks
  - Returns torch.Tensor (not PIL)
"""
from typing import List, Optional, Union, Dict, Any
import numpy as np
import torch
import torch.utils.checkpoint as tc
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from omini.pipeline.flux_omini import transformer_forward


def prepare_condition_data(pipeline, conditions):
    """Encode conditions into c_latents, c_ids, c_adapters list.

    Call this BEFORE CPU-offloading text encoders — needs VAE on GPU and
    pipeline.device reporting the GPU (which it does while text encoders
    are still on GPU).
    """
    c_latents, c_ids, c_adapters = [], [], []
    for condition in conditions:
        tokens, ids = condition.encode(pipeline)
        c_latents.append(tokens)
        c_ids.append(ids)
        c_adapters.append(condition.adapter)
    return {"c_latents": c_latents, "c_ids": c_ids, "c_adapters": c_adapters}


def flux_sample_with_grad(
    pipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    conditions: List = [],
    condition_data: Optional[Dict[str, List]] = None,  # precomputed (prepare_condition_data)
    main_adapter: Optional[str] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 10,
    k_grad_steps: int = 3,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    guidance_scale: float = 3.5,
    vae_checkpoint: bool = True,
    transformer_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
) -> torch.Tensor:
    """Returns (B, 3, H, W) tensor in ~[-1, 1], grad-enabled on trainable params.

    Caller is responsible for ensuring this is invoked OUTSIDE torch.no_grad()
    context (else k_grad_steps loop won't produce gradients).
    """
    transformer_kwargs = transformer_kwargs or {}
    assert 0 < k_grad_steps <= num_inference_steps, \
        f"k_grad_steps must be in (0, {num_inference_steps}], got {k_grad_steps}"

    self = pipeline
    # Use transformer's device directly so text-encoder CPU offload doesn't
    # confuse pipeline._execution_device
    device = next(self.transformer.parameters()).device

    # ---- encode prompt ----
    if prompt_embeds is None or pooled_prompt_embeds is None:
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
    else:
        text_ids = torch.zeros(prompt_embeds.shape[1], 3,
                               device=device, dtype=prompt_embeds.dtype)

    batch_size = prompt_embeds.shape[0]

    # ---- initial latents (noise) ----
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size, num_channels_latents, height, width,
        prompt_embeds.dtype, device, generator, latents,
    )

    # ---- encode conditions (composite) ----
    if condition_data is not None:
        c_latents = condition_data["c_latents"]
        c_ids = condition_data["c_ids"]
        c_adapters = condition_data["c_adapters"]
    else:
        c_latents, c_ids, c_adapters = [], [], []
        for condition in conditions:
            tokens, ids = condition.encode(self)
            c_latents.append(tokens)
            c_ids.append(ids)
            c_adapters.append(condition.adapter)
    n_cond = len(c_latents)
    c_timesteps = [torch.zeros([1], device=device) for _ in range(n_cond)]
    c_projections = [pooled_prompt_embeds for _ in range(n_cond)]
    c_guidances = [torch.ones([1], device=device) for _ in range(n_cond)]

    # ---- timesteps (rectified flow) ----
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, None, sigmas, mu=mu
    )

    # ---- attention group mask (same as generate) ----
    branch_n = n_cond + 2
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    group_mask[2:, 2:] = torch.diag(torch.tensor([1] * n_cond))

    # ---- denoising step helper (shared by no-grad and grad portions) ----
    def _step(latents, t):
        timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000

        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(latents.shape[0])
            g_list = [guidance, guidance]
            cg_list = c_guidances
        else:
            g_list = [None, None]
            cg_list = [None for _ in c_guidances]

        noise_pred = transformer_forward(
            self.transformer,
            image_features=[latents] + c_latents,
            text_features=[prompt_embeds],
            img_ids=[latent_image_ids] + c_ids,
            txt_ids=[text_ids],
            timesteps=[timestep, timestep] + c_timesteps,
            pooled_projections=[pooled_prompt_embeds] * 2 + c_projections,
            guidances=g_list + cg_list,
            return_dict=False,
            adapters=[main_adapter] * 2 + c_adapters,
            group_mask=group_mask,
            **transformer_kwargs,
        )[0]
        return self.scheduler.step(noise_pred, t, latents)[0]

    # ---- split denoising: no-grad first, then grad ----
    n_no_grad = num_inference_steps - k_grad_steps

    with torch.no_grad():
        for i in range(n_no_grad):
            latents = _step(latents, timesteps[i])

    for i in range(n_no_grad, num_inference_steps):
        latents = _step(latents, timesteps[i])

    # ---- VAE decode ----
    latents_u = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents_s = latents_u / self.vae.config.scaling_factor + self.vae.config.shift_factor

    if vae_checkpoint:
        image = tc.checkpoint(
            lambda z: self.vae.decode(z, return_dict=False)[0],
            latents_s,
            use_reentrant=False,
        )
    else:
        image = self.vae.decode(latents_s, return_dict=False)[0]

    return image  # (B, 3, H, W) in ~[-1, 1]
