"""AlignProp v3.2 step - detection-aligned FP suppression + preservation.

Loss form (still 2-term Focus-N-Fix structure):

    L = gamma_fp * L_FP(I_fine, gts)
      + beta    * ||(1 - M) * (I_base - I_fine)||_F

where:
    L_FP   : focal-weighted BCE-for-y=0 on specific (anchor, class) pairs
             that produce FP detections in I_fine (via NMS + GT matching).
             Borderline BG pool (p in [0.15, 0.25]) included for gradient
             density. See reward_yolo_v3_2.fp_suppression_loss.
    M      : hybrid preserve mask, union of FP bboxes in I_base and I_fine
             dilated, minus TP_base (always protected). See
             reward_yolo_v3_2.build_preserve_mask_v3_2.

No r_cls, no neg_bg. MISS rescue is not handled (Option 2 trade-off).
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from omini.train_flux.flux_sample_with_grad import flux_sample_with_grad
from omini.train_flux.reward_yolo import YOLOv8Reward
from omini.train_flux.reward_composite import preservation_frobenius
from omini.train_flux.reward_yolo_v3_2 import (
    find_fp_anchors,
    fp_suppression_loss,
    build_preserve_mask_v3_2,
)


def _set_adapter_scale(pipe, adapter_name: str, scale: float):
    from peft.tuners.lora.layer import BaseTunerLayer
    for m in pipe.transformer.modules():
        if isinstance(m, BaseTunerLayer):
            if adapter_name in m.scaling:
                m.scaling[adapter_name] = scale


def _vae_to_yolo(img_vae: torch.Tensor) -> torch.Tensor:
    return ((img_vae + 1.0) / 2.0).clamp(0.0, 1.0).float()


def _batch_condition_data(condition_data, batch_size):
    """Repeat c_latents (and c_ids if 3-D) along batch dim for bs>1 forward."""
    out = {"c_adapters": condition_data["c_adapters"]}
    out["c_latents"] = [
        t.repeat(batch_size, *([1] * (t.dim() - 1))) for t in condition_data["c_latents"]
    ]
    # c_ids: comment says [token_n, id_dim] (no batch). If already 3-D, repeat.
    out["c_ids"] = [
        t.repeat(batch_size, *([1] * (t.dim() - 1))) if t.dim() == 3 else t
        for t in condition_data["c_ids"]
    ]
    return out


def _compute_ibase_batched(
    pipe, prompt_embeds, pooled_prompt_embeds, condition_data,
    delta_adapter_name,
    height, width, num_inference_steps,
    seeds, guidance_scale, device,
):
    """Generate I_base for all seeds in a single bs=N no_grad FLUX forward.

    Returns list of detached [1, 3, H, W] tensors, one per seed.
    """
    N = len(seeds)
    _set_adapter_scale(pipe, delta_adapter_name, 0.0)

    pe_batched = prompt_embeds.repeat(N, 1, 1)                         # [N, S, D]
    pool_batched = pooled_prompt_embeds.repeat(N, 1)                   # [N, D]
    cd_batched = _batch_condition_data(condition_data, N)

    # Use the first seed's generator (FLUX's randn produces independent per-batch noise)
    gen = torch.Generator(device=device).manual_seed(seeds[0])

    with torch.no_grad():
        images_batched = flux_sample_with_grad(
            pipe, prompt_embeds=pe_batched, pooled_prompt_embeds=pool_batched,
            condition_data=cd_batched, main_adapter=None,
            height=height, width=width,
            num_inference_steps=num_inference_steps, k_grad_steps=1,
            generator=gen, guidance_scale=guidance_scale,
            vae_checkpoint=False,
        )                                                              # [N, 3, H, W]
    return [images_batched[i:i+1].detach() for i in range(N)]


def _one_backward_pass(
    pipe, prompt_embeds, pooled_prompt_embeds, condition_data,
    bboxes_xyxy, classes, reward_model,
    delta_adapter_name,
    height, width, num_inference_steps, k_grad_steps,
    # loss weights
    gamma_fp, beta, gamma_focal,
    # FP detection config
    nms_conf_thresh, nms_iou_thresh, match_iou_thresh, borderline_conf_low,
    # mask config
    mask_dilate_px, mask_area_cap,
    # sampling
    guidance_scale, seed, loss_divisor, device,
    # optional precomputed I_base (from batched forward)
    precomputed_image_base=None,
):
    """One (I_base, I_fine) -> loss -> backward at a single noise seed."""
    # ---- I_base: delta OFF, no grad ----------------------------------------
    if precomputed_image_base is not None:
        image_base = precomputed_image_base
    else:
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

    # ---- I_fine: delta ON, grad --------------------------------------------
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

    img_base_01 = _vae_to_yolo(image_base)
    img_fine_01 = _vae_to_yolo(image_fine)

    bboxes_t = torch.as_tensor(bboxes_xyxy, device=device, dtype=torch.float32)
    classes_t = torch.as_tensor(classes, device=device, dtype=torch.long)

    # ---- Identify FPs in both I_base and I_fine (no_grad) ------------------
    with torch.no_grad():
        info_base = find_fp_anchors(
            reward_model, img_base_01, bboxes_t, classes_t,
            nms_conf_thresh=nms_conf_thresh,
            nms_iou_thresh=nms_iou_thresh,
            match_iou_thresh=match_iou_thresh,
            borderline_conf_low=1.0,   # baseline doesn't need borderline
        )
        info_fine = find_fp_anchors(
            reward_model, img_fine_01.detach(), bboxes_t, classes_t,
            nms_conf_thresh=nms_conf_thresh,
            nms_iou_thresh=nms_iou_thresh,
            match_iou_thresh=match_iou_thresh,
            borderline_conf_low=borderline_conf_low,
        )

    # ---- Build preserve mask -----------------------------------------------
    preserve_mask, m_stats = build_preserve_mask_v3_2(
        H=height, W=width,
        fp_bboxes_base=info_base["fp_bboxes_xyxy"],
        fp_bboxes_fine=info_fine["fp_bboxes_xyxy"],
        tp_bboxes_base=info_base["tp_bboxes_xyxy"],
        dilate_px=mask_dilate_px,
        area_cap=mask_area_cap,
        device=device,
    )

    # ---- Forward WITH grad, extract FP probs, focal-weighted loss ----------
    preds_fine = reward_model.forward(img_fine_01)                 # [1, 4+nc, N]

    L_FP, fp_stats = fp_suppression_loss(
        preds_fine,
        fp_anchor_idx=info_fine["fp_anchor_idx"],
        fp_class_idx=info_fine["fp_class_idx"],
        gamma_focal=gamma_focal,
        n_classes=reward_model.n_classes,
    )

    # ---- Preservation (Frobenius, same as v3) ------------------------------
    pres = preservation_frobenius(image_base, image_fine, preserve_mask)

    # ---- Total loss --------------------------------------------------------
    loss = (gamma_fp * L_FP + beta * pres) / loss_divisor
    loss.backward()

    return {
        "L_FP":              fp_stats["L_FP"],
        "pres":              float(pres.item()),
        "loss":              float(loss.item() * loss_divisor),
        "n_fp_fine_nms":     info_fine["n_fp_nms"],
        "n_fp_fine_borderline": info_fine["n_fp_borderline"],
        "n_fp_base_nms":     info_base["n_fp_nms"],
        "n_tp_fine":         info_fine["n_tp"],
        "n_tp_base":         info_base["n_tp"],
        "n_miss_fine":       info_fine["n_miss"],
        "n_miss_base":       info_base["n_miss"],
        "n_gt":              info_fine["n_gt"],
        "p_fp_mean":         fp_stats["p_fp_mean"],
        "p_fp_max":          fp_stats["p_fp_max"],
        **{f"mask_{k}": (float(v) if not isinstance(v, bool) else int(v))
           for k, v in m_stats.items()},
    }


def alignprop_step_v3_2(
    pipe,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    condition_data: Dict,
    bboxes_xyxy: List[Tuple[int, int, int, int]],
    classes: List[int],
    reward_model: YOLOv8Reward,
    delta_adapter_name: str = "delta",
    height: int = 1024, width: int = 1024,
    num_inference_steps: int = 10,
    k_grad_steps: int = 3,
    # v3.2 loss weights
    gamma_fp: float = 0.1,
    beta: float = 1.0,
    gamma_focal: float = 2.0,
    # FP detection
    nms_conf_thresh: float = 0.25,
    nms_iou_thresh: float = 0.45,
    match_iou_thresh: float = 0.30,
    borderline_conf_low: float = 0.15,
    # mask
    mask_dilate_px: int = 20,
    mask_area_cap: float = 0.40,
    # sampling
    guidance_scale: float = 3.5,
    generator: Optional[torch.Generator] = None,
    num_accum: int = 2,
    batch_ibase: bool = False,
) -> Dict[str, float]:
    """v3.2 training step with gradient accumulation.

    batch_ibase=True precomputes all accum I_base images in a single bs=N
    no_grad FLUX forward, saves ~half the FLUX forward time (cheap no-grad
    doesn't need activation memory). I_fine still runs sequentially per-accum.
    """
    device = next(pipe.transformer.parameters()).device

    base_seed = int(generator.initial_seed()) if generator is not None else 42
    seeds = [(base_seed + i * 997) & 0xFFFFFFFF for i in range(num_accum)]

    # Optionally pre-compute I_base batched (no_grad, cheap memory)
    if batch_ibase and num_accum >= 2:
        precomputed_ibases = _compute_ibase_batched(
            pipe, prompt_embeds, pooled_prompt_embeds, condition_data,
            delta_adapter_name, height, width, num_inference_steps,
            seeds, guidance_scale, device,
        )
    else:
        precomputed_ibases = [None] * num_accum

    accum_logs: Dict[str, List[float]] = {}
    for i, seed in enumerate(seeds):
        log_i = _one_backward_pass(
            pipe, prompt_embeds, pooled_prompt_embeds, condition_data,
            bboxes_xyxy, classes, reward_model,
            delta_adapter_name,
            height, width, num_inference_steps, k_grad_steps,
            gamma_fp, beta, gamma_focal,
            nms_conf_thresh, nms_iou_thresh, match_iou_thresh, borderline_conf_low,
            mask_dilate_px, mask_area_cap,
            guidance_scale, seed, loss_divisor=num_accum, device=device,
            precomputed_image_base=precomputed_ibases[i],
        )
        for k, v in log_i.items():
            accum_logs.setdefault(k, []).append(float(v))

    mean_logs = {k: float(np.mean(v)) for k, v in accum_logs.items()}
    mean_logs["num_accum"] = num_accum
    return mean_logs


def alignprop_step_v3_2_ddp(
    pipe, prompt_embeds, pooled_prompt_embeds, condition_data,
    bboxes_xyxy, classes, reward_model,
    delta_params, world_size, device,
    delta_adapter_name="delta",
    height=1024, width=1024,
    num_inference_steps=10, k_grad_steps=3,
    gamma_fp=0.1, beta=1.0, gamma_focal=2.0,
    nms_conf_thresh=0.25, nms_iou_thresh=0.45,
    match_iou_thresh=0.30, borderline_conf_low=0.15,
    mask_dilate_px=20, mask_area_cap=0.40,
    guidance_scale=3.5,
    base_seed=42, num_accum=2,
    batch_ibase=False,
):
    """DDP wrapper: step + manual grad all_reduce."""
    from torch.distributed import is_initialized as _dist_ok, ReduceOp, all_reduce

    logs = alignprop_step_v3_2(
        pipe, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        condition_data=condition_data,
        bboxes_xyxy=bboxes_xyxy, classes=classes,
        reward_model=reward_model,
        delta_adapter_name=delta_adapter_name,
        height=height, width=width,
        num_inference_steps=num_inference_steps, k_grad_steps=k_grad_steps,
        gamma_fp=gamma_fp, beta=beta, gamma_focal=gamma_focal,
        nms_conf_thresh=nms_conf_thresh, nms_iou_thresh=nms_iou_thresh,
        match_iou_thresh=match_iou_thresh, borderline_conf_low=borderline_conf_low,
        mask_dilate_px=mask_dilate_px, mask_area_cap=mask_area_cap,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=device).manual_seed(base_seed),
        num_accum=num_accum,
        batch_ibase=batch_ibase,
    )

    if world_size > 1 and _dist_ok():
        for p in delta_params:
            if p.grad is not None:
                all_reduce(p.grad, op=ReduceOp.SUM)
                p.grad.div_(world_size)

    return logs
