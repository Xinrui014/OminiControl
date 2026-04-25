"""
Composite reward assembly + Frobenius preservation for AlignProp v3.

Keeps the training objective in FNF's 2-term form:

    L = - r_composite(Î, Î₀, bboxes, classes)                       # reward
        + β · ‖(1 - M) ⊙ (Î₀ - Î)‖_F                                # preservation

where r_composite is a SINGLE scalar built from task-specific sub-signals
combined with internal (not-tuned-at-train-time) weights γ_cls, γ_bg, γ_lp.

Sub-signals (all produced externally and passed in):
    r_cls    : from reward_yolo.YOLOv8Reward.r_cls    — higher = better
    neg_bg   : from reward_yolo.YOLOv8Reward.neg_bg   — lower  = better
    neg_lp   : - LPIPS(Î₀, Î)                         — lower (more neg) = worse

The composite is then:
    r = γ_cls · r_cls  −  γ_bg · neg_bg_penalty  −  γ_lp · lpips_distance
"""
from __future__ import annotations

import torch


def preservation_frobenius(
    img_base: torch.Tensor,
    img_fine: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    ‖(1 - M) ⊙ (Î₀ - Î)‖_F
    Unsquared Frobenius norm — faithful to Focus-N-Fix.

    Args:
        img_base, img_fine: [B, C, H, W]
        mask:               [H, W] or [B, 1, H, W]  binary (1 inside fix region)

    Returns:
        scalar loss (lower = better preservation)
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)                   # [1, 1, H, W]
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)                                # [B, 1, H, W]
    inv_mask = 1.0 - mask
    diff = inv_mask * (img_base - img_fine)
    return torch.linalg.vector_norm(diff.flatten())             # ‖·‖_F


def masked_lpips(
    lpips_net,
    img_base: torch.Tensor,
    img_fine: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    LPIPS distance between Î₀ and Î, optionally masked to outside-M.

    Args:
        lpips_net:       a callable lpips.LPIPS(net='vgg') style module
        img_base, img_fine: [B, C, H, W] in [-1, 1] range (LPIPS convention)
        mask:            optional [H, W] or [B, 1, H, W]; if given, pixels
                         inside mask are replaced with Î₀ in both inputs,
                         so LPIPS contribution is zero there.

    Returns:
        scalar LPIPS distance (lower = better)
    """
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # Inside mask, copy from base so LPIPS sees no change there
        img_fine_masked = img_fine * (1 - mask) + img_base * mask
    else:
        img_fine_masked = img_fine
    d = lpips_net(img_base, img_fine_masked)
    return d.mean()


def compose_reward(
    r_cls: torch.Tensor,
    neg_bg_penalty: torch.Tensor,
    lpips_distance: torch.Tensor | None = None,
    gamma_cls: float = 1.0,
    gamma_bg: float = 1.0,
    gamma_lp: float = 0.0,
) -> torch.Tensor:
    """
    Combine sub-signals into a single scalar reward. Higher = better.

        r = γ_cls · r_cls  −  γ_bg · neg_bg_penalty  −  γ_lp · lpips_distance

    All three signals are passed in as pre-computed scalars.
    γ_lp = 0.0 disables LPIPS entirely (useful if memory is tight).
    """
    r = gamma_cls * r_cls - gamma_bg * neg_bg_penalty
    if lpips_distance is not None and gamma_lp > 0:
        r = r - gamma_lp * lpips_distance
    return r


def total_loss(
    r_composite: torch.Tensor,
    pres_frobenius: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Loss-minimization form:  L = -r + β · pres
    """
    return -r_composite + beta * pres_frobenius
