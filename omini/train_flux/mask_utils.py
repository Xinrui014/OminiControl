"""Bbox -> binary mask utilities for Focus-N-Fix preservation loss."""
from typing import List, Tuple
import torch


def bboxes_to_mask(
    bboxes: List[Tuple[int, int, int, int]],
    H: int,
    W: int,
    dilate_px: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build (1, 1, H, W) mask with 1 inside (dilated) component bboxes, 0 outside.

    Args:
        bboxes: list of (x1, y1, x2, y2) in pixel coords at the (H, W) resolution
        H, W:   pixel dimensions
        dilate_px: padding around each bbox (gives model room at component edges)
    """
    mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    for x1, y1, x2, y2 in bboxes:
        x1 = max(0, int(x1) - dilate_px)
        y1 = max(0, int(y1) - dilate_px)
        x2 = min(W, int(x2) + dilate_px)
        y2 = min(H, int(y2) + dilate_px)
        if x2 > x1 and y2 > y1:
            mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


def scale_bboxes(
    bboxes: List[Tuple[float, float, float, float]],
    src_H: int, src_W: int,
    dst_H: int, dst_W: int,
) -> List[Tuple[int, int, int, int]]:
    """Rescale bboxes from (src_H, src_W) coord space to (dst_H, dst_W)."""
    sx = dst_W / src_W
    sy = dst_H / src_H
    out = []
    for x1, y1, x2, y2 in bboxes:
        out.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
    return out
