"""
YOLO-based reward module for AlignProp v3.

Provides three scalar functions of (I_fine, GT bboxes, GT classes):

    r_cls_yolo(preds, bboxes, classes)
        Mean p(correct_class) over anchors whose center lies inside
        some GT bbox, for each GT. Higher = better.

    neg_bg_yolo(preds, bboxes)
        Mean max-class-prob over anchors whose center lies OUTSIDE
        every GT bbox. Lower = better (less hallucination).

    halluc_mask_yolo(I_base, bboxes, ...)   [non-differentiable]
        Binary mask of regions in I_base where YOLO detects components
        that don't match any GT bbox. Used as M_h in adaptive-mask loss.

YOLOv8m (ultralytics 8.4+), 9 classes, 1024x1024 input → 21504 anchors.
No separate objectness channel in v8; class scores serve directly as
detection confidence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics import YOLO


class YOLOv8Reward:
    """
    Wrapper around ultralytics YOLOv8 that exposes pre-NMS grid output
    with gradients intact, plus three reward helpers.
    """

    def __init__(
        self,
        weights_path: str,
        device: str | torch.device = "cuda:0",
        n_classes: int = 9,
        img_size: int = 1024,
    ):
        self.yolo = YOLO(weights_path)
        self.det_model = self.yolo.model.to(device).eval()
        # Freeze everything — reward model should never update
        for p in self.det_model.parameters():
            p.requires_grad_(False)

        self.device = torch.device(device)
        self.n_classes = n_classes
        self.img_size = img_size

        # Precompute anchor centers for the 3 FPN levels at strides 8, 16, 32
        self.anchor_centers = self._compute_anchor_centers()       # [N, 2] xy
        self.anchor_strides = self._compute_anchor_strides()       # [N] stride per anchor

    def _compute_anchor_centers(self) -> torch.Tensor:
        """Return [N, 2] tensor of (cx, cy) anchor centers in pixel coords."""
        centers = []
        for stride in (8, 16, 32):
            h = w = self.img_size // stride
            ys = torch.arange(h, dtype=torch.float32) * stride + stride / 2
            xs = torch.arange(w, dtype=torch.float32) * stride + stride / 2
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            centers.append(torch.stack([gx.flatten(), gy.flatten()], dim=-1))
        return torch.cat(centers, dim=0).to(self.device)           # [N, 2]

    def _compute_anchor_strides(self) -> torch.Tensor:
        strides = []
        for stride in (8, 16, 32):
            h = w = self.img_size // stride
            strides.append(torch.full((h * w,), float(stride)))
        return torch.cat(strides, dim=0).to(self.device)

    # ----------------------------------------------------------------- forward

    def forward(self, img_01: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_01: [B, 3, H, W] in [0, 1] range
        Returns:
            preds: [B, 4 + nc, N]  decoded pre-NMS output with grad
                   channels [0:4]   = (cx, cy, w, h) in pixel coords
                   channels [4:4+nc] = per-class probability (post-sigmoid)
        """
        out = self.det_model(img_01)
        # eval() mode returns (preds, features) tuple in ultralytics YOLOv8
        preds = out[0] if isinstance(out, (tuple, list)) else out
        return preds

    # --------------------------------------------------------------- rewards

    def r_cls(
        self,
        preds: torch.Tensor,
        bboxes_xyxy: list[torch.Tensor],
        classes: list[torch.Tensor],
        pool: str = "max",
        weights_per_gt: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Per-component classifier reward.

        Args:
            preds: [B, 4+nc, N]
            bboxes_xyxy: list of [K_b, 4] tensors (one per batch item),
                         boxes in (x1, y1, x2, y2) image-pixel coords
            classes:     list of [K_b] tensors of class indices (0..nc-1)
            pool:        "max" (default) or "mean" — how to reduce p(correct)
                         across anchors inside each bbox.
                         MAX: reflects peak detection, matches inference NMS
                         MEAN: diluted by edge anchors, pre-v3.1 default
            weights_per_gt: optional list of [K_b] tensors, per-GT weighting.
                         If given, final reward is (Σ w_i · p_i) / Σ w_i
                         instead of equal mean over GTs. Use this to focus
                         on GTs baseline gets wrong (hardness weighting).

        Returns:
            scalar; higher = better.
        """
        B = preds.shape[0]
        cls_scores = preds[:, 4 : 4 + self.n_classes, :]    # [B, nc, N]

        cx = self.anchor_centers[:, 0]                       # [N]
        cy = self.anchor_centers[:, 1]                       # [N]

        rewards = []
        for b in range(B):
            bb = bboxes_xyxy[b].to(self.device)              # [K, 4]
            cc = classes[b].to(self.device).long()           # [K]
            K = bb.shape[0]
            if K == 0:
                continue

            x1, y1, x2, y2 = bb[:, 0:1], bb[:, 1:2], bb[:, 2:3], bb[:, 3:4]
            inside = (
                (cx[None, :] >= x1) & (cx[None, :] <= x2) &
                (cy[None, :] >= y1) & (cy[None, :] <= y2)
            )                                                 # [K, N] bool
            inside_f = inside.float()
            p_correct = cls_scores[b, cc, :]                 # [K, N]

            n_inside = inside_f.sum(dim=-1)                   # [K]
            has_match = n_inside > 0
            if not has_match.any():
                continue

            if pool == "max":
                # Replace zeros outside bbox with -1 so max picks inside-only
                p_masked = p_correct * inside_f + (inside_f - 1.0) * 1e6
                p_per_gt = p_masked.max(dim=-1).values         # [K]
            else:  # mean
                p_per_gt = (p_correct * inside_f).sum(dim=-1) / n_inside.clamp(min=1)

            p_per_gt = p_per_gt[has_match]                   # [K_valid]

            if weights_per_gt is not None:
                w = weights_per_gt[b].to(self.device).float()
                w = w[has_match]
                w_sum = w.sum().clamp(min=1e-6)
                rewards.append((w * p_per_gt).sum() / w_sum)
            else:
                rewards.append(p_per_gt.mean())

        if not rewards:
            return preds.new_zeros(())
        return torch.stack(rewards).mean()

    def neg_bg(
        self,
        preds: torch.Tensor,
        bboxes_xyxy: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Background-hallucination penalty.

        Returns:
            scalar; mean of max-class-prob over all background anchors.
            Lower = better (less hallucination).
        """
        B, _, N = preds.shape
        cls_scores = preds[:, 4 : 4 + self.n_classes, :]
        max_cls = cls_scores.max(dim=1).values              # [B, N]

        cx = self.anchor_centers[:, 0]
        cy = self.anchor_centers[:, 1]

        penalties = []
        for b in range(B):
            bb = bboxes_xyxy[b].to(self.device)
            K = bb.shape[0]
            if K == 0:
                bg_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            else:
                x1, y1, x2, y2 = bb[:, 0:1], bb[:, 1:2], bb[:, 2:3], bb[:, 3:4]
                inside_any = (
                    (cx[None, :] >= x1) & (cx[None, :] <= x2) &
                    (cy[None, :] >= y1) & (cy[None, :] <= y2)
                ).any(dim=0)                                 # [N]
                bg_mask = ~inside_any

            if not bg_mask.any():
                continue
            penalties.append(max_cls[b, bg_mask].mean())

        if not penalties:
            return preds.new_zeros(())
        return torch.stack(penalties).mean()

    # ------------------------------------------------------- adaptive 3-state mask

    @torch.no_grad()
    def adaptive_mask(
        self,
        img_base_01: torch.Tensor,
        bboxes_xyxy: torch.Tensor,
        classes: torch.Tensor,
        base_correct_conf_thresh: float = 0.5,
        halluc_conf_thresh: float = 0.25,
        iou_thresh: float = 0.30,
        dilate_px: int = 15,
    ):
        """
        3-state mask for FNF-style preservation.

        Identifies where delta SHOULD be allowed to change the image:
          (a) GT bboxes where baseline gets the wrong class / is uncertain
          (b) Regions where baseline hallucinates (FP detections outside all GTs)

        Everything else (correctly-detected GTs + clean background) is EXCLUDED
        from M → preservation pins it to I_base.

        Args:
            img_base_01:  [1, 3, H, W] baseline generation in [0, 1]
            bboxes_xyxy:  [K, 4] GT bboxes
            classes:      [K] GT class indices
            base_correct_conf_thresh: p(correct class) above this at center
                                      anchor means "baseline got it right"
            halluc_conf_thresh: detection confidence cutoff for hallucination
            iou_thresh:   detection→GT IoU cutoff to distinguish TP from FP
            dilate_px:    halo around each masked region

        Returns:
            M:           [H, W] binary mask (1.0 where delta can change)
            stats:       dict with n_gt, n_gt_wrong, n_halluc, coverages, and
                         per_gt_baseline_conf [K] tensor (for hardness weight)
        """
        assert img_base_01.shape[0] == 1, "adaptive_mask is per-sample"
        H, W = img_base_01.shape[-2:]
        device = img_base_01.device

        preds = self.forward(img_base_01)                    # [1, 4+nc, N]
        cls_scores = preds[:, 4 : 4 + self.n_classes, :]     # [1, nc, N]
        max_cls_per_anchor, pred_cls_per_anchor = cls_scores[0].max(dim=0)  # [N], [N]
        boxes_cxcywh = preds[0, :4, :]                       # [4, N]

        M = torch.zeros(H, W, device=device, dtype=torch.float32)
        K = bboxes_xyxy.shape[0]
        per_gt_base_conf = torch.zeros(K, device=device, dtype=torch.float32)
        per_gt_base_correct = torch.zeros(K, dtype=torch.bool, device=device)

        gt_bboxes = bboxes_xyxy.to(device).float()
        gt_classes = classes.to(device).long()

        # ---- 1) Per-GT: find baseline's output at the CENTER anchor --------
        for k in range(K):
            gcx = (gt_bboxes[k, 0] + gt_bboxes[k, 2]) / 2
            gcy = (gt_bboxes[k, 1] + gt_bboxes[k, 3]) / 2
            d2 = (self.anchor_centers[:, 0] - gcx).pow(2) + \
                 (self.anchor_centers[:, 1] - gcy).pow(2)
            closest = int(d2.argmin().item())
            gt_c = int(gt_classes[k].item())
            base_conf = float(cls_scores[0, gt_c, closest].item())   # prob of TRUE class
            per_gt_base_conf[k] = base_conf

            base_pred = int(pred_cls_per_anchor[closest].item())
            base_max = float(max_cls_per_anchor[closest].item())
            is_correct = (base_pred == gt_c) and (base_conf >= base_correct_conf_thresh)
            per_gt_base_correct[k] = is_correct

            if not is_correct:
                # Include in M — delta needs to fix this GT region
                x1 = max(0, int(gt_bboxes[k, 0].item()) - dilate_px)
                y1 = max(0, int(gt_bboxes[k, 1].item()) - dilate_px)
                x2 = min(W, int(gt_bboxes[k, 2].item()) + dilate_px)
                y2 = min(H, int(gt_bboxes[k, 3].item()) + dilate_px)
                if x2 > x1 and y2 > y1:
                    M[y1:y2, x1:x2] = 1.0

        # ---- 2) Hallucinations: detections outside every GT ----------------
        cand_idx = (max_cls_per_anchor > halluc_conf_thresh).nonzero(as_tuple=True)[0]
        halluc_count = 0
        if cand_idx.numel() > 0:
            cand_boxes = boxes_cxcywh[:, cand_idx]            # [4, M]
            cx_b, cy_b, bw, bh = cand_boxes[0], cand_boxes[1], cand_boxes[2], cand_boxes[3]
            det_xyxy = torch.stack(
                [cx_b - bw / 2, cy_b - bh / 2, cx_b + bw / 2, cy_b + bh / 2], dim=-1
            )                                                  # [M, 4]
            if gt_bboxes.numel() > 0:
                ious = _iou_matrix(det_xyxy, gt_bboxes)        # [M, K]
                max_iou = ious.max(dim=-1).values              # [M]
                is_halluc = max_iou < iou_thresh
            else:
                is_halluc = torch.ones(det_xyxy.shape[0], dtype=torch.bool, device=device)
            halluc_boxes = det_xyxy[is_halluc]
            for x1, y1, x2, y2 in halluc_boxes.tolist():
                xa = max(0, int(x1 - dilate_px))
                ya = max(0, int(y1 - dilate_px))
                xb = min(W, int(x2 + dilate_px))
                yb = min(H, int(y2 + dilate_px))
                if xa < xb and ya < yb:
                    M[ya:yb, xa:xb] = 1.0
                    halluc_count += 1

        stats = dict(
            n_gt=K,
            n_gt_wrong=int((~per_gt_base_correct).sum().item()),
            n_halluc=halluc_count,
            M_cov=float(M.mean().item()),
            per_gt_base_conf=per_gt_base_conf,              # [K] — for hardness weight
            per_gt_base_correct=per_gt_base_correct,        # [K] — bool
        )
        return M, stats

    # ------------------------------------------------------- hallucination mask (legacy)

    @torch.no_grad()
    def halluc_mask(
        self,
        img_base_01: torch.Tensor,
        bboxes_xyxy: torch.Tensor,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.30,
        dilate_px: int = 15,
    ) -> torch.Tensor:
        """
        Build a [H, W] binary mask of FP detections in I_base.

        Args:
            img_base_01: [1, 3, H, W] image in [0, 1] range
            bboxes_xyxy: [K, 4] GT bboxes in pixel coords
            conf_thresh: min class confidence to consider a detection
            iou_thresh:  if detection IoU with ANY GT > iou_thresh, it's NOT a
                         hallucination (just a slightly off detection)
            dilate_px:   halo to add around each hallucination bbox

        Returns:
            mask: [H, W] float tensor, 1.0 inside hallucination regions
        """
        assert img_base_01.shape[0] == 1, "halluc_mask is per-sample"
        H, W = img_base_01.shape[-2:]

        preds = self.forward(img_base_01)                    # [1, 4+nc, N]
        cls_scores = preds[:, 4 : 4 + self.n_classes, :]
        max_cls, _ = cls_scores.max(dim=1)                   # [1, N]
        boxes = preds[0, :4, :]                              # [4, N]  cxcywh

        mask = torch.zeros(H, W, device=img_base_01.device, dtype=torch.float32)

        candidate_idx = (max_cls[0] > conf_thresh).nonzero(as_tuple=True)[0]
        if candidate_idx.numel() == 0:
            return mask

        cand_boxes = boxes[:, candidate_idx]                 # [4, M]  cxcywh
        # Convert to xyxy
        cx, cy, bw, bh = cand_boxes[0], cand_boxes[1], cand_boxes[2], cand_boxes[3]
        det_xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)   # [M, 4]

        gt = bboxes_xyxy.to(img_base_01.device).float()      # [K, 4]
        # Vectorized IoU [M, K]
        if gt.numel() > 0:
            ious = _iou_matrix(det_xyxy, gt)                 # [M, K]
            max_iou_per_det = ious.max(dim=-1).values        # [M]
            is_halluc = max_iou_per_det < iou_thresh         # [M]
        else:
            is_halluc = torch.ones(det_xyxy.shape[0], dtype=torch.bool, device=self.device)

        halluc_boxes = det_xyxy[is_halluc]                   # [H_hall, 4]
        for x1, y1, x2, y2 in halluc_boxes.tolist():
            xa = max(0, int(x1 - dilate_px))
            ya = max(0, int(y1 - dilate_px))
            xb = min(W, int(x2 + dilate_px))
            yb = min(H, int(y2 + dilate_px))
            if xa < xb and ya < yb:
                mask[ya:yb, xa:xb] = 1.0

        return mask


# --------------------------------------------------------------------- utils

def _iou_matrix(a_xyxy: torch.Tensor, b_xyxy: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between [M, 4] and [K, 4] boxes in (x1,y1,x2,y2)."""
    M, K = a_xyxy.shape[0], b_xyxy.shape[0]
    ax1, ay1, ax2, ay2 = a_xyxy[:, 0:1], a_xyxy[:, 1:2], a_xyxy[:, 2:3], a_xyxy[:, 3:4]
    bx1, by1, bx2, by2 = b_xyxy[:, 0:1].T, b_xyxy[:, 1:2].T, b_xyxy[:, 2:3].T, b_xyxy[:, 3:4].T
    ix1 = torch.max(ax1, bx1)
    iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2)
    iy2 = torch.min(ay2, by2)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def bbox_union_mask(
    bboxes_xyxy: torch.Tensor, H: int, W: int, dilate_px: int = 15
) -> torch.Tensor:
    """
    Build a [H, W] binary mask covering the union of bboxes, optionally dilated.
    Pure function, no YOLO needed.
    """
    mask = torch.zeros(H, W, device=bboxes_xyxy.device, dtype=torch.float32)
    for x1, y1, x2, y2 in bboxes_xyxy.tolist():
        xa = max(0, int(x1 - dilate_px))
        ya = max(0, int(y1 - dilate_px))
        xb = min(W, int(x2 + dilate_px))
        yb = min(H, int(y2 + dilate_px))
        if xa < xb and ya < yb:
            mask[ya:yb, xa:xb] = 1.0
    return mask
