"""
YOLO detection-aligned reward for AlignProp v3.2.

Key shift from v3.1:
  - v3.1 used pre-NMS anchor-level reward (r_cls max-pool, neg_bg anchor-mean).
    Empirically proven misaligned with F1/mAP (see project_analyze_v3_results.md):
    * 72-75% of FPs at 28-step are BG hallucinations; neg_bg anchor-mean
      did not reduce them (A28 57 -> B28 55).
    * Delta increased wrong-class FPs (A28 16 -> B28 20).
  - v3.2 aligns the reward with what the eval metric actually measures:
    POST-NMS false-positive detections.

The reward operates in two passes per image:
  Pass 1 (no_grad): full NMS inference, match detections to GTs,
                    identify which anchors produced FP detections
                    (BG hallucinations or wrong-class FPs).
  Pass 2 (grad):    forward again, reach into the specific
                    (anchor_idx, class_idx) cells, apply focal-weighted
                    BCE-for-y=0 loss. Gradient flows only to the anchors
                    that produced FPs.

Sparsity handling (three stacked fixes):
  (1) Focal weighting p^gamma_focal - confident FPs get most of the
      gradient (standard Lin 2017 focal).
  (2) Borderline pool: also include anchors with conf in [p_low, 0.25)
      that are outside all GTs (borderline BG hallucinations about to
      cross the detection threshold). Expands candidate count from ~50
      to ~200-500 per image.
  (3) Sum reduction (not mean). Image with more FPs gets proportionally
      more gradient - matches blast-radius of the problem.

Mask construction:
  Preserve mask is built from the UNION of FPs in I_base and I_fine
  (dilated), minus TP_base bboxes (protected, no dilation). This
  ensures (a) preservation never fights FP suppression at the FP
  regions, and (b) TPs stay locked against drift.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch
from torchvision.ops import batched_nms

from omini.train_flux.reward_yolo import YOLOv8Reward, _iou_matrix


# =========================================================================
# 1. FP identification (no_grad)
# =========================================================================

@torch.no_grad()
def find_fp_anchors(
    reward_model: YOLOv8Reward,
    img_01: torch.Tensor,
    bboxes_xyxy: torch.Tensor,
    classes: torch.Tensor,
    nms_conf_thresh: float = 0.25,
    nms_iou_thresh: float = 0.45,
    match_iou_thresh: float = 0.30,
    borderline_conf_low: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Identify anchor-class pairs that produce FP detections.

    Args:
        img_01:       [1, 3, H, W] in [0, 1]
        bboxes_xyxy:  [K, 4] GT bboxes (may be empty K=0)
        classes:      [K] GT class indices
        nms_conf_thresh: YOLO detection threshold (matches eval)
        nms_iou_thresh:  NMS IoU suppression threshold
        match_iou_thresh: det<->GT match IoU threshold (matches eval)
        borderline_conf_low: lower bound for borderline FP pool

    Returns:
        dict with:
          fp_anchor_idx:  [K_fp] long, original anchor indices
          fp_class_idx:   [K_fp] long, predicted class at each FP
          fp_source:      [K_fp] int (0=NMS_FP, 1=borderline_BG)
          n_fp_nms:       int, count of post-NMS FPs
          n_fp_borderline: int, count of borderline BG pool
          n_tp:           int, count of TP detections (monitoring)
          n_miss:         int, count of unmatched GTs (monitoring)
          tp_bboxes_xyxy: [K_tp, 4] bboxes of TPs (for preserve mask)
          fp_bboxes_xyxy: [K_fp_nms, 4] bboxes of NMS-FPs (for preserve mask)
    """
    device = img_01.device
    K_gt = bboxes_xyxy.shape[0]
    nc = reward_model.n_classes

    preds = reward_model.forward(img_01)                          # [1, 4+nc, N]
    N = preds.shape[-1]

    # ----- decode boxes -----
    cxcywh = preds[0, :4, :].T                                    # [N, 4]
    cx, cy, w, h = cxcywh.unbind(-1)
    boxes_xyxy_all = torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
    )                                                              # [N, 4]

    cls_scores = preds[0, 4 : 4 + nc, :].T                        # [N, nc]
    max_conf, pred_cls = cls_scores.max(dim=-1)                    # [N], [N]

    # ----- anchor is inside any GT? (for classifying borderline) -----
    if K_gt > 0:
        ac = reward_model.anchor_centers                           # [N, 2]
        gt = bboxes_xyxy.to(device).float()
        x1g, y1g, x2g, y2g = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3], gt[:, 3:4]
        inside_any = (
            (ac[:, 0][None, :] >= x1g) & (ac[:, 0][None, :] <= x2g) &
            (ac[:, 1][None, :] >= y1g) & (ac[:, 1][None, :] <= y2g)
        ).any(dim=0)                                               # [N]
    else:
        inside_any = torch.zeros(N, dtype=torch.bool, device=device)

    # ----- NMS on above-threshold anchors -----
    above = max_conf > nms_conf_thresh
    if above.any():
        boxes_cand = boxes_xyxy_all[above]
        conf_cand  = max_conf[above]
        cls_cand   = pred_cls[above]
        anchor_idx_cand = torch.nonzero(above, as_tuple=False).squeeze(-1)
        keep = batched_nms(boxes_cand, conf_cand, cls_cand, nms_iou_thresh)
        surv_boxes = boxes_cand[keep]                              # [D, 4]
        surv_conf  = conf_cand[keep]                               # [D]
        surv_cls   = cls_cand[keep]                                # [D]
        surv_anchor = anchor_idx_cand[keep]                        # [D]
    else:
        surv_boxes = boxes_xyxy_all.new_zeros(0, 4)
        surv_conf  = max_conf.new_zeros(0)
        surv_cls   = pred_cls.new_zeros(0)
        surv_anchor = torch.zeros(0, dtype=torch.long, device=device)

    D = surv_boxes.shape[0]

    # ----- match each survivor to GTs -----
    fp_anchor_list, fp_class_list, fp_source_list = [], [], []
    fp_bbox_list, tp_bbox_list = [], []
    matched_gt = torch.zeros(K_gt, dtype=torch.bool, device=device)

    if D > 0 and K_gt > 0:
        # NMS survivors come out of batched_nms sorted by descending confidence.
        # Iterate in that order so highest-conf detection claims the GT first;
        # later duplicates become FPs (matches COCO eval convention).
        ious = _iou_matrix(surv_boxes, bboxes_xyxy.to(device).float())  # [D, K]
        gt_cls = classes.to(device).long()
        for d in range(D):
            row = ious[d].clone()
            row[matched_gt] = -1.0                         # taken GTs ineligible
            k_best = int(row.argmax().item())
            best_iou = float(row[k_best].item())
            has_overlap = best_iou >= match_iou_thresh
            cls_match = has_overlap and int(surv_cls[d]) == int(gt_cls[k_best])
            if cls_match:
                matched_gt[k_best] = True
                tp_bbox_list.append(surv_boxes[d])
            else:
                # FP: BG hallucination, class mismatch, or duplicate of taken GT
                fp_anchor_list.append(int(surv_anchor[d]))
                fp_class_list.append(int(surv_cls[d]))
                fp_source_list.append(0)
                fp_bbox_list.append(surv_boxes[d])
    elif D > 0:
        # No GTs -> all detections are FPs
        for d in range(D):
            fp_anchor_list.append(int(surv_anchor[d]))
            fp_class_list.append(int(surv_cls[d]))
            fp_source_list.append(0)
            fp_bbox_list.append(surv_boxes[d])

    n_fp_nms = len(fp_anchor_list)
    n_tp = D - n_fp_nms

    # ----- borderline BG pool (below threshold, outside all GTs) -----
    borderline_mask = (
        (max_conf > borderline_conf_low) &
        (max_conf <= nms_conf_thresh) &
        (~inside_any)
    )
    borderline_idx = torch.nonzero(borderline_mask, as_tuple=False).squeeze(-1)
    n_fp_borderline = int(borderline_idx.numel())
    for i in range(n_fp_borderline):
        ai = int(borderline_idx[i])
        fp_anchor_list.append(ai)
        fp_class_list.append(int(pred_cls[ai]))
        fp_source_list.append(1)
        # no bbox needed for borderline (not used in preserve mask)

    # ----- stack outputs -----
    fp_anchor_idx = torch.tensor(fp_anchor_list, dtype=torch.long, device=device)
    fp_class_idx  = torch.tensor(fp_class_list,  dtype=torch.long, device=device)
    fp_source     = torch.tensor(fp_source_list, dtype=torch.int8, device=device)
    fp_bboxes     = torch.stack(fp_bbox_list) if fp_bbox_list \
                    else boxes_xyxy_all.new_zeros(0, 4)
    tp_bboxes     = torch.stack(tp_bbox_list) if tp_bbox_list \
                    else boxes_xyxy_all.new_zeros(0, 4)
    n_miss = int((~matched_gt).sum().item()) if K_gt > 0 else 0

    return dict(
        fp_anchor_idx=fp_anchor_idx,
        fp_class_idx=fp_class_idx,
        fp_source=fp_source,
        fp_bboxes_xyxy=fp_bboxes,
        tp_bboxes_xyxy=tp_bboxes,
        n_fp_nms=n_fp_nms,
        n_fp_borderline=n_fp_borderline,
        n_tp=n_tp,
        n_miss=n_miss,
        n_gt=K_gt,
    )


# =========================================================================
# 2. Differentiable FP suppression loss
# =========================================================================

def fp_suppression_loss(
    preds_with_grad: torch.Tensor,
    fp_anchor_idx: torch.Tensor,
    fp_class_idx: torch.Tensor,
    gamma_focal: float = 2.0,
    eps: float = 1e-6,
    n_classes: int = 9,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Focal-weighted BCE-for-y=0 loss on specific (anchor, class) pairs.

    L_FP = sum_i [ (p_i.detach()^gamma) * -log(1 - p_i) ]

    where p_i is the sigmoid probability at FP anchor i for its predicted
    class. Gradient flows only to these specific (anchor, class) cells.

    Args:
        preds_with_grad: [1, 4+nc, N] - preds from forward() with grad
        fp_anchor_idx:   [K_fp] - anchor indices (from find_fp_anchors)
        fp_class_idx:    [K_fp] - predicted class at each FP
        gamma_focal:     focal exponent (Lin 2017 default=2)

    Returns:
        L_FP: scalar loss (sum over FPs, not mean)
        stats: dict with monitoring scalars
    """
    K_fp = fp_anchor_idx.numel()
    if K_fp == 0:
        return preds_with_grad.new_zeros(()), dict(
            n_fp=0, p_fp_mean=0.0, p_fp_max=0.0, L_FP=0.0,
        )

    # preds shape: [1, 4+nc, N]
    # index cls_scores[class_idx, anchor_idx]
    cls_scores = preds_with_grad[0, 4 : 4 + n_classes, :]          # [nc, N]
    p_fp = cls_scores[fp_class_idx, fp_anchor_idx]                 # [K_fp]
    p_fp = p_fp.clamp(min=eps, max=1.0 - eps)                      # numerical safety

    # Focal weight (detached - standard focal loss practice)
    focal_w = p_fp.detach().pow(gamma_focal)                       # [K_fp]

    # BCE for y=0: -log(1 - p)
    bce_y0 = -torch.log(1.0 - p_fp)                                # [K_fp]

    L_FP = (focal_w * bce_y0).sum()

    return L_FP, dict(
        n_fp=int(K_fp),
        p_fp_mean=float(p_fp.mean().item()),
        p_fp_max=float(p_fp.max().item()),
        L_FP=float(L_FP.item()),
    )


# =========================================================================
# 3. Preserve mask (hybrid: I_base FPs U I_fine FPs, minus TP_base)
# =========================================================================

def build_preserve_mask_v3_2(
    H: int,
    W: int,
    fp_bboxes_base: torch.Tensor,   # [K_b, 4] from find_fp_anchors on I_base
    fp_bboxes_fine: torch.Tensor,   # [K_f, 4] from find_fp_anchors on I_fine
    tp_bboxes_base: torch.Tensor,   # [K_t, 4] protected (no dilation)
    dilate_px: int = 20,
    area_cap: float = 0.40,
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Preserve mask for v3.2.

    mask[y, x] = 1 where delta is ALLOWED to change (FP regions, dilated)
                 0 where delta MUST match I_base (TP, clean bg, MISS regions)

    Args:
        fp_bboxes_base: NMS-survived FPs in I_base (the "known bad" list)
        fp_bboxes_fine: NMS-survived FPs in I_fine (current delta output)
                        Union catches emergent FPs created during training.
        tp_bboxes_base: NMS-survived TPs in I_base. Always protected.
        dilate_px:      halo around FP bboxes. 20 px at 1024 ~ 1/4 component.
        area_cap:       if allow mask covers more than this fraction,
                        force full lock (safety against runaway FP cluster).

    Returns:
        mask: [H, W] float32, values in {0.0, 1.0}
        stats: dict with coverages (all floats in [0, 1])
    """
    if device is None:
        device = fp_bboxes_base.device if fp_bboxes_base.numel() > 0 \
                 else fp_bboxes_fine.device
    allow = torch.zeros(H, W, device=device, dtype=torch.float32)
    tp_protect = torch.zeros(H, W, device=device, dtype=torch.float32)

    # Union of FP bboxes with dilation
    for bboxes in (fp_bboxes_base, fp_bboxes_fine):
        for x1, y1, x2, y2 in bboxes.tolist():
            xa = max(0, int(x1 - dilate_px))
            ya = max(0, int(y1 - dilate_px))
            xb = min(W, int(x2 + dilate_px))
            yb = min(H, int(y2 + dilate_px))
            if xa < xb and ya < yb:
                allow[ya:yb, xa:xb] = 1.0

    # TP protection (no dilation, tight)
    for x1, y1, x2, y2 in tp_bboxes_base.tolist():
        xa = max(0, int(x1))
        ya = max(0, int(y1))
        xb = min(W, int(x2))
        yb = min(H, int(y2))
        if xa < xb and ya < yb:
            tp_protect[ya:yb, xa:xb] = 1.0

    mask = allow * (1.0 - tp_protect)
    cov_raw    = float(allow.mean().item())
    cov_tp     = float(tp_protect.mean().item())
    cov_final  = float(mask.mean().item())

    # Safety: if too much of the image is allowed to change, fall back
    # to full lock for this sample (protects against dense FP runaway).
    if cov_final > area_cap:
        mask = torch.zeros_like(mask)

    stats = dict(
        preserve_allow_cov=cov_raw,
        preserve_tp_cov=cov_tp,
        preserve_final_cov=float(mask.mean().item()),
        preserve_capped=(cov_final > area_cap),
    )
    return mask, stats


# =========================================================================
# 4. Monitoring helper (no_grad) for val eval
# =========================================================================

@torch.no_grad()
def compute_f1_ap50(
    reward_model: YOLOv8Reward,
    img_01: torch.Tensor,
    bboxes_xyxy: torch.Tensor,
    classes: torch.Tensor,
    nms_conf_thresh: float = 0.25,
    nms_iou_thresh: float = 0.45,
    match_iou_thresh: float = 0.30,
) -> Dict[str, float]:
    """F1 + mAP@0.5 for a single image. Used by val eval.

    Precision and recall are computed at conf >= nms_conf_thresh
    (matches the eval pipeline). mAP@0.5 integrates over confidence.
    """
    info = find_fp_anchors(
        reward_model, img_01, bboxes_xyxy, classes,
        nms_conf_thresh=nms_conf_thresh,
        nms_iou_thresh=nms_iou_thresh,
        match_iou_thresh=match_iou_thresh,
        borderline_conf_low=1.0,  # disable borderline pool for eval
    )
    tp, fp, miss = info["n_tp"], info["n_fp_nms"], info["n_miss"]
    n_gt = info["n_gt"]

    prec = tp / max(tp + fp, 1)
    rec  = tp / max(n_gt, 1) if n_gt > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # mAP@0.5: per-class AP via interpolated PR curve over confidence.
    # For simplicity, compute a scalar mAP by running the full cls_score
    # grid through an all-confidence PR sweep per class.
    preds = reward_model.forward(img_01)
    nc = reward_model.n_classes
    cxcywh = preds[0, :4, :].T
    cx, cy, w, h = cxcywh.unbind(-1)
    boxes_xyxy_all = torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
    )
    cls_scores = preds[0, 4 : 4 + nc, :].T                         # [N, nc]
    max_conf, pred_cls = cls_scores.max(dim=-1)

    # Only consider anchors with pred_cls matching a GT class present
    device = img_01.device
    gt_classes_set = set(classes.cpu().tolist())
    aps = []
    for c in range(nc):
        # Anchors predicting this class
        mask_c = (pred_cls == c)
        if not mask_c.any():
            if c in gt_classes_set:
                aps.append(0.0)
            continue
        # Take top candidates (sort by conf desc)
        conf_c = max_conf[mask_c]
        boxes_c = boxes_xyxy_all[mask_c]
        order = conf_c.argsort(descending=True)
        conf_c = conf_c[order][:1000]  # cap for speed
        boxes_c = boxes_c[order][:1000]

        # Apply NMS within class
        if conf_c.numel() > 0:
            keep = batched_nms(
                boxes_c, conf_c,
                torch.zeros_like(conf_c, dtype=torch.long),
                nms_iou_thresh,
            )
            boxes_c = boxes_c[keep]
            conf_c = conf_c[keep]

        # GTs of this class
        gt_mask = (classes == c)
        gt_c = bboxes_xyxy[gt_mask].to(device).float() if gt_mask.any() \
               else bboxes_xyxy.new_zeros(0, 4)
        n_gt_c = gt_c.shape[0]
        if n_gt_c == 0 and conf_c.numel() == 0:
            continue
        if n_gt_c == 0:
            aps.append(0.0)
            continue

        # Match detections to GTs in conf-desc order; each GT can match once
        ious_c = _iou_matrix(boxes_c, gt_c) if conf_c.numel() > 0 \
                 else boxes_c.new_zeros(0, n_gt_c)
        matched = torch.zeros(n_gt_c, dtype=torch.bool, device=device)
        tp_arr = torch.zeros(conf_c.numel(), dtype=torch.float32)
        fp_arr = torch.zeros(conf_c.numel(), dtype=torch.float32)
        for i in range(conf_c.numel()):
            if n_gt_c == 0:
                fp_arr[i] = 1.0
                continue
            m_iou = ious_c[i]
            m_iou = m_iou.clone()
            m_iou[matched] = -1.0  # already-matched GTs ineligible
            best = m_iou.argmax()
            if m_iou[best] >= match_iou_thresh:
                matched[best] = True
                tp_arr[i] = 1.0
            else:
                fp_arr[i] = 1.0

        # PR curve
        cum_tp = tp_arr.cumsum(0)
        cum_fp = fp_arr.cumsum(0)
        prec_arr = cum_tp / (cum_tp + cum_fp + 1e-6)
        rec_arr  = cum_tp / n_gt_c

        # 11-point interpolated AP
        ap = 0.0
        for t in [i * 0.1 for i in range(11)]:
            mask_t = rec_arr >= t
            if mask_t.any():
                ap += prec_arr[mask_t].max().item()
        ap /= 11.0
        aps.append(ap)

    map50 = sum(aps) / max(len(aps), 1)

    return dict(
        f1=float(f1), precision=float(prec), recall=float(rec),
        map50=float(map50),
        n_tp=tp, n_fp=fp, n_miss=miss, n_gt=n_gt,
    )
