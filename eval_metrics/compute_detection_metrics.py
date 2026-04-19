#!/usr/bin/env python3
"""
Compute detection-based evaluation metrics for PCB harmonization.

Runs YOLO on harmonized images, compares detections against GT annotations.
Metrics: position accuracy (mean IoU), class accuracy, hallucination rate,
and standard COCO AP via pycocotools.

Usage:
    python eval_metrics/compute_detection_metrics.py \
        --eval_json /home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/eval_patches_full_test.json \
        --gen_dir runs/v3_newdata/eval_full/ckpt14k \
        --gen_suffix _harmonized.png \
        --output runs/v3_newdata/eval_full/ckpt14k/detection_metrics.json \
        --device cuda:0
"""
import json
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

# YOLO class index -> COCO category ID (same IDs as our GT)
YOLO_TO_COCO = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 6: 8, 7: 9, 8: 10}
CAT_ID_TO_NAME = {
    1: "RESISTOR", 2: "CAPACITOR", 3: "INDUCTOR", 4: "CONNECTOR",
    5: "DIODE", 7: "SWITCH", 8: "TRANSISTOR", 9: "IC", 10: "OSCILLATOR",
}
YOLO_MODEL_PATH = "/home/xinrui/projects/PCB_structure_layout/re-annotation/runs/yolov8m_pcb_v6/weights/best.pt"


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def compute_iou_matrix(det_boxes, gt_boxes):
    """Compute IoU matrix between all detection and GT boxes."""
    n_det = len(det_boxes)
    n_gt = len(gt_boxes)
    iou_matrix = np.zeros((n_det, n_gt))
    for i in range(n_det):
        for j in range(n_gt):
            iou_matrix[i, j] = compute_iou(det_boxes[i], gt_boxes[j])
    return iou_matrix


def greedy_match(iou_matrix, iou_threshold=0.5):
    """Greedy matching: highest IoU first, each GT matched at most once.
    Returns list of (det_idx, gt_idx, iou) for matches above threshold."""
    n_det, n_gt = iou_matrix.shape
    if n_det == 0 or n_gt == 0:
        return []

    matches = []
    matched_gt = set()
    matched_det = set()

    # Flatten and sort by IoU descending
    pairs = []
    for i in range(n_det):
        for j in range(n_gt):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    for iou, di, gi in pairs:
        if di not in matched_det and gi not in matched_gt:
            matches.append((di, gi, iou))
            matched_det.add(di)
            matched_gt.add(gi)

    return matches


def classify_fp(det_box, gt_boxes):
    """Classify a false positive as mislocalized (0.1 < max_iou < 0.5)
    or hallucinated (max_iou < 0.1)."""
    if len(gt_boxes) == 0:
        return "hallucinated"
    max_iou = max(compute_iou(det_box, gb) for gb in gt_boxes)
    if max_iou >= 0.1:
        return "mislocalized"
    return "hallucinated"


def run_yolo_on_image(model, image_path, conf=0.3):
    """Run YOLO inference on a single image, return list of detections."""
    results = model(image_path, conf=conf, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            # xyxy -> xywh
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu().item())
            score = float(boxes.conf[i].cpu().item())
            coco_id = YOLO_TO_COCO.get(cls_id)
            if coco_id is None:
                continue
            detections.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "category_id": coco_id,
                "category_name": CAT_ID_TO_NAME[coco_id],
                "score": score,
            })
    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--gen_suffix", default="_harmonized.png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--yolo_model", default=YOLO_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching")
    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load eval JSON
    with open(args.eval_json) as f:
        eval_patches = json.load(f)
    print(f"Loaded {len(eval_patches)} patches from eval JSON")

    # Load YOLO model
    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)
    model.to(args.device)

    # ── Per-image evaluation ──
    all_matched_ious = []
    all_class_correct = []  # (correct, total) among matched
    all_fp_per_image = []
    all_fp_hallucinated = []
    all_fp_mislocalized = []
    all_fn_per_image = []
    all_n_det = []
    all_n_gt = []

    # For confusion matrix
    confusion = defaultdict(int)  # (gt_cat, pred_cat) -> count

    # For pycocotools
    coco_gt_anns = []
    coco_dt_anns = []
    coco_images = []
    ann_id = 1

    processed = 0
    skipped = 0

    for idx, patch in enumerate(eval_patches):
        patch_id = patch["patch_id"]
        gt_annotations = patch["annotations"]
        crop_size = patch["crop_size"]

        # Check generated image exists
        gen_path = gen_dir / f"{patch_id}{args.gen_suffix}"
        if not gen_path.exists():
            skipped += 1
            continue

        # Run YOLO
        detections = run_yolo_on_image(model, str(gen_path), conf=args.conf)

        # GT boxes and classes
        gt_boxes = [a["bbox"] for a in gt_annotations]
        gt_cats = [a["category_id"] for a in gt_annotations]
        det_boxes = [d["bbox"] for d in detections]
        det_cats = [d["category_id"] for d in detections]

        # IoU matrix and matching (class-agnostic matching for class accuracy eval)
        iou_matrix = compute_iou_matrix(det_boxes, gt_boxes)
        matches = greedy_match(iou_matrix, iou_threshold=args.iou_threshold)

        matched_det_idx = set(m[0] for m in matches)
        matched_gt_idx = set(m[1] for m in matches)

        # 1. Position accuracy: mean IoU of matched pairs
        for di, gi, iou in matches:
            all_matched_ious.append(iou)

        # 2. Class accuracy among matched pairs
        n_class_correct = 0
        for di, gi, iou in matches:
            pred_cat = det_cats[di]
            gt_cat = gt_cats[gi]
            confusion[(gt_cat, pred_cat)] += 1
            if pred_cat == gt_cat:
                n_class_correct += 1
        all_class_correct.append((n_class_correct, len(matches)))

        # 3. False positives (unmatched detections)
        n_fp = 0
        n_fp_hallucinated = 0
        n_fp_mislocalized = 0
        for di in range(len(detections)):
            if di not in matched_det_idx:
                n_fp += 1
                fp_type = classify_fp(det_boxes[di], gt_boxes)
                if fp_type == "hallucinated":
                    n_fp_hallucinated += 1
                else:
                    n_fp_mislocalized += 1

        # False negatives (unmatched GT)
        n_fn = len(gt_annotations) - len(matched_gt_idx)

        all_fp_per_image.append(n_fp)
        all_fp_hallucinated.append(n_fp_hallucinated)
        all_fp_mislocalized.append(n_fp_mislocalized)
        all_fn_per_image.append(n_fn)
        all_n_det.append(len(detections))
        all_n_gt.append(len(gt_annotations))

        # Build COCO format for pycocotools
        image_id = idx
        coco_images.append({
            "id": image_id,
            "width": crop_size,
            "height": crop_size,
            "file_name": f"{patch_id}{args.gen_suffix}",
        })
        for a in gt_annotations:
            coco_gt_anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": a["category_id"],
                "bbox": a["bbox"],
                "area": a["bbox"][2] * a["bbox"][3],
                "iscrowd": 0,
            })
            ann_id += 1
        for d in detections:
            coco_dt_anns.append({
                "image_id": image_id,
                "category_id": d["category_id"],
                "bbox": d["bbox"],
                "score": d["score"],
            })

        processed += 1
        if processed % 200 == 0:
            print(f"  [{processed}/{len(eval_patches)}] det={len(detections)} gt={len(gt_annotations)} matched={len(matches)} fp={n_fp}")

    print(f"\nProcessed: {processed}, Skipped (no image): {skipped}")

    # ── Aggregate metrics ──
    total_tp = len(all_matched_ious)
    total_fp = sum(all_fp_per_image)
    total_fn = sum(all_fn_per_image)
    total_gt = sum(all_n_gt)
    total_det = sum(all_n_det)

    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    total_class_correct = sum(c for c, t in all_class_correct)
    total_class_total = sum(t for c, t in all_class_correct)
    class_accuracy = total_class_correct / total_class_total if total_class_total > 0 else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    fp_per_image = float(np.mean(all_fp_per_image))
    fp_hallucinated_per_image = float(np.mean(all_fp_hallucinated))
    fp_mislocalized_per_image = float(np.mean(all_fp_mislocalized))

    print(f"\n{'='*60}")
    print(f"  Total GT components:     {total_gt}")
    print(f"  Total detections:        {total_det}")
    print(f"  True positives:          {total_tp}")
    print(f"  False positives:         {total_fp}")
    print(f"  False negatives:         {total_fn}")
    print(f"  Mean IoU (matched):      {mean_iou:.4f}")
    print(f"  Classification accuracy: {class_accuracy:.4f}")
    print(f"  Precision:               {precision:.4f}")
    print(f"  Recall:                  {recall:.4f}")
    print(f"  F1:                      {f1:.4f}")
    print(f"  FP/image:                {fp_per_image:.2f}")
    print(f"    Hallucinated/image:    {fp_hallucinated_per_image:.2f}")
    print(f"    Mislocalized/image:    {fp_mislocalized_per_image:.2f}")
    print(f"{'='*60}")

    # ── Per-class breakdown ──
    per_class = {}
    for cat_id, cat_name in CAT_ID_TO_NAME.items():
        # Count GT, TP, FP for this class
        cls_gt = sum(1 for p in eval_patches for a in p["annotations"] if a["category_id"] == cat_id)
        cls_tp = sum(1 for (gc, pc), cnt in confusion.items() if gc == cat_id and pc == cat_id for _ in range(cnt))
        cls_misclassified = sum(cnt for (gc, pc), cnt in confusion.items() if gc == cat_id and pc != cat_id)
        cls_recall = cls_tp / cls_gt if cls_gt > 0 else 0.0
        per_class[cat_name] = {
            "n_gt": cls_gt,
            "n_tp": cls_tp,
            "n_misclassified": cls_misclassified,
            "recall": round(cls_recall, 4),
        }

    # ── COCO AP via pycocotools ──
    ap_results = {}
    if coco_gt_anns and coco_dt_anns:
        print("\nComputing COCO AP via pycocotools...")
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        categories = [{"id": cid, "name": cname} for cid, cname in CAT_ID_TO_NAME.items()]
        coco_gt_dict = {
            "images": coco_images,
            "annotations": coco_gt_anns,
            "categories": categories,
        }

        # Write GT to temp file (pycocotools needs a file)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(coco_gt_dict, f)
            gt_file = f.name

        coco_gt_obj = COCO(gt_file)
        coco_dt_obj = coco_gt_obj.loadRes(coco_dt_anns)

        coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap_results = {
            "AP_0.5_0.95": round(coco_eval.stats[0], 4),
            "AP_0.5": round(coco_eval.stats[1], 4),
            "AP_0.75": round(coco_eval.stats[2], 4),
            "AP_small": round(coco_eval.stats[3], 4),
            "AP_medium": round(coco_eval.stats[4], 4),
            "AP_large": round(coco_eval.stats[5], 4),
            "AR_1": round(coco_eval.stats[6], 4),
            "AR_10": round(coco_eval.stats[7], 4),
            "AR_100": round(coco_eval.stats[8], 4),
        }

        import os
        os.unlink(gt_file)

    # ── Build confusion matrix (readable) ──
    confusion_readable = {}
    for (gc, pc), cnt in sorted(confusion.items()):
        gt_name = CAT_ID_TO_NAME.get(gc, str(gc))
        pred_name = CAT_ID_TO_NAME.get(pc, str(pc))
        key = f"{gt_name} -> {pred_name}"
        confusion_readable[key] = cnt

    # ── Save results ──
    results = {
        "n_images": processed,
        "n_gt_total": total_gt,
        "n_det_total": total_det,
        "mean_iou": round(mean_iou, 4),
        "classification_accuracy": round(class_accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_per_image": round(fp_per_image, 4),
        "fp_hallucinated_per_image": round(fp_hallucinated_per_image, 4),
        "fp_mislocalized_per_image": round(fp_mislocalized_per_image, 4),
        "coco_ap": ap_results,
        "per_class": per_class,
        "confusion_matrix": confusion_readable,
        "config": {
            "yolo_model": args.yolo_model,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou_threshold,
            "gen_suffix": args.gen_suffix,
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
