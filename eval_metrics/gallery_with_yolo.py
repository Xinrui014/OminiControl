"""
Generate gallery comparison with YOLO detection overlays.
Green box = correct detection (class + IoU match)
Red box = wrong class (IoU match but wrong category)
Yellow box = hallucinated (no GT match)
Blue box = missed GT (not detected)
"""
import json
import os
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Paths
YOLO_MODEL_PATH = "/home/xinrui/projects/PCB_structure_layout/re-annotation/runs/yolov8m_pcb_v6/weights/best.pt"
EVAL_JSON = "/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/eval_patches_full_test.json"
GALLERY_HTML = "/home/xinrui/projects/OminiControl/eval_metrics/gallery_comparison.html"

MODELS = {
    "v3_newdata ckpt14k": "/home/xinrui/projects/OminiControl/runs/v3_newdata/eval_full/ckpt14k",
    "v3.1_1024 ckpt6k": "/home/xinrui/projects/OminiControl/runs/v3.1_newdata_1024/eval_full/ckpt6k",
}

OUTPUT_DIR = "/home/xinrui/projects/OminiControl/eval_metrics/gallery_yolo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO class index -> our category ID
YOLO_TO_CAT = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 6: 8, 7: 9, 8: 10}
CAT_NAMES = {1: "R", 2: "C", 3: "L", 4: "J", 5: "D", 7: "SW", 8: "Q", 9: "IC", 10: "OSC"}

# Load YOLO
print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL_PATH)

# Load eval patches JSON for GT
print("Loading eval patches...")
with open(EVAL_JSON) as f:
    eval_patches = json.load(f)
# Index by patch_id
gt_by_patch = {p["patch_id"]: p for p in eval_patches}

# Extract patch IDs from existing gallery
print("Parsing existing gallery...")
with open(GALLERY_HTML) as f:
    html_content = f.read()
patch_ids = re.findall(r'<h3>([^<]+)</h3>', html_content)
print(f"Found {len(patch_ids)} patches in gallery")


def compute_iou(box_a, box_b):
    """box format: [x1, y1, x2, y2]"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def draw_detections(img, detections, gt_annotations, patch_id):
    """
    Draw detection results on image.
    Returns annotated image and stats dict.
    """
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except:
        font = ImageFont.load_default()

    # Convert GT annotations to [x1, y1, x2, y2, cat_id] format
    gt_boxes = []
    for ann in gt_annotations:
        bx, by, bw, bh = ann["bbox"]
        gt_boxes.append([bx, by, bx + bw, by + bh, ann["category_id"]])

    # Match detections to GT
    matched_gt = set()
    stats = {"correct": 0, "wrong_class": 0, "hallucinated": 0, "missed": 0}

    det_results = []  # (box, cat_id, conf, match_type)

    for det in detections:
        box = det[:4]  # x1, y1, x2, y2
        det_cat = YOLO_TO_CAT.get(int(det[5]), -1)
        conf = float(det[4])

        best_iou = 0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(box, gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= 0.3 and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            gt_cat = gt_boxes[best_gt_idx][4]
            if det_cat == gt_cat:
                det_results.append((box, det_cat, conf, "correct"))
                stats["correct"] += 1
            else:
                det_results.append((box, det_cat, conf, "wrong_class", gt_cat))
                stats["wrong_class"] += 1
        else:
            det_results.append((box, det_cat, conf, "hallucinated"))
            stats["hallucinated"] += 1

    # Count missed (but don't draw them)
    for gi, gt in enumerate(gt_boxes):
        if gi not in matched_gt:
            stats["missed"] += 1

    # Draw detections — thicker lines, no confidence score
    COLORS = {"correct": (0, 255, 0), "wrong_class": (255, 0, 0), "hallucinated": (255, 255, 0)}
    for result in det_results:
        box = result[0]
        det_cat = result[1]
        match_type = result[3]

        x1, y1, x2, y2 = [int(v) for v in box]
        color = COLORS[match_type]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        cat_name = CAT_NAMES.get(det_cat, "?")
        if match_type == "wrong_class":
            gt_cat = result[4]
            gt_name = CAT_NAMES.get(gt_cat, "?")
            label = f"{cat_name}>{gt_name}"
        elif match_type == "hallucinated":
            label = f"H:{cat_name}"
        else:
            label = cat_name
        draw.text((x1, y1 - 13), label, fill=color, font=font)

    return img, stats


# Process each patch
all_results = []
for patch_id in patch_ids:
    gt_info = gt_by_patch.get(patch_id)
    if gt_info is None:
        print(f"  SKIP {patch_id} — not in eval JSON")
        continue

    gt_anns = gt_info.get("annotations", [])

    patch_results = {"patch_id": patch_id, "board": gt_info.get("board_name", ""), "models": {}}

    for model_name, model_dir in MODELS.items():
        harmonized_path = os.path.join(model_dir, f"{patch_id}_harmonized.png")
        if not os.path.exists(harmonized_path):
            print(f"  SKIP {patch_id} / {model_name} — harmonized not found")
            continue

        # Load harmonized image
        img = Image.open(harmonized_path).convert("RGB")

        # Run YOLO at original resolution
        results = yolo(img, conf=0.25, iou=0.5, verbose=False)
        dets = results[0].boxes.data.cpu().numpy() if len(results[0].boxes) > 0 else []

        # Upscale 2x for sharper bbox drawing
        w, h = img.size
        img_up = img.resize((w * 2, h * 2), Image.LANCZOS)
        # Scale detections and GT to 2x
        dets_scaled = dets.copy()
        dets_scaled[:, :4] *= 2 if len(dets) > 0 else None
        gt_anns_scaled = []
        for ann in gt_anns:
            ann_s = dict(ann)
            bx, by, bw, bh = ann["bbox"]
            ann_s["bbox"] = [bx * 2, by * 2, bw * 2, bh * 2]
            gt_anns_scaled.append(ann_s)

        # Draw detections on upscaled image
        img_annotated, stats = draw_detections(img_up, dets_scaled, gt_anns_scaled, patch_id)

        # Save annotated image
        safe_model = model_name.replace(" ", "_").replace(".", "_")
        out_name = f"{patch_id}_{safe_model}_yolo.jpg"
        img_annotated.save(os.path.join(OUTPUT_DIR, out_name), quality=92)

        patch_results["models"][model_name] = {
            "yolo_img": out_name,
            "stats": stats,
            "n_det": len(dets),
            "n_gt": len(gt_anns),
        }

    if patch_results["models"]:
        all_results.append(patch_results)
    print(f"  {patch_id}: {len(gt_anns)} GT, " +
          ", ".join(f"{m}: {r['stats']}" for m, r in patch_results["models"].items()))

# Generate HTML gallery
print(f"\nGenerating HTML gallery with {len(all_results)} patches...")

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PCB Harmonization — YOLO Detection Comparison</title>
<style>
body{background:#1a1a2e;color:#eee;font-family:sans-serif;max-width:1800px;margin:auto;padding:20px}
h1{color:#4ECDC4;text-align:center}
h2{color:#4ECDC4;margin-top:30px}
.legend{text-align:center;margin:10px;font-size:14px}
.legend span{margin:0 12px;padding:2px 8px;border-radius:4px}
.sample{background:#16213e;border-radius:12px;padding:16px;margin:20px auto}
h3{color:#FF6B6B;margin:0 0 4px}
.meta{color:#4ECDC4;font-size:13px;margin:4px 0}
.row{display:flex;gap:10px;margin:8px 0}
.col{flex:1}
.col img{width:100%;border-radius:6px}
.model-name{color:#FFD93D;font-size:14px;font-weight:bold;margin:4px 0}
.stats{font-size:12px;color:#aaa;margin:2px 0}
.stats .bad{color:#FF6B6B}
.stats .good{color:#4ECDC4}
table{border-collapse:collapse;width:100%;margin:10px 0}
th,td{border:1px solid #333;padding:8px;text-align:center;font-size:13px}
th{background:#16213e;color:#4ECDC4}
td{background:#0f1a2e}
</style></head><body>
<h1>PCB Harmonization — YOLO Detection Results</h1>
<div class="legend">
    <span style="background:#004400;color:#0f0">■ Correct</span>
    <span style="background:#440000;color:#f00">■ Wrong Class</span>
    <span style="background:#444400;color:#ff0">■ Hallucinated</span>
</div>
<p style="text-align:center;color:#aaa">50 patches — YOLO detections overlaid on harmonized images</p>

<h2>Aggregate Summary</h2>
<table>
<tr><th>Model</th><th>Correct</th><th>Wrong Class</th><th>Hallucinated</th></tr>
"""

# Compute aggregate stats
for model_name in MODELS:
    agg = {"correct": 0, "wrong_class": 0, "hallucinated": 0, "missed": 0}
    for r in all_results:
        if model_name in r["models"]:
            for k in agg:
                agg[k] += r["models"][model_name]["stats"][k]
    total_det = agg["correct"] + agg["wrong_class"] + agg["hallucinated"]
    html += f"""<tr><td>{model_name}</td>
<td>{agg['correct']} ({100*agg['correct']/max(total_det,1):.0f}%)</td>
<td>{agg['wrong_class']} ({100*agg['wrong_class']/max(total_det,1):.0f}%)</td>
<td>{agg['hallucinated']}</td></tr>\n"""

html += "</table>\n<h2>Per-Patch Comparison</h2>\n"

# Sort: most errors first
def error_score(r):
    total = 0
    for m in r["models"].values():
        s = m["stats"]
        total += s["wrong_class"] * 3 + s["hallucinated"] * 2 + s["missed"]
    return -total

all_results.sort(key=error_score)

for r in all_results:
    patch_id = r["patch_id"]
    html += f"""<div class="sample">
<h3>{patch_id}</h3>
<p class="meta">Board: {r['board']}</p>
<div class="row">\n"""

    for model_name in MODELS:
        if model_name not in r["models"]:
            continue
        m = r["models"][model_name]
        s = m["stats"]
        html += f"""<div class="col">
<p class="model-name">{model_name}</p>
<p class="stats"><span class="good">✓{s['correct']}</span>
<span class="bad">✗class:{s['wrong_class']}</span>
<span class="bad">halluc:{s['hallucinated']}</span>
(det:{m['n_det']}/gt:{m['n_gt']})</p>
<img src="{m['yolo_img']}">
</div>\n"""

    html += "</div></div>\n"

html += "</body></html>"

out_html = os.path.join(OUTPUT_DIR, "gallery.html")
with open(out_html, "w") as f:
    f.write(html)

print(f"\nGallery saved to {out_html}")
print(f"View at: http://10.97.27.230:8877/OminiControl/eval_metrics/gallery_yolo/gallery.html")
