#!/usr/bin/env python3
"""
Generate side-by-side comparison gallery for multiple model outputs.
Uses the same sampled patches across all models.

Usage:
    python eval_metrics/generate_gallery.py \
        --sample_ids eval_metrics/gallery_sample_ids.json \
        --dirs runs/v3_newdata/eval_full/ckpt14k runs/v3.1_newdata_1024/eval_full/ckpt6k \
        --names "v3_newdata_14k (512-trained)" "v3.1_1024_6k (1024-trained)" \
        --output eval_metrics/gallery_comparison.html
"""
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_ids", required=True)
    parser.add_argument("--eval_json", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/eval_patches_full_test.json")
    parser.add_argument("--dirs", nargs="+", required=True, help="Output dirs for each model")
    parser.add_argument("--names", nargs="+", required=True, help="Display names for each model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.sample_ids) as f:
        sample_ids = json.load(f)

    with open(args.eval_json) as f:
        all_patches = json.load(f)
    patch_map = {p["patch_id"]: p for p in all_patches}

    dirs = [Path(d) for d in args.dirs]
    names = args.names

    # Build rows
    rows = []
    for pid in sample_ids:
        patch = patch_map.get(pid)
        if patch is None:
            continue

        meta = f"Board: {patch['board']} | Color: {patch['board_color']} | Components: {patch['n_components']}"

        # Each model's comparison image (4-panel: bbox | composite | harmonized | original)
        model_imgs = []
        for d, name in zip(dirs, names):
            comp_path = d / f"{pid}_comparison.png"
            if comp_path.exists():
                # HTTP server root is /home/xinrui/projects/, so prefix with OminiControl/
                web_path = f"/OminiControl/{comp_path}"
                model_imgs.append(f"""
                <div class="model">
                  <p class="model-name">{name}</p>
                  <img src="{web_path}" style="width:100%;border-radius:6px">
                </div>""")
            else:
                model_imgs.append(f'<div class="model"><p class="model-name">{name}</p><p style="color:#f66">Image not found</p></div>')

        rows.append(f"""
        <div class="sample">
          <h3>{pid}</h3>
          <p class="meta">{meta}</p>
          <p class="labels">Panels: bbox | composite | harmonized (1024→512) | original</p>
          {"".join(model_imgs)}
        </div>""")

    # Load metrics if available
    metrics_rows = []
    for d, name in zip(dirs, names):
        fid_path = d / "fid_kid_metrics.json"
        det_path = d / "detection_metrics.json"
        fid_data = json.load(open(fid_path)) if fid_path.exists() else {}
        det_data = json.load(open(det_path)) if det_path.exists() else {}
        metrics_rows.append(f"""
        <tr>
          <td>{name}</td>
          <td>{fid_data.get('fid', 'N/A')}</td>
          <td>{fid_data.get('kid', 'N/A')}</td>
          <td>{det_data.get('mean_iou', 'N/A')}</td>
          <td>{det_data.get('classification_accuracy', 'N/A')}</td>
          <td>{det_data.get('precision', 'N/A')}</td>
          <td>{det_data.get('recall', 'N/A')}</td>
          <td>{det_data.get('fp_per_image', 'N/A')}</td>
          <td>{det_data.get('fp_hallucinated_per_image', 'N/A')}</td>
          <td>{det_data.get('coco_ap', {}).get('AP_0.5', 'N/A')}</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PCB Harmonization Comparison</title>
<style>
body{{background:#1a1a2e;color:#eee;font-family:sans-serif;max-width:1600px;margin:auto;padding:20px}}
h1{{color:#4ECDC4;text-align:center}}
h2{{color:#4ECDC4;margin-top:30px}}
.sample{{background:#16213e;border-radius:12px;padding:16px;margin:20px auto}}
h3{{color:#FF6B6B;margin:0 0 4px}}
.meta{{color:#4ECDC4;font-size:13px;margin:4px 0}}
.labels{{color:#aaa;font-size:12px;margin:2px 0 8px}}
.model{{margin:8px 0}}
.model-name{{color:#FFD93D;font-size:14px;font-weight:bold;margin:4px 0}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}
th,td{{border:1px solid #333;padding:8px;text-align:center;font-size:13px}}
th{{background:#16213e;color:#4ECDC4}}
td{{background:#0f1a2e}}
</style></head><body>
<h1>PCB Harmonization — Model Comparison</h1>
<p style="text-align:center;color:#aaa">{len(sample_ids)} sampled patches across {len(names)} models</p>

<h2>Metrics Summary</h2>
<table>
<tr><th>Model</th><th>FID↓</th><th>KID↓</th><th>Mean IoU↑</th><th>Class Acc↑</th><th>Precision↑</th><th>Recall↑</th><th>FP/img↓</th><th>Halluc/img↓</th><th>AP@0.5↑</th></tr>
{"".join(metrics_rows)}
</table>

<h2>Visual Comparison</h2>
{"".join(rows)}
</body></html>"""

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Gallery saved: {output_path} ({len(sample_ids)} patches)")


if __name__ == "__main__":
    main()
