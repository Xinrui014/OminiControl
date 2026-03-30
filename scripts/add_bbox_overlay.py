#!/usr/bin/env python3
"""
Post-process: draw bounding boxes on harmonized results and rebuild gallery.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw

BBOX_COLORS = {
    "RESISTOR": (255, 107, 107), "CAPACITOR": (78, 205, 196),
    "INDUCTOR": (69, 183, 209), "CONNECTOR": (150, 206, 180),
    "DIODE": (255, 234, 167), "LED": (221, 160, 221),
    "SWITCH": (240, 230, 140), "TRANSISTOR": (255, 179, 71),
    "IC": (135, 206, 235), "OSCILLATOR": (255, 160, 122),
    "FUSE": (200, 200, 200),
}
SMALL_THRESH = 30


def draw_bboxes_on_image(img, annotations, alpha=0.5):
    """Draw semi-transparent bboxes on a copy of img."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    for ann in annotations:
        cat = ann["category_name"]
        x, y, w, h = ann["bbox"]
        is_small = max(w, h) < SMALL_THRESH
        color = (255, 255, 0) if is_small else BBOX_COLORS.get(cat, (128, 128, 128))
        width = 2
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    return overlay


def main():
    eval_json = Path("config/eval_patches_small_components.json")
    output_dir = Path("output_1024/v2.1_12k")

    with open(eval_json) as f:
        eval_patches = json.load(f)

    results = []
    for patch in eval_patches:
        pid = patch["patch_id"]
        annotations = patch["annotations"]

        # Load existing harmonized images
        img_512 = output_dir / f"{pid}_512.png"
        img_1024 = output_dir / f"{pid}_1024to512.png"

        if not img_512.exists() or not img_1024.exists():
            print(f"  SKIP {pid}: missing outputs")
            continue

        h512 = Image.open(img_512).convert("RGB")
        h1024 = Image.open(img_1024).convert("RGB")

        # Draw bboxes
        h512_bbox = draw_bboxes_on_image(h512, annotations)
        h1024_bbox = draw_bboxes_on_image(h1024, annotations)

        h512_bbox.save(output_dir / f"{pid}_512_bbox.png")
        h1024_bbox.save(output_dir / f"{pid}_1024to512_bbox.png")

        results.append(patch)
        print(f"  {pid}: bbox overlay saved")

    # Rebuild gallery with bbox overlay row
    rows = []
    for p in results:
        pid = p["patch_id"]
        n_small = sum(1 for a in p["annotations"] if max(a["bbox"][2], a["bbox"][3]) < SMALL_THRESH)
        rows.append(f"""
        <div class="sample">
          <h3>{pid}</h3>
          <p class="meta">Components: {p['n_components']} | Small (&lt;30px): {n_small}</p>
          <p class="labels">bbox | composite | 512 standard | 1024→512 | original</p>
          <img src="{pid}_comparison.png" style="width:100%;border-radius:8px;margin-top:4px">
          <p class="labels" style="margin-top:12px">With component bboxes (yellow = small &lt;30px):</p>
          <div style="display:flex;gap:8px;margin-top:4px">
            <div style="flex:1;text-align:center">
              <div class="labels">512 standard</div>
              <img src="{pid}_512_bbox.png" style="width:100%;border-radius:6px">
            </div>
            <div style="flex:1;text-align:center">
              <div class="labels">1024→512</div>
              <img src="{pid}_1024to512_bbox.png" style="width:100%;border-radius:6px">
            </div>
          </div>
          <p class="labels" style="margin-top:8px">Full 1024x1024:</p>
          <img src="{pid}_1024_full.png" style="width:50%;border-radius:8px">
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PCB 1024 Eval + BBox</title>
<style>
body{{background:#1a1a2e;color:#eee;font-family:sans-serif;max-width:1400px;margin:auto;padding:20px}}
h1{{color:#4ECDC4;text-align:center}}
.sample{{background:#16213e;border-radius:12px;padding:16px;margin:16px auto}}
h3{{color:#FF6B6B;margin:0 0 4px}}
.meta{{color:#4ECDC4;font-size:13px;margin:4px 0}}
.labels{{color:#aaa;font-size:12px;margin:2px 0}}
</style></head><body>
<h1>PCB 1024 Harmonization — v2.1 12k + BBox Overlay</h1>
<p style="text-align:center;color:#aaa">Upscale 512→1024, harmonize, downscale back | Yellow boxes = small components (&lt;30px)</p>
{"".join(rows)}
</body></html>"""
    (output_dir / "gallery.html").write_text(html)
    print(f"\nDone! Updated gallery with bbox overlays → {output_dir}/gallery.html")


if __name__ == "__main__":
    main()
