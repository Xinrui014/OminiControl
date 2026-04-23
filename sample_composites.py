"""Quick script to visualize ComponentBankV2.1 on-the-fly pasting results."""
import json
import os
import random
from pathlib import Path
from PIL import Image

from lib.component_bank_v2_1 import (
    ComponentBankV2_1, get_annotations_in_crop, CAT_ID_TO_NAME, parse_resolution_class,
)

anno_dir = '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/annotation/train'
image_dir = '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/image/train'
output_dir = '/projects/_ssd/xrssd/OminiControl/runs/v4_subclass_1024/sample_composites'
os.makedirs(output_dir, exist_ok=True)

bank = ComponentBankV2_1(anno_dir=anno_dir, image_dir=image_dir)

# Pick 10 random boards
anno_files = sorted(Path(anno_dir).glob("*.json"))
random.seed(42)
selected = random.sample(anno_files, min(10, len(anno_files)))

crop_size = 512
results = []

for anno_path in selected:
    board_name = anno_path.stem
    img_path = os.path.join(image_dir, f"{board_name}.png")
    if not os.path.exists(img_path):
        continue

    board_img = Image.open(img_path)
    if board_img.mode in ("RGBA", "PA", "P"):
        bg = Image.new("RGB", board_img.size, (255, 255, 255))
        bg.paste(board_img, mask=board_img.convert("RGBA").split()[3])
        board_img = bg
    else:
        board_img = board_img.convert("RGB")

    with open(anno_path) as f:
        data = json.load(f)

    board_color = data.get("board_color", "green")
    resolution_class = parse_resolution_class(data.get("resolution_class", "R3"))
    all_anns = data.get("annotations", [])
    for ann in all_anns:
        if "category_name" not in ann:
            ann["category_name"] = CAT_ID_TO_NAME.get(ann.get("category_id"), "unknown")

    img_w, img_h = board_img.size

    # Try to find a good crop
    for attempt in range(10):
        cx = random.randint(0, max(0, img_w - crop_size))
        cy = random.randint(0, max(0, img_h - crop_size))
        crop_anns = get_annotations_in_crop(all_anns, cx, cy, crop_size, 0.5)
        if len(crop_anns) >= 3:
            break

    if len(crop_anns) < 2:
        continue

    # Real patch
    real_patch = board_img.crop((cx, cy, cx + crop_size, cy + crop_size))

    # Build composite
    canvas = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
    stats = {"matched": 0, "rotated": 0, "self": 0}

    for ann in crop_anns:
        cat_name = ann["category_name"]
        rx, ry, rw, rh = ann["bbox"]
        rw_int, rh_int = int(rw), int(rh)
        if rw_int < 3 or rh_int < 3:
            continue

        orientation = ann.get("orientation", 0)
        sub_class = ann.get("sub_class", -1)

        result = bank.find_match(
            category=cat_name, sub_class=sub_class,
            target_w=rw, target_h=rh,
            board_color=board_color, resolution_class=resolution_class,
            orientation=orientation, exclude_board=board_name,
        )

        if result is not None:
            entry, rotation = result
            crop = bank.load_crop(entry, rw_int, rh_int, rotation=rotation, resize_jitter=0.15)
            if rotation != 0:
                stats["rotated"] += 1
            else:
                stats["matched"] += 1
        else:
            # Self-fallback
            orig_bbox = ann.get("original_bbox")
            if orig_bbox:
                crop = bank.load_self_crop(board_name, orig_bbox, rw_int, rh_int)
            else:
                abs_x, abs_y = cx + int(rx), cy + int(ry)
                crop = board_img.crop((abs_x, abs_y, abs_x + rw_int, abs_y + rh_int))
                crop = crop.resize((rw_int, rh_int), Image.LANCZOS)
            stats["self"] += 1

        if crop is None:
            continue
        px = max(0, min(int(rx), crop_size - crop.width))
        py = max(0, min(int(ry), crop_size - crop.height))
        canvas.paste(crop, (px, py))

    # Build mask (white background, gray where components are)
    mask = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    colors = {
        "RESISTOR": (255, 100, 100), "CAPACITOR": (100, 100, 255),
        "IC": (100, 255, 100), "CONNECTOR": (255, 200, 100),
        "DIODE": (200, 100, 255), "INDUCTOR": (255, 255, 100),
        "SWITCH": (100, 255, 255), "TRANSISTOR": (255, 150, 150),
        "OSCILLATOR": (150, 255, 150), "LED": (255, 200, 200),
        "FUSE": (200, 200, 200),
    }
    for ann in crop_anns:
        rx, ry, rw, rh = ann["bbox"]
        c = colors.get(ann["category_name"], (180, 180, 180))
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=c, outline=(0, 0, 0))

    # Save side by side: real | mask | composite
    W = crop_size
    combined = Image.new("RGB", (W * 3, W))
    combined.paste(real_patch, (0, 0))
    combined.paste(mask, (W, 0))
    combined.paste(canvas, (W * 2, 0))
    fname = f"{board_name}.jpg"
    combined.save(os.path.join(output_dir, fname), quality=90)
    results.append((fname, board_name, board_color, len(crop_anns), stats))
    print(f"{board_name}: {len(crop_anns)} comps, matched={stats['matched']}, rotated={stats['rotated']}, self={stats['self']}")

# Generate HTML gallery
html = """<html><head><title>ComponentBankV2.1 Samples</title>
<style>body{font-family:monospace;background:#111;color:#eee}
img{max-width:100%} .row{margin:10px 0;padding:10px;background:#222;border-radius:8px}
.labels{display:flex;justify-content:space-around;font-size:14px;margin-bottom:5px}
</style></head><body><h2>ComponentBankV2.1 On-the-fly Pasting (v4_subclass)</h2>
<p>Left: Real patch | Center: Component mask | Right: Composite (pasted from other boards)</p>
"""
for fname, bname, bcolor, ncomps, stats in results:
    html += f"""<div class="row">
<div class="labels"><span>Real</span><span>Mask ({ncomps} comps)</span><span>Composite (match={stats['matched']}, rot={stats['rotated']}, self={stats['self']})</span></div>
<img src="{fname}">
<p>{bname} — color={bcolor}</p></div>\n"""
html += "</body></html>"

with open(os.path.join(output_dir, "gallery.html"), "w") as f:
    f.write(html)
print(f"\nGallery saved to {output_dir}/gallery.html")
print(f"Total: {len(results)} samples")
