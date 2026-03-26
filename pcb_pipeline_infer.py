#!/usr/bin/env python3
"""
PCB Full Pipeline Inference
Layout generation (Qwen 3B) → Component pasting → PCB harmonization (OminiControl)

4-panel output: [Layout bbox] | [Composite] | [Harmonized] | [Original board]
"""
import os
import sys
import json
import re
import csv
import random
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers.pipelines import FluxPipeline

# OminiControl must be on path
OMINI_ROOT = "/home/xinrui/projects/OminiControl"
sys.path.insert(0, OMINI_ROOT)
from omini.pipeline.flux_omini import Condition, generate, seed_everything

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOARD_SIZE = 512
CATEGORIES = {
    "RESISTOR", "CAPACITOR", "INDUCTOR", "CONNECTOR",
    "DIODE", "LED", "SWITCH", "TRANSISTOR", "IC",
}
CAT_NAME_TO_TOKEN = {
    "Resistor": "RESISTOR", "Capacitor": "CAPACITOR", "Inductor": "INDUCTOR",
    "Connector": "CONNECTOR", "Diode": "DIODE", "LED": "LED",
    "Switch": "SWITCH", "Transistor": "TRANSISTOR",
    "Integrated Circuit": "IC", "Integrated_Circuit": "IC",
}
COLORS = {
    "RESISTOR":   (255, 107, 107), "CAPACITOR":  (78,  205, 196),
    "INDUCTOR":   (69,  183, 209), "CONNECTOR":  (150, 206, 180),
    "DIODE":      (255, 234, 167), "LED":        (221, 160, 221),
    "SWITCH":     (240, 230, 140), "TRANSISTOR": (255, 179,  71),
    "IC":         (135, 206, 235),
}
SYSTEM_MSG = (
    "You are a PCB layout generator. Given a description of components, "
    "output their placement as a sequence of tokens representing component "
    "type, x-position, y-position, width, and height on a 512×512 board."
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
PROJECT_DIR   = Path("/home/xinrui/projects/PCB_structure_layout")
DATA_ROOT     = Path("/home/xinrui/projects/data/ti_pcb")
DEFAULT_CKPT  = str(PROJECT_DIR / "qwen3b_lora" / "checkpoint-9000")
DEFAULT_BASE  = "Qwen/Qwen2.5-3B-Instruct"
TEST_JSONL    = str(DATA_ROOT / "layout_data" / "test.jsonl")
METADATA_CSV  = str(DATA_ROOT / "components" / "metadata_train.csv")
TRAIN_IMG_DIR = str(DATA_ROOT / "COCO_label" / "cropped_512" / "images" / "train")
TEST_IMG_DIR  = str(DATA_ROOT / "COCO_label" / "cropped_512" / "images" / "test")
EXCLUDE_FILE  = str(PROJECT_DIR / "excluded_components.json")
RECLASS_FILE  = str(PROJECT_DIR / "reclassified_components.json")
DEFAULT_OMINI = "/home/xinrui/projects/OminiControl/runs/20260305-001915/ckpt/24000"

# ---------------------------------------------------------------------------
# Layout parsing
# ---------------------------------------------------------------------------
def parse_layout(text):
    components = []
    tokens = re.findall(r"\[(\w+)\]", text)
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in CATEGORIES and i + 4 < len(tokens):
            try:
                x, y, w, h = (int(tokens[i+k]) for k in range(1, 5))
                if w > 0 and h > 0:
                    components.append((t, x, y, w, h))
                i += 5
                continue
            except ValueError:
                pass
        i += 1
    return components

# ---------------------------------------------------------------------------
# Component Bank  (on-the-fly crops, edge filtering, exclusion, size+AR match)
# ---------------------------------------------------------------------------
class ComponentBank:
    def __init__(self, metadata_csv, board_img_dir, edge_margin=10,
                 exclude_file=None, reclass_file=None):
        self.board_img_dir = board_img_dir
        self.by_category = defaultdict(list)
        self._img_cache = {}

        excluded_ids = set()
        if exclude_file and os.path.exists(exclude_file):
            excluded_ids = set(json.load(open(exclude_file)))
            print(f"  Exclusion list: {len(excluded_ids)} crops")

        reclass_map = {}
        if reclass_file and os.path.exists(reclass_file):
            reclass_map = json.load(open(reclass_file))
            print(f"  Reclassification map: {len(reclass_map)} crops")

        print(f"Loading component bank from {metadata_csv}...")
        skipped_edge = skipped_excl = reclassed = 0
        with open(metadata_csv) as f:
            for row in csv.DictReader(f):
                cat_name = row["category_name"]
                token_cat = CAT_NAME_TO_TOKEN.get(cat_name)
                if token_cat is None:
                    continue
                x, y, w, h = (float(row[k]) for k in ("bbox_x","bbox_y","bbox_w","bbox_h"))
                if w <= 0 or h <= 0:
                    continue
                # Skip edge crops (likely truncated by patch boundaries)
                if (x < edge_margin or y < edge_margin or
                        x + w > BOARD_SIZE - edge_margin or
                        y + h > BOARD_SIZE - edge_margin):
                    skipped_edge += 1
                    continue
                if row["id"] in excluded_ids:
                    skipped_excl += 1
                    continue
                if row["id"] in reclass_map:
                    token_cat = reclass_map[row["id"]]
                    reclassed += 1
                self.by_category[token_cat].append({
                    "source_image": row["source_image"],
                    "bbox_x": x, "bbox_y": y,
                    "bbox_w": w, "bbox_h": h,
                    "ar": w / h,
                    "area": w * h,
                })

        print(f"  Skipped {skipped_edge} edge | {skipped_excl} excluded | {reclassed} reclassified")
        for cat, entries in sorted(self.by_category.items()):
            print(f"  {cat}: {len(entries)}")

    def _get_board_image(self, source_image):
        if source_image not in self._img_cache:
            path = os.path.join(self.board_img_dir, f"{source_image}.png")
            if not os.path.exists(path):
                return None
            self._img_cache[source_image] = Image.open(path).convert("RGB")
            if len(self._img_cache) > 200:
                del self._img_cache[next(iter(self._img_cache))]
        return self._img_cache.get(source_image)

    def find_match(self, cat, pred_w, pred_h, top_k=10, size_thresh=0.95):
        candidates = self.by_category.get(cat, [])
        if not candidates:
            return None
        pred_area = pred_w * pred_h
        pred_ar   = pred_w / pred_h if pred_h > 0 else 1.0
        pred_horiz = pred_w >= pred_h

        # Step 1: size filter
        filtered = [e for e in candidates
                    if min(pred_area, e["area"]) / max(pred_area, e["area"]) >= size_thresh]
        if len(filtered) < top_k:
            filtered = sorted(candidates, key=lambda e: abs(pred_area - e["area"]))[:max(top_k*5, 50)]

        # Step 2: prefer same orientation, rank by AR similarity
        same = [e for e in filtered if (e["bbox_w"] >= e["bbox_h"]) == pred_horiz]
        pool = same if len(same) >= top_k else filtered
        pool.sort(key=lambda e: abs(pred_ar - e["ar"]))
        return random.choice(pool[:top_k])

    def load_crop(self, entry, target_w, target_h):
        board = self._get_board_image(entry["source_image"])
        if board is None:
            return None
        x1, y1 = entry["bbox_x"], entry["bbox_y"]
        crop = board.crop((x1, y1, x1 + entry["bbox_w"], y1 + entry["bbox_h"]))
        return crop.resize((max(target_w, 1), max(target_h, 1)), Image.LANCZOS)

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_bboxes(components):
    img  = Image.new("RGB", (BOARD_SIZE, BOARD_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for cat, x, y, w, h in components:
        color = COLORS.get(cat, (128, 128, 128))
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
        draw.text((x+2, y+2), cat[:3], fill=color)
    return img

def paste_components(components, bank, top_k=10):
    img = Image.new("RGB", (BOARD_SIZE, BOARD_SIZE), (255, 255, 255))
    for cat, x, y, w, h in components:
        match = bank.find_match(cat, w, h, top_k=top_k)
        if match is None:
            continue
        crop = bank.load_crop(match, w, h)
        if crop is None:
            continue
        px = max(0, min(x, BOARD_SIZE - w))
        py = max(0, min(y, BOARD_SIZE - h))
        img.paste(crop, (px, py))
    return img

# ---------------------------------------------------------------------------
# OminiControl harmonization
# ---------------------------------------------------------------------------
def load_omini_model(checkpoint_path, device):
    print(f"[OminiControl] Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to(device)
    print(f"[OminiControl] Loading LoRA: {checkpoint_path}")
    pipe.load_lora_weights(
        checkpoint_path, weight_name="default.safetensors", adapter_name="pcb_harmonize"
    )
    pipe.set_adapters(["pcb_harmonize"])
    print("[OminiControl] Ready.")
    return pipe

def harmonize(pipe, composite_img, prompt, seed):
    condition = Condition(composite_img, "pcb_harmonize")
    seed_everything(seed)
    return generate(pipe, prompt=prompt, conditions=[condition]).images[0]

# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
def build_gallery(output_dir, results):
    rows = []
    for r in results:
        name, prompt = r["name"], r["prompt"]
        rows.append(f"""
        <div class="sample">
          <h3>{name}</h3>
          <p class="prompt">{prompt}</p>
          <p class="stats">GT: {len(r['gt_comps'])} | Pred: {len(r['pred_comps'])}</p>
          <img src="{name}_comparison.png" style="width:100%;border-radius:8px;margin-top:8px">
        </div>""")
    legend = "".join(
        f'<span><span class="swatch" style="background:rgb{c}"></span>{cat}</span>'
        for cat, c in COLORS.items()
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PCB Pipeline Gallery</title>
<style>
body{{background:#1a1a2e;color:#eee;font-family:sans-serif;max-width:1600px;margin:auto;padding:20px}}
h1{{color:#4ECDC4;text-align:center}}
.legend{{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin:16px}}
.swatch{{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:4px}}
.sample{{background:#16213e;border-radius:12px;padding:16px;margin:16px auto}}
h3{{color:#FF6B6B;margin:0 0 4px}}
.prompt{{color:#888;font-size:13px;margin:4px 0}}
.stats{{color:#555;font-size:12px}}
</style></head><body>
<h1>PCB Full Pipeline — Qwen 3B Layout → Paste → OminiControl Harmonize</h1>
<p style="text-align:center;color:#aaa;font-size:13px">
  Panel order: bbox viz &nbsp;|&nbsp; composite (crops pasted) &nbsp;|&nbsp; harmonized &nbsp;|&nbsp; original board
</p>
<div class="legend">{legend}</div>
{"".join(rows)}
</body></html>"""
    path = Path(output_dir) / "gallery.html"
    path.write_text(html)
    print(f"[Gallery] {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PCB Full Pipeline Inference")
    parser.add_argument("--layout_ckpt",  default=DEFAULT_CKPT,
                        help="Qwen 3B LoRA checkpoint dir")
    parser.add_argument("--layout_base",  default=DEFAULT_BASE)
    parser.add_argument("--omini_ckpt",   default=DEFAULT_OMINI,
                        help="OminiControl LoRA checkpoint dir (contains default.safetensors)")
    parser.add_argument("--test_jsonl",   default=TEST_JSONL)
    parser.add_argument("--metadata_csv", default=METADATA_CSV)
    parser.add_argument("--train_img_dir",default=TRAIN_IMG_DIR,
                        help="Source board images for component bank (train)")
    parser.add_argument("--test_img_dir", default=TEST_IMG_DIR,
                        help="Original board images for reference panel (test)")
    parser.add_argument("--exclude_file", default=EXCLUDE_FILE)
    parser.add_argument("--reclass_file", default=RECLASS_FILE)
    parser.add_argument("--output_dir",   default="./pipeline_output")
    parser.add_argument("--num_samples",  type=int, default=10)
    parser.add_argument("--top_k",        type=int, default=10)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--layout_gpu",   type=int, default=0,
                        help="GPU index (after CUDA_VISIBLE_DEVICES filtering)")
    parser.add_argument("--omini_gpu",    type=int, default=0,
                        help="GPU index (after CUDA_VISIBLE_DEVICES filtering)")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build component bank (CPU only, no GPU needed) ──────────────────────
    bank = ComponentBank(
        args.metadata_csv, args.train_img_dir,
        exclude_file=args.exclude_file, reclass_file=args.reclass_file,
    )

    # ── Load test data ───────────────────────────────────────────────────────
    with open(args.test_jsonl) as f:
        test_data = [json.loads(l) for l in f]
    if args.num_samples == -1:
        samples = test_data          # all samples, no shuffle
    else:
        random.shuffle(test_data)
        samples = test_data[:args.num_samples]

    print(f"\n{'='*70}")
    print(f"Running full pipeline on {len(samples)} samples  →  {output_dir}")
    print(f"{'='*70}\n")

    # ── Stage 1 (all samples): Layout generation ─────────────────────────────
    # Load layout model, run all samples, then free GPU memory before loading FLUX
    print(f"[Layout] Loading {args.layout_base} + {args.layout_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.layout_base, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.layout_base, torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.layout_gpu}", trust_remote_code=True,
    )
    layout_model = PeftModel.from_pretrained(base_model, args.layout_ckpt)
    layout_model.eval()
    print("[Layout] Ready. Generating layouts...\n")

    layout_results = []
    for i, sample in enumerate(samples):
        msgs     = sample["messages"]
        prompt   = msgs[1]["content"]
        gt_text  = msgs[2]["content"]
        img_name = sample.get("_meta", {}).get("image", f"sample_{i:04d}")
        chat = [{"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": prompt}]
        text   = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(layout_model.device)
        with torch.no_grad():
            out = layout_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        pred_text  = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gt_comps   = parse_layout(gt_text)
        pred_comps = parse_layout(pred_text)
        print(f"  [{i+1}/{len(samples)}] {img_name}: GT={len(gt_comps)} Pred={len(pred_comps)}")
        layout_results.append((img_name, prompt, gt_comps, pred_comps, i))

    # Free layout model GPU memory before loading FLUX
    print("\n[Layout] Done. Freeing GPU memory...")
    del layout_model, base_model
    torch.cuda.empty_cache()

    # ── Load OminiControl ────────────────────────────────────────────────────
    omini_pipe = load_omini_model(args.omini_ckpt, device=f"cuda:{args.omini_gpu}")

    results = []
    for (img_name, prompt, gt_comps, pred_comps, i) in layout_results:
        print(f"\n[{i+1}/{len(samples)}] {img_name}  |  {prompt[:80]}")

        if not pred_comps:
            print("  ⚠ No valid components, skipping.")
            continue

        # Stage 2: component pasting
        composite = paste_components(pred_comps, bank, top_k=args.top_k)

        # Stage 3: OminiControl harmonization
        harmonized = harmonize(omini_pipe, composite, prompt, seed=args.seed + i)

        # Build 4-panel comparison
        bbox_img = draw_bboxes(pred_comps)
        orig_path = Path(args.test_img_dir) / f"{img_name}.png"
        orig_img  = Image.open(orig_path).convert("RGB") if orig_path.exists() \
                    else Image.new("RGB", (BOARD_SIZE, BOARD_SIZE), (30, 30, 30))

        panel = Image.new("RGB", (BOARD_SIZE * 4, BOARD_SIZE))
        panel.paste(bbox_img,    (0,            0))
        panel.paste(composite,   (BOARD_SIZE,   0))
        panel.paste(harmonized,  (BOARD_SIZE*2, 0))
        panel.paste(orig_img,    (BOARD_SIZE*3, 0))
        panel.save(output_dir / f"{img_name}_comparison.png")
        print(f"  ✓ {img_name}_comparison.png")

        results.append({"name": img_name, "prompt": prompt,
                         "gt_comps": gt_comps, "pred_comps": pred_comps})

    build_gallery(output_dir, results)
    print(f"\n✅ Done! {len(results)} samples in: {output_dir}")
    print(f"   Gallery: {output_dir}/gallery.html")


if __name__ == "__main__":
    main()
