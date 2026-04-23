#!/usr/bin/env python3
"""AlignProp sanity check: run first N eval patches through two configs and
save 5-panel comparison images [bbox | composite | v3.4 | v3.4+delta | gt].

Usage:
  python infer_alignprop_sanity.py \
      --v34_ckpt runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000 \
      --delta_ckpt runs/alignprop_prod_4gpu_500step_v2/ckpt/step_300.safetensors \
      --output_dir runs/alignprop_prod_4gpu_500step_v2/sanity_20 \
      --n_patches 20
"""
import json, argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from diffusers import FluxPipeline
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file

import sys
sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from lib.component_bank_v2_1 import ComponentBankV2_1, CAT_ID_TO_NAME as _CAT_ID_TO_NAME, parse_resolution_class

FLUX = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
GEN = 1024

# Reuse the working composite builder + draw from infer_v3_3_full_eval
sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")
from infer_v3_3_full_eval import paste_patch_composite, draw_patch_bboxes, PROMPT_TEMPLATES


def _set_delta_scale(pipe, scale):
    from peft.tuners.lora.layer import BaseTunerLayer
    for m in pipe.transformer.modules():
        if isinstance(m, BaseTunerLayer) and "delta" in getattr(m, "scaling", {}):
            m.scaling["delta"] = scale


def gen_one(pipe, prompt, composite_1024, main_adapter, seed):
    cond = Condition(composite_1024, "pcb_harmonize")
    seed_everything(seed)
    out = generate(
        pipe, prompt=prompt, conditions=[cond],
        main_adapter=main_adapter,
        height=GEN, width=GEN, num_inference_steps=28,
    )
    return out.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v34_ckpt", required=True)
    ap.add_argument("--delta_ckpt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_patches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_json",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json")
    ap.add_argument("--anno_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/annotation/train")
    ap.add_argument("--image_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/image/train")
    ap.add_argument("--board_image_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/image/test")
    ap.add_argument("--anno_test_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/annotation/test")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Load FLUX + v3.4 as adapter
    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16).to("cuda")
    pipe.transformer.requires_grad_(False)
    pipe.load_lora_weights(args.v34_ckpt, weight_name="default.safetensors", adapter_name="pcb_harmonize")

    # Add delta, load weights
    delta_cfg = LoraConfig(r=8, lora_alpha=8,
        target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
        init_lora_weights=True)
    inject_adapter_in_model(delta_cfg, pipe.transformer, adapter_name="delta")
    sd = load_file(args.delta_ckpt)
    clean = {}
    for k, v in sd.items():
        k2 = k
        for prefix in ("base_model.model.", "transformer."):
            if k2.startswith(prefix): k2 = k2[len(prefix):]
        clean[k2] = v
    set_peft_model_state_dict(pipe.transformer, clean, adapter_name="delta")
    pipe.set_adapters(["pcb_harmonize", "delta"])
    print(f"[ready] v3.4={args.v34_ckpt}\n        delta={args.delta_ckpt}")

    # Load eval patches + board meta
    with open(args.eval_json) as f:
        eval_patches = json.load(f)[:args.n_patches]
    test_board_meta = {}
    for anno_path in Path(args.anno_test_dir).glob("*.json"):
        with open(anno_path) as f:
            d = json.load(f)
        test_board_meta[anno_path.stem] = {
            "color": d.get("board_color") or "green",
            "resolution_class": parse_resolution_class(d.get("resolution_class") or "R3"),
        }
    bank = ComponentBankV2_1(anno_dir=args.anno_dir, image_dir=args.image_dir, edge_margin=5)

    board_images = {}
    for idx, patch in enumerate(eval_patches):
        patch_id = patch["patch_id"]
        board = patch["board"]
        board_color = patch["board_color"]
        cx = patch["crop_x"]; cy = patch["crop_y"]
        crop_size = patch.get("crop_size", 512)
        annotations = patch["annotations"]
        prompt = PROMPT_TEMPLATES.get(board_color) or PROMPT_TEMPLATES["green"]
        meta = test_board_meta.get(board, {"color": board_color or "green", "resolution_class": "R3"})
        for ann in annotations:
            if "category_name" not in ann:
                ann["category_name"] = _CAT_ID_TO_NAME.get(ann.get("category_id"), "unknown")

        # Build composite
        comp_512, placed = paste_patch_composite(
            annotations, bank, meta["color"], meta["resolution_class"],
            exclude_board=board, crop_size=crop_size, top_k=10,
        )
        comp_1024 = comp_512.resize((GEN, GEN), Image.LANCZOS)

        # Original patch (if available)
        if board not in board_images:
            p = Path(args.board_image_dir) / f"{board}.png"
            board_images[board] = Image.open(p).convert("RGB") if p.exists() else None
        orig_patch = None
        if board_images[board] is not None:
            raw = board_images[board].crop((cx, cy, cx + crop_size, cy + crop_size))
            orig_patch = raw.resize((GEN, GEN), Image.LANCZOS)

        bbox_img = draw_patch_bboxes(annotations, size=GEN, crop_size=crop_size)

        # Generate v3.4 only (main_adapter=None), then with delta (main_adapter="delta")
        _set_delta_scale(pipe, 0.0)
        v34 = gen_one(pipe, prompt, comp_1024, main_adapter=None, seed=args.seed + idx)
        _set_delta_scale(pipe, 1.0)
        v34d = gen_one(pipe, prompt, comp_1024, main_adapter="delta", seed=args.seed + idx)

        panel = Image.new("RGB", (GEN * 5, GEN))
        panel.paste(bbox_img, (0, 0))
        panel.paste(comp_1024, (GEN, 0))
        panel.paste(v34, (GEN * 2, 0))
        panel.paste(v34d, (GEN * 3, 0))
        if orig_patch: panel.paste(orig_patch, (GEN * 4, 0))
        panel.save(out / f"{patch_id}_5panel.png")
        v34.save(out / f"{patch_id}_v34.png")
        v34d.save(out / f"{patch_id}_v34_delta300.png")
        print(f"  [{idx+1}/{args.n_patches}] {patch_id} placed={placed}/{len(annotations)}")

    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
