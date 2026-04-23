#!/usr/bin/env python3
"""Run v3.4+delta inference using the EXACT composites saved in the v3.4 ckpt14k
full eval (read-only). Extracts composite from the 4-panel comparison PNG so
there's no composite-building noise between v3.4 and AlignProp runs.

Input layout per patch (in --composites_dir, READ-ONLY):
  {patch_id}_comparison.png   4096×1024   [bbox | composite | harmonized | gt]
  → composite column = crop(1024, 0, 2048, 1024)

Output layout (in --output_dir):
  {patch_id}_harmonized.png   1024×1024   v3.4 + delta@1 output

Launch via run_infer_alignprop_4gpu.sh; each GPU handles a shard of patches.
"""
import argparse, json, sys
from pathlib import Path
import torch
from PIL import Image
from diffusers import FluxPipeline
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from infer_v3_3_full_eval import PROMPT_TEMPLATES  # type: ignore

FLUX = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
GEN = 1024


def _set_delta_scale(pipe, scale):
    from peft.tuners.lora.layer import BaseTunerLayer
    for m in pipe.transformer.modules():
        if isinstance(m, BaseTunerLayer) and "delta" in getattr(m, "scaling", {}):
            m.scaling["delta"] = scale


def extract_composite(comparison_path: Path) -> Image.Image:
    im = Image.open(comparison_path).convert("RGB")
    # 4-panel 4096×1024: bbox|composite|harmonized|gt — composite is panel #2
    return im.crop((GEN, 0, GEN * 2, GEN))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v34_ckpt", required=True)
    ap.add_argument("--delta_ckpt", required=True)
    ap.add_argument("--composites_dir", required=True,
                    help="folder containing {patch_id}_comparison.png (READ-ONLY)")
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--delta_scale", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    comp_dir = Path(args.composites_dir)

    with open(args.eval_json) as f:
        eval_patches = json.load(f)
    total = len(eval_patches)
    if args.end > 0:
        eval_patches = eval_patches[args.start:args.end]
    elif args.start > 0:
        eval_patches = eval_patches[args.start:]
    print(f"[shard] {len(eval_patches)}/{total} patches [{args.start}:{args.end}]")

    pipe = FluxPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16).to("cuda")
    pipe.transformer.requires_grad_(False)
    pipe.load_lora_weights(args.v34_ckpt, weight_name="default.safetensors", adapter_name="pcb_harmonize")

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
    _set_delta_scale(pipe, args.delta_scale)
    print(f"[ready] v3.4={args.v34_ckpt}\n        delta={args.delta_ckpt} scale={args.delta_scale}")

    ok, skip_exist, skip_missing = 0, 0, 0
    for idx, patch in enumerate(eval_patches):
        patch_id = patch["patch_id"]
        out_path = out / f"{patch_id}_harmonized.png"
        if out_path.exists():
            skip_exist += 1
            continue
        comp_file = comp_dir / f"{patch_id}_comparison.png"
        if not comp_file.exists():
            skip_missing += 1
            print(f"  [skip] {patch_id}: no comparison.png")
            continue
        composite_1024 = extract_composite(comp_file)
        prompt = PROMPT_TEMPLATES.get(patch.get("board_color")) or PROMPT_TEMPLATES["green"]

        abs_idx = args.start + idx
        cond = Condition(composite_1024, "pcb_harmonize")
        seed_everything(args.seed + abs_idx)
        img = generate(
            pipe, prompt=prompt, conditions=[cond],
            main_adapter="delta",
            height=GEN, width=GEN, num_inference_steps=28,
        ).images[0]
        img.save(out_path)
        ok += 1
        if ok % 10 == 0 or idx == len(eval_patches) - 1:
            print(f"  [{idx+1}/{len(eval_patches)}] ok={ok} skip_exist={skip_exist} skip_missing={skip_missing}")

    print(f"\n[done] shard ok={ok} skip_exist={skip_exist} skip_missing={skip_missing} → {out}")


if __name__ == "__main__":
    main()
