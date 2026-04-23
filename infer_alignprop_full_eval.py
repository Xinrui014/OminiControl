#!/usr/bin/env python3
"""Full-eval inference for AlignProp delta LoRA stacked on v3.4.

Loads FLUX + fuses v3.4 LoRA into base + adds AlignProp delta LoRA on top,
then runs the same 2186-patch eval pipeline as infer_v3_3_full_eval.py.

Differences vs infer_v3_3_full_eval.py:
  - Replaces single load_lora_weights call with v3.4 fuse + delta adapter
  - Other args: --v34_ckpt (v3.4 LoRA dir), --delta_ckpt (delta .safetensors file)

Usage (same 4-GPU pattern as v3.4 eval):
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python infer_alignprop_full_eval.py \\
      --eval_json "$EVAL_JSON" \\
      --v34_ckpt "$V34_DIR" \\
      --delta_ckpt "$DELTA_FILE" \\
      --output_dir "$OUT_DIR" \\
      --start $((i*547)) --end $((i*547+547)) &
  done; wait
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import FluxPipeline
from peft import LoraConfig

# Reuse helpers from existing script
sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from lib.component_bank_v2_1 import ComponentBankV2_1, CAT_ID_TO_NAME, parse_resolution_class

# Also pull the original helpers from infer_v3_3_full_eval
from infer_v3_3_full_eval import (
    PROMPT_TEMPLATES, BBOX_COLORS, paste_patch_composite, draw_patch_bboxes,
    FLUX_PATH, GEN_SIZE,
)

CAT_NAMES_DINO = ["Resistor","Capacitor","Inductor","Connector",
                  "Diode","Switch","Transistor","IC","Oscillator"]


def load_model_alignprop(v34_dir: str, delta_ckpt: str, device: str = "cuda") -> FluxPipeline:
    """FLUX + v3.4 LoRA (fused into base) + delta LoRA (active adapter)."""
    print(f"[load] FLUX base {FLUX_PATH}")
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=torch.bfloat16).to(device)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    pipe.transformer.eval()   # inference-only; no gradient checkpointing

    # 1. Fuse v3.4 into base
    print(f"[load] v3.4 LoRA from {v34_dir}")
    pipe.load_lora_weights(v34_dir, weight_name="default.safetensors",
                            adapter_name="pcb_harmonize")
    pipe.set_adapters(["pcb_harmonize"])
    print("[fuse] v3.4 -> base")
    pipe.fuse_lora(adapter_names=["pcb_harmonize"], lora_scale=1.0)
    pipe.unload_lora_weights()

    # 2. Add delta LoRA from AlignProp
    print(f"[load] delta LoRA from {delta_ckpt}")
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict

    delta_cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
        init_lora_weights=True,
    )
    pipe.transformer.add_adapter(delta_cfg, adapter_name="delta")
    sd = load_file(delta_ckpt)
    # Strip 'base_model.model.' or similar prefixes if present
    clean = {}
    for k, v in sd.items():
        k2 = k
        for prefix in ("base_model.model.", "transformer."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        clean[k2] = v
    missing, unexpected = set_peft_model_state_dict(
        pipe.transformer, clean, adapter_name="delta"
    ), None  # PEFT returns incompat keys info; ignore for now
    print(f"  loaded {len(clean)} delta tensors")

    # Activate delta adapter for inference
    from peft.tuners.lora.layer import BaseTunerLayer
    for m in pipe.transformer.modules():
        if isinstance(m, BaseTunerLayer) and "delta" in getattr(m, "scaling", {}):
            m.scaling["delta"] = 1.0

    print("Model ready (FLUX + v3.4-fused + delta LoRA).")
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--v34_ckpt", required=True,
                       help="Directory with v3.4 default.safetensors (e.g. runs/.../ckpt/6000)")
    parser.add_argument("--delta_ckpt", required=True,
                       help="Path to AlignProp delta .safetensors file")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2186)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipe = load_model_alignprop(args.v34_ckpt, args.delta_ckpt)

    # Reuse the existing eval loop from infer_v3_3_full_eval by calling its main with
    # patched load_model. Cleanest: import main and pass pipe.
    # But infer_v3_3_full_eval.main() reloads model. Simpler: inline the main body here.

    from infer_v3_3_full_eval import main as v33_main
    # Monkey-patch load_model to return our pre-loaded pipe
    import infer_v3_3_full_eval as v33
    v33.load_model = lambda checkpoint_path, device="cuda": pipe
    # Invoke v33.main with our args (pipe ignores checkpoint_path)
    sys.argv = [
        sys.argv[0],
        "--eval_json", args.eval_json,
        "--omini_ckpt", args.v34_ckpt,    # unused in monkey-patched load_model
        "--output_dir", args.output_dir,
        "--start", str(args.start),
        "--end", str(args.end),
        "--seed", str(args.seed),
    ]
    v33_main()


if __name__ == "__main__":
    main()
