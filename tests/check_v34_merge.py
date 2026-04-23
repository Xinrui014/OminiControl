"""Verify v3.4 LoRA is effective after our merge+add-delta procedure.

Three generations with SAME noise:
  A: vanilla FLUX (no LoRA at all)          → baseline
  B: FLUX + v3.4 adapter (standard)         → ground-truth reference
  C: FLUX + v3.4 merged + delta (scale=0)   → our approach

If ||C-A|| << ||C-B|| → v3.4 was LOST (merge undone by PEFT)
If ||C-B|| << ||C-A|| → v3.4 is EFFECTIVE
"""
import os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from omini.pipeline.flux_omini import Condition, generate

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
V34_CKPT  = "/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000/default.safetensors"
DTYPE = torch.bfloat16
DEVICE = "cuda"
RES = 512
SEED = 42


def make_composite():
    img = Image.new("RGB", (RES, RES), (40, 80, 40))
    arr = np.array(img)
    specs = [
        (50, 50, 90, 30,   (20, 20, 20)),
        (175, 100, 60, 30, (180, 160, 40)),
        (300, 75, 70, 70,  (30, 30, 40)),
        (100, 250, 30, 20, (40, 40, 40)),
        (250, 300, 150, 40,(120, 90, 60)),
        (50, 400, 50, 50,  (180, 180, 180)),
    ]
    for x, y, w, h, c in specs:
        arr[y:y+h, x:x+w] = c
    return Image.fromarray(arr)


def gen_with_pipe(pipe, adapter_name_or_none, prompt, composite, seed):
    """One generation at RES×RES. adapter_name_or_none may be None for vanilla."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    cond = Condition(composite, adapter_name_or_none)
    out = generate(
        pipe, prompt=prompt,
        conditions=[cond] if adapter_name_or_none is not None else [],
        main_adapter=adapter_name_or_none,
        height=RES, width=RES,
        num_inference_steps=10,
        generator=g,
        output_type="pt",
    )
    img = out.images[0] if hasattr(out, "images") else out[0][0]
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.float().cpu()  # (1,3,H,W) in [0,1]


def main():
    composite = make_composite()
    prompt = "a pcb board"

    # ---- run A: vanilla ----
    print("[A] vanilla FLUX (no LoRA) ...")
    pipe_a = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe_a.transformer.eval()
    img_A = gen_with_pipe(pipe_a, None, prompt, composite, SEED)
    print(f"  A stats: min={img_A.min():.3f} max={img_A.max():.3f} mean={img_A.mean():.3f}")
    del pipe_a
    torch.cuda.empty_cache()

    # ---- run B: v3.4 as adapter (standard OminiControl path) ----
    print("\n[B] FLUX + v3.4 as adapter (standard) ...")
    pipe_b = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe_b.transformer.eval()
    pipe_b.load_lora_weights(os.path.dirname(V34_CKPT),
                              weight_name=os.path.basename(V34_CKPT),
                              adapter_name="pcb_harmonize")
    pipe_b.set_adapters(["pcb_harmonize"])
    img_B = gen_with_pipe(pipe_b, "pcb_harmonize", prompt, composite, SEED)
    print(f"  B stats: min={img_B.min():.3f} max={img_B.max():.3f} mean={img_B.mean():.3f}")
    del pipe_b
    torch.cuda.empty_cache()

    # ---- run C: v3.4 merged + delta (our training pipeline) ----
    print("\n[C] FLUX + v3.4 MERGED + delta@scale 0 (our AlignProp path) ...")
    pipe_c = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe_c.transformer.eval()

    # Add v3.4 adapter
    v34_cfg = LoraConfig(r=16, lora_alpha=16,
        target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
        init_lora_weights=True)
    pipe_c.transformer.add_adapter(v34_cfg, adapter_name="pcb_harmonize")
    clean = {k.replace("transformer.","",1) if k.startswith("transformer.") else k: v
             for k, v in load_file(V34_CKPT).items()}
    set_peft_model_state_dict(pipe_c.transformer, clean, adapter_name="pcb_harmonize")

    # Merge
    pipe_c.transformer.set_adapter("pcb_harmonize")
    from peft.tuners.tuners_utils import BaseTunerLayer
    for m in pipe_c.transformer.modules():
        if isinstance(m, BaseTunerLayer) and "pcb_harmonize" in m.active_adapters:
            m.merge(adapter_names=["pcb_harmonize"])
    pipe_c.transformer.delete_adapters(["pcb_harmonize"])

    # Add delta (this triggered the "unmerge" warning in Phase 3)
    delta_cfg = LoraConfig(r=8, lora_alpha=8,
        target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
        init_lora_weights="gaussian")
    pipe_c.transformer.add_adapter(delta_cfg, adapter_name="delta")
    # Set delta scale to 0 so it doesn't affect output
    from peft.tuners.lora.layer import BaseTunerLayer as BTL
    for m in pipe_c.transformer.modules():
        if isinstance(m, BTL) and "delta" in getattr(m, "scaling", {}):
            m.scaling["delta"] = 0.0

    img_C = gen_with_pipe(pipe_c, "delta", prompt, composite, SEED)
    print(f"  C stats: min={img_C.min():.3f} max={img_C.max():.3f} mean={img_C.mean():.3f}")

    # ---- compare ----
    diff_AB = (img_A - img_B).abs().mean().item()
    diff_AC = (img_A - img_C).abs().mean().item()
    diff_BC = (img_B - img_C).abs().mean().item()

    print("\n" + "="*60)
    print(f"Pixel L1 differences:")
    print(f"  ||A (vanilla) - B (v3.4 adapter)||  = {diff_AB:.4f}  (how much v3.4 changes output)")
    print(f"  ||A (vanilla) - C (our pipeline)||  = {diff_AC:.4f}")
    print(f"  ||B (v3.4 adap) - C (our pipeline)||= {diff_BC:.4f}  (difference btwn ref and ours)")
    print()
    ratio = diff_BC / max(diff_AB, 1e-8)
    print(f"  diff_BC / diff_AB = {ratio:.3f}")
    print()
    if diff_BC < 0.3 * diff_AB:
        print("  [PASS] C matches B -> v3.4 IS effective in merged pipeline")
    elif diff_AC < 0.3 * diff_AB:
        print("  [FAIL] C matches A -> v3.4 was LOST during merge")
    else:
        print("  [AMBIGUOUS] C is between A and B; partial effect")

    # Save images for visual inspection
    from torchvision.utils import save_image
    out_dir = "/projects/_ssd/xrssd/logs/v34_merge_check"
    os.makedirs(out_dir, exist_ok=True)
    save_image(img_A, f"{out_dir}/A_vanilla.png")
    save_image(img_B, f"{out_dir}/B_v34_adapter.png")
    save_image(img_C, f"{out_dir}/C_merged_with_delta.png")
    print(f"\n  Images saved to {out_dir}")


if __name__ == "__main__":
    main()
