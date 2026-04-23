"""v3.4 merge check v2: use diffusers load_lora + fuse_lora (auto-detects modules)."""
import os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig
from omini.pipeline.flux_omini import Condition, generate

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
V34_DIR   = "/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000"
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


def gen(pipe, adapter, composite, seed=SEED):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    cond = Condition(composite, adapter)
    out = generate(
        pipe, prompt="a pcb board",
        conditions=[cond] if adapter is not None else [],
        main_adapter=adapter,
        height=RES, width=RES, num_inference_steps=10,
        generator=g, output_type="pt",
    )
    img = out.images[0] if hasattr(out, "images") else out[0][0]
    if img.dim() == 3: img = img.unsqueeze(0)
    return img.float().cpu()


def main():
    comp = make_composite()

    # B (reference): v3.4 adapter + delta unused
    print("[B] v3.4 as adapter (reference)...")
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.transformer.eval()
    pipe.load_lora_weights(V34_DIR, weight_name="default.safetensors", adapter_name="pcb_harmonize")
    pipe.set_adapters(["pcb_harmonize"])
    img_B = gen(pipe, "pcb_harmonize", comp)
    print(f"  B mean={img_B.mean():.3f}")
    del pipe; torch.cuda.empty_cache()

    # C (our fused approach): load -> fuse -> unload -> add delta
    print("\n[C] v3.4 FUSED via diffusers fuse_lora + delta@scale 0 ...")
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.transformer.eval()
    pipe.load_lora_weights(V34_DIR, weight_name="default.safetensors", adapter_name="pcb_harmonize")
    pipe.set_adapters(["pcb_harmonize"])
    print("  -> fuse_lora ...")
    pipe.fuse_lora(adapter_names=["pcb_harmonize"], lora_scale=1.0)
    print("  -> unload_lora_weights ...")
    pipe.unload_lora_weights()

    # Add fresh delta LoRA (separate from v3.4, now fused in base)
    delta_cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
        init_lora_weights="gaussian",
    )
    pipe.transformer.add_adapter(delta_cfg, adapter_name="delta")

    # Force delta scale to 0 so only fused v3.4 should matter
    from peft.tuners.lora.layer import BaseTunerLayer
    for m in pipe.transformer.modules():
        if isinstance(m, BaseTunerLayer) and "delta" in getattr(m, "scaling", {}):
            m.scaling["delta"] = 0.0

    img_C = gen(pipe, "delta", comp)
    print(f"  C mean={img_C.mean():.3f}")

    # Compare
    diff_BC = (img_B - img_C).abs().mean().item()
    diff_B = img_B.abs().mean().item()
    print(f"\n||B - C|| = {diff_BC:.5f}")
    print(f"||B||     = {diff_B:.5f}")
    print(f"ratio     = {diff_BC/diff_B:.4f}")
    if diff_BC < 0.005:
        print("[PASS] C matches B exactly -> v3.4 is fully fused")
    elif diff_BC < 0.02:
        print("[PASS-SOFT] C nearly matches B -> fuse effective (minor numeric drift)")
    else:
        print("[FAIL] C differs from B materially -> something still wrong")

    # Save for inspection
    from torchvision.utils import save_image
    out = "/projects/_ssd/xrssd/logs/v34_merge_check_v2"
    os.makedirs(out, exist_ok=True)
    save_image(img_B, f"{out}/B_v34_adapter.png")
    save_image(img_C, f"{out}/C_fused_with_delta.png")
    save_image((img_B - img_C).abs() * 5, f"{out}/diff_BC_x5.png")  # amplified diff


if __name__ == "__main__":
    main()
