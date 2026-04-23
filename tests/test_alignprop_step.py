"""Phase 2 test: end-to-end AlignProp training step.

Loads FLUX + delta LoRA + DINO reward model. Uses a real v2.2 composite with
known (bbox, class) list, runs one alignprop_step, verifies outputs.
"""
import os, sys, time, json, csv
import numpy as np
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig
from omini.pipeline.flux_omini import Condition
from omini.train_flux.flux_sample_with_grad import flux_sample_with_grad, prepare_condition_data
from omini.train_flux.reward_dino import DinoLocalReward, CAT_NAMES
from omini.train_flux.mask_utils import bboxes_to_mask, scale_bboxes
from omini.train_flux.alignprop_step import alignprop_step

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
DINO_CKPT = "/projects/_ssd/xrssd/rewards/dino_cls_v2_2_transfix/best.pt"
DTYPE = torch.bfloat16
DEVICE = "cuda"


def load_pipe_with_delta():
    print("[load] FLUX + delta LoRA...")
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    pipe.transformer.train()                      # needed for gradient_checkpointing
    pipe.transformer.gradient_checkpointing = True
    pipe.transformer.gc_stride_double = 1
    pipe.transformer.gc_stride_single = 1         # more aggressive ckpt for Phase 2 memory headroom

    lora_cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        init_lora_weights="gaussian",
    )
    pipe.transformer.add_adapter(lora_cfg, adapter_name="delta")
    for n, p in pipe.transformer.named_parameters():
        if "lora_" in n and "delta" in n:
            p.requires_grad_(True)
    delta_params = [(n, p) for n, p in pipe.transformer.named_parameters() if p.requires_grad]
    print(f"  {sum(p.numel() for _,p in delta_params)/1e6:.2f}M delta params")
    return pipe, delta_params


def build_fake_pcb_composite(H=1024, W=1024):
    """Synthetic composite: paste 6 colored rectangles as 'components' at known bboxes.

    Returns: (PIL.Image, bboxes, classes) where classes are ints in 0..8.
    """
    img = Image.new("RGB", (W, H), (40, 80, 40))  # green-ish board
    np.random.seed(0)
    arr = np.array(img)
    # Define 6 component regions spread around the board
    specs = [
        # (x, y, w, h, cat_idx, colour)
        (100, 100, 180, 60,  0, (20, 20, 20)),       # Resistor: small dark chip
        (350, 200, 120, 60,  1, (180, 160, 40)),     # Capacitor: yellowish
        (600, 150, 140, 140, 7, (30, 30, 40)),       # IC: large dark square
        (200, 500, 60,  40,  6, (40, 40, 40)),       # Transistor: small dark
        (500, 600, 300, 80,  3, (120, 90, 60)),      # Connector: brown, long
        (100, 800, 100, 100, 8, (180, 180, 180)),    # Oscillator: silver
    ]
    bboxes, classes = [], []
    for x, y, w, h, ci, c in specs:
        arr[y:y+h, x:x+w, 0] = c[0]
        arr[y:y+h, x:x+w, 1] = c[1]
        arr[y:y+h, x:x+w, 2] = c[2]
        bboxes.append((x, y, x+w, y+h))
        classes.append(ci)
    img = Image.fromarray(arr)
    return img, bboxes, classes


def main():
    print("="*60)
    print("Phase 2: AlignProp end-to-end step")
    print("="*60)

    # Choose resolution via env var (default 1024 for full test, 512 for quick iter)
    RES = int(os.environ.get("TEST_RES", "1024"))
    K = int(os.environ.get("TEST_K", "3"))
    print(f"[cfg] resolution={RES}x{RES}, k_grad_steps={K}")

    pipe, delta_params = load_pipe_with_delta()

    # Load DINO reward in fp32 (small model, numerical stability)
    print("[load] DINO reward model...")
    reward_model = DinoLocalReward(DINO_CKPT, device=DEVICE, dtype=torch.float32)

    # Build composite at target resolution
    print(f"[data] building synthetic {RES}x{RES} composite...")
    # Compose at 1024, downscale if needed
    comp_1024, bboxes_1024, classes = build_fake_pcb_composite(1024, 1024)
    if RES != 1024:
        comp = comp_1024.resize((RES, RES), Image.LANCZOS)
        bboxes = scale_bboxes(bboxes_1024, 1024, 1024, RES, RES)
    else:
        comp = comp_1024
        bboxes = bboxes_1024
    print(f"  {len(bboxes)} components: {[CAT_NAMES[c] for c in classes]}")

    cond = Condition(comp, adapter_setting="delta")

    # Pre-encode prompt + condition before text encoder offload
    print("[encode] prompt + condition, then offload text encoders...")
    cond_data = prepare_condition_data(pipe, [cond])
    pe, pool, _ = pipe.encode_prompt(prompt="a pcb board", prompt_2=None, device=DEVICE,
                                     num_images_per_prompt=1, max_sequence_length=512)
    mem_before = torch.cuda.memory_allocated() / 1e9
    pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    mem_after = torch.cuda.memory_allocated() / 1e9
    print(f"  offload freed {mem_before - mem_after:.2f} GB")

    # Run one alignprop step
    for _, p in delta_params:
        p.grad = None
    torch.cuda.reset_peak_memory_stats()

    print(f"\n[step] running one alignprop step...")
    t0 = time.time()
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    logs = alignprop_step(
        pipe, prompt_embeds=pe, pooled_prompt_embeds=pool,
        condition_data=cond_data, bboxes=bboxes, classes=classes,
        reward_model=reward_model, delta_adapter_name="delta",
        height=RES, width=RES,
        num_inference_steps=10, k_grad_steps=K,
        lambda_preserve=1.0, mask_dilate_px=4,
        generator=gen,
    )
    dur = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[done] {dur:.1f}s, peak VRAM {peak_vram:.2f} GB")
    print(f"  loss:    {logs['loss']:+.4f}")
    print(f"  reward:  {logs['reward']:+.4f}  (mean log-P of correct class)")
    print(f"  preserv: {logs['preserv']:.4e}  (Focus-N-Fix term)")
    print(f"  per-class reward:")
    for k, v in logs["per_class_reward"].items():
        print(f"    {k:<12} {v:+.3f}")

    # Gradient check
    nz, z, n = 0, 0, 0
    gm = []
    for _, p in delta_params:
        if p.grad is None: n += 1
        elif p.grad.abs().mean().item() < 1e-12: z += 1
        else: nz += 1; gm.append(p.grad.abs().mean().item())
    print(f"\n[grad] nonzero={nz}, zero={z}, None={n}")
    if gm: print(f"       mean |grad|={np.mean(gm):.3e}, max={np.max(gm):.3e}")

    # Pass criteria
    assertions = [
        ("loss is finite",              np.isfinite(logs["loss"])),
        ("reward is finite",            np.isfinite(logs["reward"])),
        ("preserv is finite non-neg",   np.isfinite(logs["preserv"]) and logs["preserv"] >= 0),
        ("peak VRAM < 94 GB",           peak_vram < 94.0),
        ("delta LoRA got grads",        nz >= len(delta_params) // 2),
    ]
    all_pass = all(ok for _, ok in assertions)
    print(f"\n{'='*60}")
    for label, ok in assertions:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    print(f"\n  {'[READY] AlignProp step wired end-to-end' if all_pass else '[BLOCKED]'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
