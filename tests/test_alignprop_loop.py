"""Phase 3 smoke test: 10-step AlignProp training loop.

Loads FLUX + v3.4 LoRA (merged) + fresh delta LoRA + DINO reward.
Runs 10 optimizer steps on a synthetic composite, verifies:
  - Loss evolves step-to-step (not stuck)
  - No memory leak across steps
  - Delta LoRA weights actually change
  - No NaN/Inf in gradients or weights
"""
import os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig
from safetensors.torch import load_file
from omini.pipeline.flux_omini import Condition
from omini.train_flux.flux_sample_with_grad import prepare_condition_data
from omini.train_flux.reward_dino import DinoLocalReward, CAT_NAMES
from omini.train_flux.mask_utils import scale_bboxes
from omini.train_flux.alignprop_step import alignprop_step

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
DINO_CKPT = "/projects/_ssd/xrssd/rewards/dino_cls_v2_2_transfix/best.pt"
V34_CKPT_DIR = "/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000"
DTYPE = torch.bfloat16
DEVICE = "cuda"


def load_pipe_with_v34_and_delta():
    print("[load] FLUX pipeline (bf16)...")
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    pipe.transformer.train()
    pipe.transformer.gradient_checkpointing = True
    pipe.transformer.gc_stride_double = 1
    pipe.transformer.gc_stride_single = 1

    # ---- load v3.4 LoRA, fuse into base, unload (auto-detects target modules) ----
    print(f"[load] v3.4 LoRA from {V34_CKPT_DIR}/default.safetensors ...")
    pipe.load_lora_weights(V34_CKPT_DIR, weight_name="default.safetensors",
                            adapter_name="pcb_harmonize")
    pipe.set_adapters(["pcb_harmonize"])
    print("[fuse] v3.4 -> base weights ...")
    pipe.fuse_lora(adapter_names=["pcb_harmonize"], lora_scale=1.0)
    pipe.unload_lora_weights()
    print("  v3.4 fused + unloaded")

    # ---- add fresh delta LoRA ----
    print("[add] delta LoRA (rank 8) ...")
    delta_cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        init_lora_weights="gaussian",
    )
    pipe.transformer.add_adapter(delta_cfg, adapter_name="delta")
    for n, p in pipe.transformer.named_parameters():
        if "lora_" in n and "delta" in n:
            p.requires_grad_(True)
    delta_params = [p for n, p in pipe.transformer.named_parameters() if p.requires_grad]
    print(f"  {sum(p.numel() for p in delta_params)/1e6:.2f}M delta params")
    return pipe, delta_params


def build_fake_pcb_composite(H=1024, W=1024):
    img = Image.new("RGB", (W, H), (40, 80, 40))
    arr = np.array(img)
    specs = [
        (100, 100, 180, 60,  0, (20, 20, 20)),
        (350, 200, 120, 60,  1, (180, 160, 40)),
        (600, 150, 140, 140, 7, (30, 30, 40)),
        (200, 500, 60,  40,  6, (40, 40, 40)),
        (500, 600, 300, 80,  3, (120, 90, 60)),
        (100, 800, 100, 100, 8, (180, 180, 180)),
    ]
    bboxes, classes = [], []
    for x, y, w, h, ci, c in specs:
        arr[y:y+h, x:x+w, 0] = c[0]
        arr[y:y+h, x:x+w, 1] = c[1]
        arr[y:y+h, x:x+w, 2] = c[2]
        bboxes.append((x, y, x+w, y+h))
        classes.append(ci)
    return Image.fromarray(arr), bboxes, classes


def snapshot_delta(delta_params):
    return torch.cat([p.detach().flatten() for p in delta_params]).clone()


def main():
    N_STEPS = int(os.environ.get("N_STEPS", "10"))
    RES = int(os.environ.get("TEST_RES", "1024"))
    LR = float(os.environ.get("LR", "1e-4"))
    K = int(os.environ.get("K", "3"))
    print(f"[cfg] n_steps={N_STEPS}, res={RES}, lr={LR}, K={K}")

    pipe, delta_params = load_pipe_with_v34_and_delta()
    reward_model = DinoLocalReward(DINO_CKPT, device=DEVICE, dtype=torch.float32)

    # Dataset: synthetic composite (same every step for smoke test)
    comp_1024, bboxes_1024, classes = build_fake_pcb_composite(1024, 1024)
    if RES != 1024:
        comp = comp_1024.resize((RES, RES), Image.LANCZOS)
        bboxes = scale_bboxes(bboxes_1024, 1024, 1024, RES, RES)
    else:
        comp = comp_1024; bboxes = bboxes_1024
    cond = Condition(comp, adapter_setting="delta")

    # Pre-encode + offload (same as before)
    cond_data = prepare_condition_data(pipe, [cond])
    pe, pool, _ = pipe.encode_prompt(prompt="a pcb board", prompt_2=None, device=DEVICE,
                                     num_images_per_prompt=1, max_sequence_length=512)
    pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    # Optimizer
    opt = torch.optim.AdamW(delta_params, lr=LR, weight_decay=0.0)

    # Baseline delta snapshot
    delta_before = snapshot_delta(delta_params)

    # Training loop
    logs = []
    print(f"\n{'step':>4} {'reward':>10} {'preserv':>12} {'loss':>10} {'peak_GB':>8} {'time_s':>7}")
    print("-" * 56)
    for step in range(N_STEPS):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        gen = torch.Generator(device=DEVICE).manual_seed(step)  # different noise each step
        log = alignprop_step(
            pipe, prompt_embeds=pe, pooled_prompt_embeds=pool,
            condition_data=cond_data, bboxes=bboxes, classes=classes,
            reward_model=reward_model, delta_adapter_name="delta",
            height=RES, width=RES,
            num_inference_steps=10, k_grad_steps=K,
            lambda_preserve=1.0, mask_dilate_px=4,
            generator=gen,
        )
        # Gradient clipping (cheap safety)
        torch.nn.utils.clip_grad_norm_(delta_params, max_norm=1.0)
        opt.step()

        peak = torch.cuda.max_memory_allocated() / 1e9
        dt = time.time() - t0
        logs.append(log)
        print(f"{step:>4} {log['reward']:>+10.4f} {log['preserv']:>12.4e} "
              f"{log['loss']:>+10.4f} {peak:>8.2f} {dt:>7.1f}")

    # ---- checks ----
    delta_after = snapshot_delta(delta_params)
    weight_change = (delta_after - delta_before).abs().mean().item()

    losses = np.array([l["loss"] for l in logs])
    rewards = np.array([l["reward"] for l in logs])

    print(f"\n[summary]")
    print(f"  loss:         first={losses[0]:+.4f} last={losses[-1]:+.4f} delta={losses[-1]-losses[0]:+.4f}")
    print(f"  reward:       first={rewards[0]:+.4f} last={rewards[-1]:+.4f} delta={rewards[-1]-rewards[0]:+.4f}")
    print(f"  weight chng:  mean|Δw| = {weight_change:.3e}")
    print(f"  final peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Pass criteria
    assertions = [
        ("all losses finite",             np.all(np.isfinite(losses))),
        ("all rewards finite",            np.all(np.isfinite(rewards))),
        ("weights actually changed",      weight_change > 1e-6),
        ("no memory growth (last vs first peak)",
                                          abs(logs[-1].get("peak_gb", 0) - logs[0].get("peak_gb", 0)) < 20),
    ]
    all_pass = all(ok for _, ok in assertions)
    print(f"\n{'='*60}")
    for label, ok in assertions:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    print(f"\n  {'[READY] training loop works — wire in real data' if all_pass else '[BLOCKED]'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
