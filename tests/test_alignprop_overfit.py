"""Phase 4-B: overfit-one-composite sanity test.

Pick one real v2.2 composite. Run N steps on it, each with different noise.
Expectation: reward should monotonically increase OR clearly trend up,
proving the training signal is real (variance in real-data run was sample
variance, not broken gradient).
"""
import os, sys, time, random
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig
from omini.pipeline.flux_omini import Condition
from omini.train_flux.flux_sample_with_grad import prepare_condition_data
from omini.train_flux.reward_dino import DinoLocalReward, CAT_NAMES
from omini.train_flux.alignprop_step import alignprop_step
from train_pcb_v3_4 import PCBHarmonizeDatasetV3_4
from lib.component_bank_v2_1 import ComponentBankV2_1

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
DINO_CKPT = "/projects/_ssd/xrssd/rewards/dino_cls_v2_2_transfix/best.pt"
V34_DIR   = "/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000"
DATA_DIR  = "/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass"
DTYPE = torch.bfloat16
DEVICE = "cuda"

PCB_TO_DINO = {
    "RESISTOR": 0, "CAPACITOR": 1, "INDUCTOR": 2, "CONNECTOR": 3,
    "DIODE": 4, "SWITCH": 5, "TRANSISTOR": 6, "IC": 7, "OSCILLATOR": 8,
}


def load_pipe():
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    pipe.transformer.train()
    pipe.transformer.gradient_checkpointing = True
    pipe.transformer.gc_stride_double = 1
    pipe.transformer.gc_stride_single = 1

    pipe.load_lora_weights(V34_DIR, weight_name="default.safetensors",
                            adapter_name="pcb_harmonize")
    pipe.set_adapters(["pcb_harmonize"])
    pipe.fuse_lora(adapter_names=["pcb_harmonize"], lora_scale=1.0)
    pipe.unload_lora_weights()

    delta_cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
        init_lora_weights="gaussian",
    )
    pipe.transformer.add_adapter(delta_cfg, adapter_name="delta")
    for n, p in pipe.transformer.named_parameters():
        if "lora_" in n and "delta" in n:
            p.requires_grad_(True)
    dp = [p for n, p in pipe.transformer.named_parameters() if p.requires_grad]
    return pipe, dp


def pick_one_composite(ds, min_comp=5, max_comp=15, max_tries=100):
    """Find a composite with moderate number of DINO-classifiable components."""
    random.seed(123)
    for _ in range(max_tries):
        idx = random.randrange(len(ds))
        item = ds[idx]
        bboxes, classes = [], []
        for bb, nm in zip(item["bboxes_xyxy"], item["cat_names"]):
            ci = PCB_TO_DINO.get(nm.upper())
            if ci is not None:
                bboxes.append(bb); classes.append(ci)
        if min_comp <= len(bboxes) <= max_comp:
            return item["composite_pil"], item["prompt"], bboxes, classes, idx
    raise RuntimeError("no suitable composite found")


def main():
    N_STEPS = int(os.environ.get("N_STEPS", "30"))
    RES = int(os.environ.get("TEST_RES", "1024"))
    LR = float(os.environ.get("LR", "3e-5"))
    K = int(os.environ.get("K", "3"))
    LAMBDA = float(os.environ.get("LAMBDA", "1.0"))
    ACCUM = int(os.environ.get("ACCUM", "4"))
    print(f"[cfg] steps={N_STEPS} res={RES} lr={LR} K={K} lambda={LAMBDA} accum={ACCUM}")

    anno_dir = os.path.join(DATA_DIR, "annotation/train")
    image_dir = os.path.join(DATA_DIR, "image/train")
    bank = ComponentBankV2_1(anno_dir=anno_dir, image_dir=image_dir)
    ds = PCBHarmonizeDatasetV3_4(
        anno_dir=anno_dir, image_dir=image_dir,
        condition_size=(RES, RES), target_size=(RES, RES),
        component_bank=bank,
        zoom_prob=0.4, zoom_crop_size=256,
        drop_text_prob=0.0, drop_image_prob=0.0,
        return_annotations=True,
    )
    print(f"  dataset size: {len(ds)}")

    # Pick ONE composite
    composite, prompt, bboxes, classes, chosen_idx = pick_one_composite(ds)
    class_names = [CAT_NAMES[c] for c in classes]
    print(f"\n[data] chose ds[{chosen_idx}] with {len(bboxes)} components")
    print(f"  classes: {class_names}")
    print(f"  prompt: {prompt[:120]}...")

    pipe, delta_params = load_pipe()
    reward_model = DinoLocalReward(DINO_CKPT, device=DEVICE, dtype=torch.float32)
    opt = torch.optim.AdamW(delta_params, lr=LR, weight_decay=0.0)

    # Pre-encode once (same composite every step)
    pipe.text_encoder.to(DEVICE); pipe.text_encoder_2.to(DEVICE)
    cond = Condition(composite, adapter_setting="delta")
    cond_data = prepare_condition_data(pipe, [cond])
    pe, pool, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=DEVICE,
                                     num_images_per_prompt=1, max_sequence_length=512)
    pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    # Training loop on this one composite
    print(f"\n{'step':>4} {'reward':>9} {'r_std':>7} {'preserv':>11} {'loss':>9} {'peak':>6} {'t_s':>5}")
    print("-" * 63)
    rewards, preservs = [], []
    for step in range(N_STEPS):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        # Unique seed block per step so ACCUM seeds don't collide across steps
        gen = torch.Generator(device=DEVICE).manual_seed(1000 + step * ACCUM)
        log = alignprop_step(
            pipe, prompt_embeds=pe, pooled_prompt_embeds=pool,
            condition_data=cond_data, bboxes=bboxes, classes=classes,
            reward_model=reward_model, delta_adapter_name="delta",
            height=RES, width=RES,
            num_inference_steps=10, k_grad_steps=K,
            lambda_preserve=LAMBDA, mask_dilate_px=4,
            generator=gen, num_accum=ACCUM,
        )
        torch.nn.utils.clip_grad_norm_(delta_params, max_norm=1.0)
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        dt = time.time() - t0
        rewards.append(log["reward"]); preservs.append(log["preserv"])
        print(f"{step:>4} {log['reward']:>+9.4f} {log.get('reward_stdev',0):>7.3f} "
              f"{log['preserv']:>11.3e} {log['loss']:>+9.4f} {peak:>6.1f} {dt:>5.1f}")

    # Trend analysis
    rewards = np.array(rewards)
    W = 5   # window
    early = rewards[:W].mean()
    late = rewards[-W:].mean()
    # Linear fit slope
    x = np.arange(len(rewards))
    slope, intercept = np.polyfit(x, rewards, 1)

    print(f"\n[trend analysis on SAME composite, varying noise]")
    print(f"  first {W} mean: {early:+.4f}")
    print(f"  last {W} mean:  {late:+.4f}")
    print(f"  delta:          {late-early:+.4f}")
    print(f"  linear slope:   {slope:+.4e} per step (total gain over {N_STEPS}: {slope*N_STEPS:+.4f})")
    print(f"  preserv:        final={preservs[-1]:.2e} (was 0 at step 0)")

    # Per-class reward at first vs last step (requires another forward)
    # Skipped to save time; trend analysis is main signal.

    if late > early + 0.1:
        verdict = "[PASS] reward trending UP — gradient signal is real"
    elif late < early - 0.1:
        verdict = "[FAIL] reward trending DOWN — signal is going wrong direction"
    else:
        verdict = "[FLAT] reward not moving — check LR or lambda"
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
