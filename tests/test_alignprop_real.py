"""Phase 4: 20-step AlignProp training on REAL v2.2 composites.

Iterates real PCB boards, builds composites via ComponentBankV2.1, computes
DINO reward + preservation loss, runs optimizer on delta LoRA.
Expects reward to trend upward (unlike random synthetic).
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

# v2.2 uppercase -> DINO index (LED cat_id 6 and FUSE cat_id 11 skipped)
PCB_TO_DINO = {
    "RESISTOR": 0, "CAPACITOR": 1, "INDUCTOR": 2, "CONNECTOR": 3,
    "DIODE": 4, "SWITCH": 5, "TRANSISTOR": 6, "IC": 7, "OSCILLATOR": 8,
}


def load_pipe():
    print("[load] FLUX + v3.4 fused + delta LoRA...")
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    pipe.transformer.train()
    pipe.transformer.gradient_checkpointing = True
    pipe.transformer.gc_stride_double = 1
    pipe.transformer.gc_stride_single = 1

    # Fuse v3.4 into base
    pipe.load_lora_weights(V34_DIR, weight_name="default.safetensors",
                            adapter_name="pcb_harmonize")
    pipe.set_adapters(["pcb_harmonize"])
    pipe.fuse_lora(adapter_names=["pcb_harmonize"], lora_scale=1.0)
    pipe.unload_lora_weights()

    # Fresh delta
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
    print(f"  {sum(p.numel() for p in dp)/1e6:.2f}M delta params")
    return pipe, dp


def sample_from_dataset(ds, max_retries=5):
    """Pull a sample with at least 1 DINO-classifiable component."""
    for _ in range(max_retries):
        idx = random.randrange(len(ds))
        item = ds[idx]
        # Filter to DINO-classifiable classes
        bboxes, classes, names = [], [], []
        for bb, nm in zip(item["bboxes_xyxy"], item["cat_names"]):
            ci = PCB_TO_DINO.get(nm.upper())
            if ci is not None:
                bboxes.append(bb)
                classes.append(ci)
                names.append(nm)
        if len(bboxes) >= 2:
            return item["composite_pil"], item["prompt"], bboxes, classes, names
    raise RuntimeError("no sample with >=2 DINO-classifiable components after retries")


def main():
    N_STEPS = int(os.environ.get("N_STEPS", "20"))
    RES = int(os.environ.get("TEST_RES", "1024"))
    LR = float(os.environ.get("LR", "1e-4"))
    K = int(os.environ.get("K", "3"))
    LAMBDA = float(os.environ.get("LAMBDA", "1.0"))
    print(f"[cfg] steps={N_STEPS} res={RES} lr={LR} K={K} lambda={LAMBDA}")

    # Dataset
    print("[data] loading ComponentBankV2.1 + v2.2 dataset ...")
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

    # Pipe + reward
    pipe, delta_params = load_pipe()
    reward_model = DinoLocalReward(DINO_CKPT, device=DEVICE, dtype=torch.float32)
    opt = torch.optim.AdamW(delta_params, lr=LR, weight_decay=0.0)

    random.seed(0)
    print(f"\n{'step':>4} {'n':>3} {'reward':>9} {'preserv':>11} {'loss':>9} {'peak':>6} {'t_s':>5}  classes")
    print("-" * 90)
    logs = []
    for step in range(N_STEPS):
        # Sample a real composite + encode on the fly
        # (caller re-offloads text encoders after each encode — only if needed)
        composite, prompt, bboxes, classes, names = sample_from_dataset(ds)
        # Prompt encoding needs text_encoders — put them back on GPU before encoding
        pipe.text_encoder.to(DEVICE); pipe.text_encoder_2.to(DEVICE)
        cond = Condition(composite, adapter_setting="delta")
        cond_data = prepare_condition_data(pipe, [cond])
        pe, pool, _ = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=DEVICE,
            num_images_per_prompt=1, max_sequence_length=512,
        )
        pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        gen = torch.Generator(device=DEVICE).manual_seed(step)
        log = alignprop_step(
            pipe, prompt_embeds=pe, pooled_prompt_embeds=pool,
            condition_data=cond_data, bboxes=bboxes, classes=classes,
            reward_model=reward_model, delta_adapter_name="delta",
            height=RES, width=RES,
            num_inference_steps=10, k_grad_steps=K,
            lambda_preserve=LAMBDA, mask_dilate_px=4,
            generator=gen,
        )
        torch.nn.utils.clip_grad_norm_(delta_params, max_norm=1.0)
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        dt = time.time() - t0
        cl_str = ",".join(str(c) for c in classes[:6])
        print(f"{step:>4} {len(bboxes):>3} {log['reward']:>+9.3f} {log['preserv']:>11.3e} "
              f"{log['loss']:>+9.3f} {peak:>6.1f} {dt:>5.1f}  [{cl_str}]")
        logs.append(log)

    # Summary + trend
    rewards = np.array([l["reward"] for l in logs])
    print(f"\n[summary] N={N_STEPS}")
    print(f"  reward   first5 mean: {rewards[:5].mean():+.3f}")
    print(f"  reward   last5 mean:  {rewards[-5:].mean():+.3f}")
    print(f"  improvement:          {rewards[-5:].mean() - rewards[:5].mean():+.3f}")
    print(f"  peak VRAM all runs:   {max(l.get('peak_gb', 0) for l in logs if 'peak_gb' in l):.2f} GB (last: {peak:.2f})")

    if rewards[-5:].mean() > rewards[:5].mean():
        print(f"\n  [TREND_UP] reward increasing — learning signal real")
    else:
        print(f"\n  [FLAT/DOWN] reward not trending up — investigate (need more steps, or λ too high)")


if __name__ == "__main__":
    main()
