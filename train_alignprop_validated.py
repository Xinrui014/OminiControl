"""AlignProp training with fixed-val-set evaluation.

Trains delta LoRA on real v2.2 composites with gradient accumulation,
evaluates on a fixed 20-sample val set every EVAL_EVERY steps.
Primary signal: val_reward trajectory (not training reward).
"""
import os, sys, time, random, json
import numpy as np
import torch
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig
from omini.pipeline.flux_omini import Condition
from omini.train_flux.flux_sample_with_grad import prepare_condition_data
from omini.train_flux.reward_dino import DinoLocalReward, CAT_NAMES
from omini.train_flux.alignprop_step import alignprop_step
from omini.train_flux.eval_val_set import eval_on_val
from train_pcb_v3_4 import PCBHarmonizeDatasetV3_4
from lib.component_bank_v2_1 import ComponentBankV2_1

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
DINO_CKPT = "/projects/_ssd/xrssd/rewards/dino_cls_v2_2_transfix/best.pt"
V34_DIR   = "/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000"
DATA_DIR  = "/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass"
VAL_PT    = "/projects/_ssd/xrssd/rewards/alignprop_val_20.pt"
DTYPE = torch.bfloat16
DEVICE = "cuda"

PCB_TO_DINO = {"RESISTOR":0,"CAPACITOR":1,"INDUCTOR":2,"CONNECTOR":3,
               "DIODE":4,"SWITCH":5,"TRANSISTOR":6,"IC":7,"OSCILLATOR":8}


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

    pipe.load_lora_weights(V34_DIR, weight_name="default.safetensors", adapter_name="pcb_harmonize")
    pipe.set_adapters(["pcb_harmonize"])
    pipe.fuse_lora(adapter_names=["pcb_harmonize"], lora_scale=1.0)
    pipe.unload_lora_weights()

    delta_cfg = LoraConfig(r=8, lora_alpha=8,
        target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
        init_lora_weights="gaussian")
    pipe.transformer.add_adapter(delta_cfg, adapter_name="delta")
    for n, p in pipe.transformer.named_parameters():
        if "lora_" in n and "delta" in n:
            p.requires_grad_(True)
    dp = [p for n, p in pipe.transformer.named_parameters() if p.requires_grad]
    return pipe, dp


def sample_from_dataset(ds, min_c=3, max_tries=5):
    for _ in range(max_tries):
        idx = random.randrange(len(ds))
        item = ds[idx]
        bboxes, classes = [], []
        for bb, nm in zip(item["bboxes_xyxy"], item["cat_names"]):
            ci = PCB_TO_DINO.get(nm.upper())
            if ci is not None:
                bboxes.append(bb); classes.append(ci)
        if len(bboxes) >= min_c:
            return item["composite_pil"], item["prompt"], bboxes, classes
    return None


def save_delta_lora(pipe, out_path):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    sd = get_peft_model_state_dict(pipe.transformer, adapter_name="delta")
    save_file(sd, out_path)


def main():
    N_STEPS = int(os.environ.get("N_STEPS", "50"))
    RES = int(os.environ.get("TEST_RES", "1024"))
    LR = float(os.environ.get("LR", "1e-4"))
    K = int(os.environ.get("K", "3"))
    LAMBDA = float(os.environ.get("LAMBDA", "1.0"))
    ACCUM = int(os.environ.get("ACCUM", "4"))
    EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "10"))
    CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "25"))
    RUN_NAME = os.environ.get("RUN_NAME", "alignprop_v1")
    SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "OminiControl-PCB")
    WANDB_MODE = os.environ.get("WANDB_MODE", "online")

    run_dir = f"/projects/_ssd/xrssd/OminiControl/runs/{RUN_NAME}"
    ckpt_dir = f"{run_dir}/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = f"{run_dir}/train.jsonl"

    cfg = dict(steps=N_STEPS, res=RES, lr=LR, K=K, lambda_preserve=LAMBDA,
               accum=ACCUM, eval_every=EVAL_EVERY, ckpt_every=CKPT_EVERY,
               method="alignprop+focus_n_fix", base="v3.4_ckpt6k_resumed",
               reward="dino_pad20_transfix", delta_rank=8)

    if wandb is not None:
        wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config=cfg, mode=WANDB_MODE)
        print(f"  wandb: {wandb.run.url if WANDB_MODE == 'online' else '(offline)'}")
    else:
        print("  wandb not installed — skipping online logging")

    print(f"[cfg] steps={N_STEPS} res={RES} lr={LR} K={K} lambda={LAMBDA} accum={ACCUM}")
    print(f"      eval_every={EVAL_EVERY} ckpt_every={CKPT_EVERY} run={RUN_NAME}")

    # Dataset
    anno_dir = os.path.join(DATA_DIR, "annotation/train")
    image_dir = os.path.join(DATA_DIR, "image/train")
    bank = ComponentBankV2_1(anno_dir=anno_dir, image_dir=image_dir)
    ds = PCBHarmonizeDatasetV3_4(
        anno_dir=anno_dir, image_dir=image_dir,
        condition_size=(RES, RES), target_size=(RES, RES),
        component_bank=bank, zoom_prob=0.4, zoom_crop_size=256,
        drop_text_prob=0.0, drop_image_prob=0.0,
        return_annotations=True,
    )
    print(f"  train set: {len(ds)} samples")

    val_samples = torch.load(VAL_PT, weights_only=False)
    print(f"  val set: {len(val_samples)} samples from {VAL_PT}")

    pipe, delta_params = load_pipe()
    reward_model = DinoLocalReward(DINO_CKPT, device=DEVICE, dtype=torch.float32)
    opt = torch.optim.AdamW(delta_params, lr=LR, weight_decay=0.0)
    print(f"  delta params: {sum(p.numel() for p in delta_params)/1e6:.2f}M")

    random.seed(SEED_OFFSET)

    # ---- Initial val eval (before any training) ----
    print("\n[eval] initial (step 0)...")
    t0 = time.time()
    init_val = eval_on_val(pipe, val_samples, reward_model,
                           delta_adapter_name="delta",
                           height=RES, width=RES,
                           num_inference_steps=10,
                           eval_noise_seeds=(42, 43, 44))
    print(f"  val_reward={init_val['val_reward_mean']:+.4f} (std {init_val['val_reward_std']:.3f})")
    print(f"  eval took {time.time()-t0:.0f}s")
    with open(log_path, "a") as f:
        f.write(json.dumps({"step": 0, "event": "eval", **init_val}) + "\n")
    if wandb is not None:
        wandb.log({f"eval/{k}": v for k, v in init_val.items() if isinstance(v, (int, float))}, step=0)

    # ---- Training loop ----
    print(f"\n{'step':>4} {'reward':>9} {'r_std':>6} {'preserv':>11} {'loss':>9} {'peak':>6} {'t_s':>5}  n")
    print("-" * 75)
    best_val = init_val["val_reward_mean"]

    for step in range(1, N_STEPS + 1):
        res = sample_from_dataset(ds)
        if res is None:
            print(f"  (skip step {step} — no sample)"); continue
        composite, prompt, bboxes, classes = res

        pipe.text_encoder.to(DEVICE); pipe.text_encoder_2.to(DEVICE)
        cond = Condition(composite, adapter_setting="delta")
        cond_data = prepare_condition_data(pipe, [cond])
        pe, pool, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=DEVICE,
                                         num_images_per_prompt=1, max_sequence_length=512)
        pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED_OFFSET + step * ACCUM)
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

        print(f"{step:>4} {log['reward']:>+9.4f} {log.get('reward_stdev',0):>6.3f} "
              f"{log['preserv']:>11.3e} {log['loss']:>+9.4f} {peak:>6.1f} {dt:>5.1f}  {len(bboxes)}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "event": "train",
                                "reward": log["reward"], "preserv": log["preserv"],
                                "loss": log["loss"], "n_comp": len(bboxes),
                                "peak_gb": peak, "time_s": dt}) + "\n")
        if wandb is not None:
            wandb.log({
                "train/reward": log["reward"],
                "train/reward_stdev": log.get("reward_stdev", 0),
                "train/preserv": log["preserv"],
                "train/loss": log["loss"],
                "train/n_comp": len(bboxes),
                "train/peak_gb": peak,
                "train/time_s": dt,
            }, step=step)

        # Periodic val eval
        if step % EVAL_EVERY == 0:
            print(f"\n[eval] step {step}...")
            t0 = time.time()
            val = eval_on_val(pipe, val_samples, reward_model,
                              delta_adapter_name="delta",
                              height=RES, width=RES,
                              num_inference_steps=10,
                              eval_noise_seeds=(42, 43, 44))
            print(f"  val_reward={val['val_reward_mean']:+.4f} (std {val['val_reward_std']:.3f}) — eval {time.time()-t0:.0f}s")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "event": "eval", **val}) + "\n")
            if wandb is not None:
                wandb.log({f"eval/{k}": v for k, v in val.items() if isinstance(v, (int, float))}, step=step)

            if val["val_reward_mean"] > best_val:
                best_val = val["val_reward_mean"]
                save_delta_lora(pipe, f"{ckpt_dir}/best.safetensors")
                print(f"  ↑ new best val_reward={best_val:+.4f}, saved ckpt")

        if step % CKPT_EVERY == 0:
            save_delta_lora(pipe, f"{ckpt_dir}/step_{step}.safetensors")

    # Final save + final eval
    save_delta_lora(pipe, f"{ckpt_dir}/final.safetensors")
    print(f"\n[final eval]")
    val = eval_on_val(pipe, val_samples, reward_model,
                      delta_adapter_name="delta",
                      height=RES, width=RES,
                      num_inference_steps=10,
                      eval_noise_seeds=(42, 43, 44))
    print(f"  val_reward={val['val_reward_mean']:+.4f}")
    with open(log_path, "a") as f:
        f.write(json.dumps({"step": N_STEPS, "event": "final_eval", **val}) + "\n")

    print(f"\n[summary]")
    print(f"  initial val_reward: {init_val['val_reward_mean']:+.4f}")
    print(f"  final   val_reward: {val['val_reward_mean']:+.4f}")
    print(f"  best    val_reward: {best_val:+.4f}")
    print(f"  improvement:        {val['val_reward_mean'] - init_val['val_reward_mean']:+.4f}")

    if wandb is not None:
        wandb.log({f"final_eval/{k}": v for k, v in val.items() if isinstance(v, (int, float))}, step=N_STEPS)
        wandb.summary["improvement"] = val["val_reward_mean"] - init_val["val_reward_mean"]
        wandb.summary["best_val_reward"] = best_val
        wandb.finish()


if __name__ == "__main__":
    main()
