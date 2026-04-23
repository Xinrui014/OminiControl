"""AlignProp training with DDP gradient sync across multiple GPUs.

Launch:
  torchrun --nproc_per_node=4 train_alignprop_validated_ddp.py

Each rank:
  - Loads full FLUX pipeline + v3.4 fused + delta LoRA (same everywhere)
  - Samples its own composite per step (different seed per rank)
  - Computes its own accum forward/backward passes
  - all_reduce(SUM) gradients across ranks, divide by world_size
  - opt.step() (same grad → same update on all ranks)

Val eval:
  - Val set split evenly across ranks
  - Each rank evaluates its shard
  - all_gather rewards, rank 0 aggregates + logs

Rank 0 only: wandb, ckpt save, console print.
"""
import os, sys, time, random, json
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig
from omini.pipeline.flux_omini import Condition
from omini.train_flux.flux_sample_with_grad import prepare_condition_data, flux_sample_with_grad
from omini.train_flux.reward_dino import DinoLocalReward, CAT_NAMES
from omini.train_flux.alignprop_step import alignprop_step, _set_adapter_scale
from omini.train_flux.mask_utils import bboxes_to_mask
from train_pcb_v3_4 import PCBHarmonizeDatasetV3_4
from lib.component_bank_v2_1 import ComponentBankV2_1

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
DINO_CKPT = "/projects/_ssd/xrssd/rewards/dino_cls_v2_2_transfix/best.pt"
V34_DIR   = "/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000"
DATA_DIR  = "/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass"
VAL_PT    = "/projects/_ssd/xrssd/rewards/alignprop_val_20.pt"
DTYPE = torch.bfloat16

PCB_TO_DINO = {"RESISTOR":0,"CAPACITOR":1,"INDUCTOR":2,"CONNECTOR":3,
               "DIODE":4,"SWITCH":5,"TRANSISTOR":6,"IC":7,"OSCILLATOR":8}

# Class-weighted reward — up-weight rare / regressing classes seen in 50-step run.
# Matches DINO classifier's class order (name strings from reward_dino.CAT_NAMES).
CLASS_WEIGHTS_DEFAULT = {
    "Resistor":    1.0,
    "Capacitor":   1.0,
    "Connector":   1.0,
    "IC":          1.0,
    "Diode":       2.0,   # baseline bad, barely moved in 50-step
    "Inductor":    3.0,   # REGRESSED in 50-step — priority fix
    "Switch":      3.0,   # rare, missing from prior val set
    "Transistor":  3.0,   # rare, big improvement potential
    "Oscillator":  3.0,   # very rare
}


# ---------- DDP helpers ----------

def setup_ddp():
    """Returns (rank, world_size, local_rank, device)."""
    if "RANK" not in os.environ:
        # Single-process fallback (for debug)
        return 0, 1, 0, torch.device("cuda:0")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device(f"cuda:{local_rank}")


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_grads(params, world_size):
    """Average gradients across all ranks (manual DDP sync)."""
    if world_size == 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def all_gather_scalars(value, world_size, device):
    """Returns a list of `value` (float) from all ranks."""
    if world_size == 1:
        return [value]
    t = torch.tensor([value], device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    return [float(g.item()) for g in gathered]


# ---------- Pipeline loading ----------

def load_pipe(device):
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(device)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    pipe.transformer.train()
    pipe.transformer.gradient_checkpointing = True
    pipe.transformer.gc_stride_double = 1
    pipe.transformer.gc_stride_single = 1

    # v3.4 loaded as adapter, NOT fused. OminiControl design: v3.4 is a
    # condition-only adapter (active on composite branch via specify_lora,
    # zeroed on main branch via main_adapter=None). Fusing would apply v3.4
    # everywhere and corrupt main denoising.
    pipe.load_lora_weights(V34_DIR, weight_name="default.safetensors", adapter_name="pcb_harmonize")

    delta_cfg = LoraConfig(r=8, lora_alpha=8,
        target_modules=["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"],
        init_lora_weights="gaussian")
    pipe.transformer.add_adapter(delta_cfg, adapter_name="delta")
    # Activate both: pcb_harmonize gates condition branch; delta gates main branch.
    pipe.set_adapters(["pcb_harmonize", "delta"])
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


def eval_on_val_distributed(
    pipe, val_samples, reward_model, rank, world_size, device,
    delta_adapter_name="delta",
    height=1024, width=1024,
    num_inference_steps=10,
    eval_noise_seeds=(42, 43, 44),
    image_out_dir=None,
    n_images_to_save=0,
    step=None,
):
    """Each rank evaluates a shard of val_samples, then all_gather results.

    If image_out_dir + n_images_to_save > 0: rank 0 saves the first N eval
    samples (first seed only) as side-by-side PNGs [composite | generated] and
    returns a "wandb_images" list in result (rank 0 only) for wandb logging.
    """
    _set_adapter_scale(pipe, delta_adapter_name, 1.0)

    # Each rank takes samples at indices [rank, rank+world, rank+2*world, ...]
    my_samples = val_samples[rank::world_size]
    local_rewards = []           # per-sample mean reward
    local_per_class = {}         # class -> list of per-seed rewards
    wandb_images = []            # rank 0 only, collected for wandb logging

    save_this_rank = (rank == 0) and image_out_dir is not None and n_images_to_save > 0
    if save_this_rank:
        os.makedirs(image_out_dir, exist_ok=True)

    for si, sample in enumerate(my_samples):
        composite = sample["composite"]; prompt = sample["prompt"]
        bboxes = sample["bboxes"]; classes = sample["classes"]

        pipe.text_encoder.to(device); pipe.text_encoder_2.to(device)
        cond = Condition(composite, adapter_setting="pcb_harmonize")
        cond_data = prepare_condition_data(pipe, [cond])
        pe, pool, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=device,
                                         num_images_per_prompt=1, max_sequence_length=512)
        pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

        seed_rewards = []
        for seed_idx, seed in enumerate(eval_noise_seeds):
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                image = flux_sample_with_grad(
                    pipe, prompt_embeds=pe, pooled_prompt_embeds=pool,
                    condition_data=cond_data, main_adapter=delta_adapter_name,
                    height=height, width=width,
                    num_inference_steps=num_inference_steps, k_grad_steps=1,
                    generator=gen, vae_checkpoint=False,
                )
                r, pc, _pc2 = reward_model(
                    image, bboxes, classes,
                    return_per_class=True, return_per_component=False,
                )
            seed_rewards.append(r.item())
            for k, v in pc.items():
                local_per_class.setdefault(k, []).append(v)

            # Save first seed of first N samples (rank 0) as side-by-side PNG
            if save_this_rank and si < n_images_to_save and seed_idx == 0:
                img_t = ((image[0].float() + 1.0) / 2.0).clamp(0, 1).cpu()
                from torchvision.transforms.functional import to_pil_image
                gen_pil = to_pil_image(img_t)
                comp_pil = composite.resize((width, height), Image.LANCZOS) if composite.size != (width, height) else composite
                panel = Image.new("RGB", (width * 2, height), (20, 20, 20))
                panel.paste(comp_pil, (0, 0)); panel.paste(gen_pil, (width, 0))
                tag = f"step{step if step is not None else 'x'}_sample{si:02d}_r{float(r.item()):+.3f}"
                out_path = os.path.join(image_out_dir, f"{tag}.png")
                panel.save(out_path)
                if wandb is not None:
                    wandb_images.append(wandb.Image(panel, caption=tag))

        local_rewards.append(float(np.mean(seed_rewards)))

    # All-gather per-sample rewards (padded so all ranks have same length)
    max_n = max(len(val_samples) // world_size + 1, 1)
    padded = local_rewards + [0.0] * (max_n - len(local_rewards))
    valid  = [1.0] * len(local_rewards) + [0.0] * (max_n - len(local_rewards))

    if world_size > 1:
        t_rew = torch.tensor(padded, device=device, dtype=torch.float64)
        t_val = torch.tensor(valid,  device=device, dtype=torch.float64)
        gathered_rew = [torch.zeros_like(t_rew) for _ in range(world_size)]
        gathered_val = [torch.zeros_like(t_val) for _ in range(world_size)]
        dist.all_gather(gathered_rew, t_rew)
        dist.all_gather(gathered_val, t_val)
        all_rewards = []
        for g_r, g_v in zip(gathered_rew, gathered_val):
            for r, v in zip(g_r.tolist(), g_v.tolist()):
                if v > 0.5:
                    all_rewards.append(r)
    else:
        all_rewards = local_rewards

    result = {
        "val_reward_mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "val_reward_std":  float(np.std(all_rewards)) if all_rewards else 0.0,
        "val_n_samples":   len(all_rewards),
    }

    # Per-class — all_reduce (sum, count) across ranks for accurate global means
    from omini.train_flux.reward_dino import CAT_NAMES as _CAT_NAMES
    sums = torch.zeros(len(_CAT_NAMES), device=device, dtype=torch.float64)
    counts = torch.zeros(len(_CAT_NAMES), device=device, dtype=torch.float64)
    for k, vs in local_per_class.items():
        if k in _CAT_NAMES:
            ci = _CAT_NAMES.index(k)
            sums[ci] = float(np.sum(vs))
            counts[ci] = float(len(vs))
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if is_main():
        for ci, name in enumerate(_CAT_NAMES):
            if counts[ci].item() > 0:
                result[f"val_reward_{name}"] = float(sums[ci].item() / counts[ci].item())
        if wandb_images:
            result["_wandb_images"] = wandb_images
    return result


# ---------- Training step with DDP sync ----------

def alignprop_step_ddp(
    pipe, prompt_embeds, pooled_prompt_embeds, condition_data,
    bboxes, classes, reward_model, delta_params, world_size, device,
    delta_adapter_name="delta",
    height=1024, width=1024,
    num_inference_steps=10, k_grad_steps=3,
    lambda_preserve=1.0, mask_dilate_px=4,
    guidance_scale=3.5,
    base_seed=42, num_accum=4,
    class_weights=None,
):
    """One step with local accum, then DDP grad sync, returns scalar logs."""
    logs = alignprop_step(
        pipe, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        condition_data=condition_data, bboxes=bboxes, classes=classes,
        reward_model=reward_model, delta_adapter_name=delta_adapter_name,
        height=height, width=width,
        num_inference_steps=num_inference_steps, k_grad_steps=k_grad_steps,
        lambda_preserve=lambda_preserve, mask_dilate_px=mask_dilate_px,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=device).manual_seed(base_seed),
        num_accum=num_accum,
        class_weights=class_weights,
    )
    # Sync grads across ranks BEFORE opt.step() (opt.step() called by caller)
    all_reduce_grads(delta_params, world_size)
    return logs


# ---------- Main ----------

def main():
    N_STEPS = int(os.environ.get("N_STEPS", "500"))
    RES = int(os.environ.get("TEST_RES", "1024"))
    LR = float(os.environ.get("LR", "1e-4"))
    K = int(os.environ.get("K", "3"))
    LAMBDA = float(os.environ.get("LAMBDA", "1.0"))
    ACCUM = int(os.environ.get("ACCUM", "2"))
    EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "25"))
    CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "50"))
    RUN_NAME = os.environ.get("RUN_NAME", "alignprop_ddp_v1")
    SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "OminiControl-PCB")
    VAL_PT_PATH = os.environ.get("VAL_PT", VAL_PT)
    USE_CLASS_WEIGHTS = os.environ.get("CLASS_WEIGHTS", "1") == "1"
    SAVE_IMAGES_N = int(os.environ.get("SAVE_IMAGES_N", "4"))  # per val eval, first N samples
    RESUME_FROM = os.environ.get("RESUME_FROM", "")  # path to step_N.safetensors to resume delta from
    START_STEP  = int(os.environ.get("START_STEP", "0"))  # resume step offset; loop runs (START_STEP, N_STEPS]
    WANDB_RESUME_ID = os.environ.get("WANDB_RESUME_ID", "")  # optional: wandb run id to resume

    class_weights = CLASS_WEIGHTS_DEFAULT if USE_CLASS_WEIGHTS else None

    rank, world, local_rank, device = setup_ddp()

    run_dir = f"/projects/_ssd/xrssd/OminiControl/runs/{RUN_NAME}"
    ckpt_dir = f"{run_dir}/ckpt"
    if is_main():
        os.makedirs(ckpt_dir, exist_ok=True)
    log_path = f"{run_dir}/train.jsonl"

    cfg = dict(steps=N_STEPS, res=RES, lr=LR, K=K, lambda_preserve=LAMBDA,
               accum=ACCUM, eval_every=EVAL_EVERY, ckpt_every=CKPT_EVERY,
               world_size=world, effective_batch=world * ACCUM,
               method="alignprop+focus_n_fix_ddp", base="v3.4_ckpt6k_resumed",
               reward="dino_pad20_transfix", delta_rank=8,
               use_class_weights=USE_CLASS_WEIGHTS,
               class_weights=class_weights if class_weights else "uniform",
               val_pt=VAL_PT_PATH, save_images_n=SAVE_IMAGES_N)

    if is_main():
        if wandb is not None:
            wandb_kwargs = dict(project=WANDB_PROJECT, name=RUN_NAME, config=cfg)
            if WANDB_RESUME_ID:
                wandb_kwargs.update(id=WANDB_RESUME_ID, resume="allow")
            wandb.init(**wandb_kwargs)
            print(f"  wandb: {wandb.run.url}")
        print(f"[ddp] rank {rank}/{world}, device={device}")
        print(f"[cfg] {cfg}")
        if RESUME_FROM:
            print(f"[resume] delta ckpt = {RESUME_FROM}, starting from step {START_STEP + 1}")

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
    val_samples = torch.load(VAL_PT_PATH, weights_only=False)
    if is_main():
        print(f"  train: {len(ds)}, val: {len(val_samples)}")

    pipe, delta_params = load_pipe(device)

    # Optional resume: load delta LoRA weights from prior ckpt
    if RESUME_FROM:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(RESUME_FROM)
        clean = {}
        for k, v in sd.items():
            k2 = k
            for prefix in ("base_model.model.", "transformer."):
                if k2.startswith(prefix): k2 = k2[len(prefix):]
            clean[k2] = v
        set_peft_model_state_dict(pipe.transformer, clean, adapter_name="delta")
        if is_main():
            print(f"[resume] loaded {len(clean)} delta keys from {RESUME_FROM}")

    reward_model = DinoLocalReward(DINO_CKPT, device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(delta_params, lr=LR, weight_decay=0.0)

    # Seed per rank so each rank samples different composites
    random.seed(SEED_OFFSET + rank * 1000)

    # Initial val eval (skip on resume to save ~3 min — prior ckpt already evaluated)
    eval_img_dir = f"{run_dir}/eval_images"
    if START_STEP == 0:
        if is_main(): print("\n[eval] initial (step 0)...")
        t0 = time.time()
        init_val = eval_on_val_distributed(
            pipe, val_samples, reward_model, rank, world, device,
            height=RES, width=RES, num_inference_steps=10,
            eval_noise_seeds=(42, 43, 44),
            image_out_dir=eval_img_dir, n_images_to_save=SAVE_IMAGES_N, step=0,
        )
        if is_main():
            print(f"  val_reward={init_val['val_reward_mean']:+.4f} (std {init_val['val_reward_std']:.3f}) — eval {time.time()-t0:.0f}s")
            _imgs = init_val.pop("_wandb_images", None)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": 0, "event": "eval", **{k: v for k, v in init_val.items() if isinstance(v, (int, float))}}) + "\n")
            if wandb is not None:
                wandb.log({f"eval/{k}": v for k, v in init_val.items() if isinstance(v, (int, float))}, step=0)
                if _imgs:
                    wandb.log({"eval/images": _imgs}, step=0)
    else:
        init_val = {"val_reward_mean": 0.0}  # unknown baseline on resume; summary will show N/A

    # Training loop
    if is_main():
        print(f"\n{'step':>4} {'reward':>9} {'r_std':>6} {'preserv':>11} {'loss':>9} {'peak':>6} {'t_s':>5}  n")
        print("-" * 75)
    best_val = init_val["val_reward_mean"]

    for step in range(START_STEP + 1, N_STEPS + 1):
        res = sample_from_dataset(ds)
        if res is None:
            if is_main(): print(f"  (skip step {step})")
            continue
        composite, prompt, bboxes, classes = res

        pipe.text_encoder.to(device); pipe.text_encoder_2.to(device)
        cond = Condition(composite, adapter_setting="pcb_harmonize")
        cond_data = prepare_condition_data(pipe, [cond])
        pe, pool, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=device,
                                         num_images_per_prompt=1, max_sequence_length=512)
        pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        log = alignprop_step_ddp(
            pipe, pe, pool, cond_data, bboxes, classes,
            reward_model, delta_params, world, device,
            height=RES, width=RES, num_inference_steps=10, k_grad_steps=K,
            lambda_preserve=LAMBDA, mask_dilate_px=4,
            base_seed=(SEED_OFFSET + step * ACCUM + rank * 10000),
            num_accum=ACCUM,
            class_weights=class_weights,
        )
        torch.nn.utils.clip_grad_norm_(delta_params, max_norm=1.0)
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        dt = time.time() - t0

        # Aggregate reward across ranks for logging
        rew_gather = all_gather_scalars(log["reward"], world, device)
        gl_reward = float(np.mean(rew_gather))

        if is_main():
            print(f"{step:>4} {gl_reward:>+9.4f} {log.get('reward_stdev',0):>6.3f} "
                  f"{log['preserv']:>11.3e} {log['loss']:>+9.4f} {peak:>6.1f} {dt:>5.1f}  {len(bboxes)}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "event": "train",
                                    "reward": gl_reward, "reward_local": log["reward"],
                                    "preserv": log["preserv"], "loss": log["loss"],
                                    "n_comp": len(bboxes), "peak_gb": peak, "time_s": dt}) + "\n")
            if wandb is not None:
                wandb.log({
                    "train/reward": gl_reward,
                    "train/reward_local": log["reward"],
                    "train/reward_stdev": log.get("reward_stdev", 0),
                    "train/preserv": log["preserv"],
                    "train/loss": log["loss"],
                    "train/n_comp": len(bboxes),
                    "train/peak_gb": peak, "train/time_s": dt,
                }, step=step)

        if step % EVAL_EVERY == 0:
            if is_main(): print(f"\n[eval] step {step}...")
            t0 = time.time()
            val = eval_on_val_distributed(
                pipe, val_samples, reward_model, rank, world, device,
                height=RES, width=RES, num_inference_steps=10,
                eval_noise_seeds=(42, 43, 44),
                image_out_dir=eval_img_dir, n_images_to_save=SAVE_IMAGES_N, step=step,
            )
            if is_main():
                print(f"  val_reward={val['val_reward_mean']:+.4f} (std {val['val_reward_std']:.3f}) — eval {time.time()-t0:.0f}s")
                _imgs = val.pop("_wandb_images", None)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "event": "eval", **{k: v for k, v in val.items() if isinstance(v, (int, float))}}) + "\n")
                if wandb is not None:
                    wandb.log({f"eval/{k}": v for k, v in val.items() if isinstance(v, (int, float))}, step=step)
                    if _imgs:
                        wandb.log({"eval/images": _imgs}, step=step)
                if val["val_reward_mean"] > best_val:
                    best_val = val["val_reward_mean"]
                    save_delta_lora(pipe, f"{ckpt_dir}/best.safetensors")
                    print(f"  ↑ new best val_reward={best_val:+.4f}, saved ckpt")

        if step % CKPT_EVERY == 0 and is_main():
            save_delta_lora(pipe, f"{ckpt_dir}/step_{step}.safetensors")

        if dist.is_initialized():
            dist.barrier()

    # Final save + eval
    if is_main(): save_delta_lora(pipe, f"{ckpt_dir}/final.safetensors")
    if is_main(): print(f"\n[final eval]")
    val = eval_on_val_distributed(
        pipe, val_samples, reward_model, rank, world, device,
        height=RES, width=RES, num_inference_steps=10,
        eval_noise_seeds=(42, 43, 44),
        image_out_dir=eval_img_dir, n_images_to_save=SAVE_IMAGES_N, step=N_STEPS,
    )
    if is_main():
        print(f"  val_reward={val['val_reward_mean']:+.4f}")
        _imgs = val.pop("_wandb_images", None)
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": N_STEPS, "event": "final_eval", **{k: v for k, v in val.items() if isinstance(v, (int, float))}}) + "\n")
        if wandb is not None and _imgs:
            wandb.log({"eval/images": _imgs}, step=N_STEPS)
        print(f"\n[summary]")
        print(f"  initial val_reward: {init_val['val_reward_mean']:+.4f}")
        print(f"  final   val_reward: {val['val_reward_mean']:+.4f}")
        print(f"  best    val_reward: {best_val:+.4f}")
        print(f"  improvement:        {val['val_reward_mean'] - init_val['val_reward_mean']:+.4f}")
        if wandb is not None:
            wandb.summary["improvement"] = val["val_reward_mean"] - init_val["val_reward_mean"]
            wandb.summary["best_val_reward"] = best_val
            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
