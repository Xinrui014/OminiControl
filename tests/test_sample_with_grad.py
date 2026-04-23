"""Phase 1d: verify flux_sample_with_grad mechanics + memory.

Memory optimization: encode prompt + conditions once, CPU-offload text encoders.
"""
import os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig
from omini.pipeline.flux_omini import Condition
from omini.train_flux.flux_sample_with_grad import flux_sample_with_grad, prepare_condition_data

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
DEVICE = "cuda"


def load_pipe_with_lora():
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    # Enable gradient checkpointing — requires train() mode (see _should_ckpt in flux_omini)
    pipe.transformer.train()
    pipe.transformer.gradient_checkpointing = True
    pipe.transformer.gc_stride_double = 1
    pipe.transformer.gc_stride_single = 2

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
    print(f"  pipe loaded in {time.time()-t0:.1f}s, {sum(p.numel() for _,p in delta_params)/1e6:.2f}M delta params")
    return pipe, delta_params


def run_one(pipe, delta_params, prompt_embeds, pooled, cond_data, H, W, T, K, tag):
    print(f"\n{'='*60}\nRun {tag}: H={H}, W={W}, T={T}, K={K}\n{'='*60}")
    for _, p in delta_params:
        p.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_start = torch.cuda.memory_allocated() / 1e9

    t1 = time.time()
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    image = flux_sample_with_grad(
        pipe,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
        condition_data=cond_data,
        main_adapter="delta",
        height=H, width=W,
        num_inference_steps=T, k_grad_steps=K,
        generator=gen,
        vae_checkpoint=True,
    )
    peak_fwd = torch.cuda.max_memory_allocated() / 1e9
    dur_fwd = time.time() - t1
    print(f"[fwd] {dur_fwd:.1f}s, VRAM start={mem_start:.2f} peak={peak_fwd:.2f} GB")
    print(f"      shape={tuple(image.shape)}, min={image.min().item():.3f}, max={image.max().item():.3f}, NaN={image.isnan().any().item()}")

    t2 = time.time()
    loss = image.mean()
    loss.backward()
    peak_bwd = torch.cuda.max_memory_allocated() / 1e9
    dur_bwd = time.time() - t2
    print(f"[bwd] {dur_bwd:.1f}s, peak VRAM {peak_bwd:.2f} GB, loss={loss.item():.6f}")

    nonzero, zero, none_ = 0, 0, 0
    gm = []
    for n, p in delta_params:
        if p.grad is None: none_ += 1
        elif p.grad.abs().mean().item() < 1e-12: zero += 1
        else: nonzero += 1; gm.append(p.grad.abs().mean().item())
    print(f"[grad] nonzero={nonzero}, zero={zero}, None={none_}")
    if gm: print(f"       mean |grad|={np.mean(gm):.3e}, max={np.max(gm):.3e}")

    ok = (tuple(image.shape)==(1,3,H,W)
          and not image.isnan().any().item() and not image.isinf().any().item()
          and image.requires_grad and nonzero >= len(delta_params)//2)
    print(f"{tag} result: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("="*60)
    print("Phase 1d v3: pre-encode conditions + offload")
    print("="*60)
    pipe, delta_params = load_pipe_with_lora()

    # Build dummy composites at each resolution (VAE down-samples by 8x)
    np.random.seed(0)
    arr512  = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    arr1024 = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
    cond_512  = Condition(Image.fromarray(arr512),  adapter_setting="delta")
    cond_1024 = Condition(Image.fromarray(arr1024), adapter_setting="delta")

    # Pre-encode BOTH conditions + prompt while text encoders still on GPU
    print("\nPre-encoding prompt + conditions...")
    cond_data_512  = prepare_condition_data(pipe, [cond_512])
    cond_data_1024 = prepare_condition_data(pipe, [cond_1024])
    prompt_embeds, pooled, text_ids = pipe.encode_prompt(
        prompt="a pcb board", prompt_2=None, device=DEVICE,
        num_images_per_prompt=1, max_sequence_length=512,
    )
    mem_before = torch.cuda.memory_allocated() / 1e9
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    mem_after = torch.cuda.memory_allocated() / 1e9
    print(f"  after offload: {mem_before:.2f} -> {mem_after:.2f} GB (freed {mem_before-mem_after:.2f})")

    ok_A = run_one(pipe, delta_params, prompt_embeds, pooled, cond_data_512,  512, 512, 10, 3, "A (512)")
    torch.cuda.empty_cache()
    if ok_A:
        ok_B = run_one(pipe, delta_params, prompt_embeds, pooled, cond_data_1024, 1024, 1024, 10, 3, "B (1024)")
    else:
        ok_B = False
        print("Skipping B (1024) because A failed")

    print(f"\n{'='*60}\n  A (512):  {'PASS' if ok_A else 'FAIL'}\n  B (1024): {'PASS' if ok_B else 'FAIL'}")
    print(f"  {'[READY] Phase 2' if (ok_A and ok_B) else '[BLOCKED]'}")
    sys.exit(0 if (ok_A and ok_B) else 1)


if __name__ == "__main__":
    main()
