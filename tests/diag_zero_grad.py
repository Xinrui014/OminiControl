"""Diagnose which LoRA params get zero gradient in flux_sample_with_grad."""
import os, sys, time
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")

from diffusers import FluxPipeline
from peft import LoraConfig
from omini.pipeline.flux_omini import Condition
from omini.train_flux.flux_sample_with_grad import flux_sample_with_grad, prepare_condition_data

FLUX_PATH = "/projects/_ssd/xrssd/DiffSynth-Studio/models/black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
DEVICE = "cuda"


def main():
    pipe = FluxPipeline.from_pretrained(FLUX_PATH, torch_dtype=DTYPE).to(DEVICE)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.text_encoder_2.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    pipe.transformer.train()
    pipe.transformer.gradient_checkpointing = True
    pipe.transformer.gc_stride_double = 1
    pipe.transformer.gc_stride_single = 1  # more aggressive

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
    print(f"Total delta LoRA tensors: {len(delta_params)}")

    # Small test at 512
    np.random.seed(0)
    arr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    cond = Condition(Image.fromarray(arr), adapter_setting="delta")
    cond_data = prepare_condition_data(pipe, [cond])
    pe, pool, _ = pipe.encode_prompt(prompt="a pcb", prompt_2=None, device=DEVICE,
                                     num_images_per_prompt=1, max_sequence_length=512)
    pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    image = flux_sample_with_grad(
        pipe, prompt_embeds=pe, pooled_prompt_embeds=pool,
        condition_data=cond_data, main_adapter="delta",
        height=512, width=512, num_inference_steps=10, k_grad_steps=3,
        generator=gen, vae_checkpoint=True,
    )
    image.mean().backward()

    # Categorize params
    # Naming pattern: transformer_blocks.{i}.{attn|ff|ff_context}.{to_q/k/v/out.0|net.0.proj|net.2}.{lora_A|lora_B}.delta.weight
    # or: single_transformer_blocks.{i}.{attn|proj_mlp|proj_out}.{to_q/k/v}.{lora_A|lora_B}.delta.weight
    categories = defaultdict(lambda: {"zero": 0, "nonzero": 0, "total": 0, "grad_sum": 0.0})
    zero_examples = []
    nonzero_examples = []

    for name, p in delta_params:
        # Classify
        if "single_transformer_blocks" in name:
            block_type = "single"
        elif "transformer_blocks" in name:
            block_type = "double"
        else:
            block_type = "other"

        if "lora_A" in name:
            ab = "A"
        elif "lora_B" in name:
            ab = "B"
        else:
            ab = "?"

        if ".attn.to_q" in name: mod = "to_q"
        elif ".attn.to_k" in name: mod = "to_k"
        elif ".attn.to_v" in name: mod = "to_v"
        elif ".attn.to_out.0" in name: mod = "to_out"
        elif "ff.net.0.proj" in name: mod = "ff_0"
        elif "ff.net.2" in name: mod = "ff_2"
        elif "ff_context" in name: mod = "ff_ctx"
        elif "proj_mlp" in name: mod = "proj_mlp"
        elif "proj_out" in name: mod = "proj_out"
        else: mod = "unknown"

        key = f"{block_type}/{mod}/{ab}"
        categories[key]["total"] += 1
        if p.grad is None or p.grad.abs().mean().item() < 1e-12:
            categories[key]["zero"] += 1
            if len(zero_examples) < 5:
                zero_examples.append(name)
        else:
            categories[key]["nonzero"] += 1
            categories[key]["grad_sum"] += p.grad.abs().mean().item()
            if len(nonzero_examples) < 5:
                nonzero_examples.append(name)

    print("\n=== Per-category gradient status ===")
    print(f"{'category':<25} {'total':>6} {'zero':>6} {'nonzero':>8} {'mean|grad|':>14}")
    for k in sorted(categories.keys()):
        c = categories[k]
        m = c["grad_sum"] / max(1, c["nonzero"])
        print(f"{k:<25} {c['total']:>6} {c['zero']:>6} {c['nonzero']:>8} {m:>14.3e}")

    print("\n=== Examples of ZERO-grad params ===")
    for n in zero_examples: print(f"  {n}")
    print("\n=== Examples of NON-ZERO-grad params ===")
    for n in nonzero_examples: print(f"  {n}")


if __name__ == "__main__":
    main()
