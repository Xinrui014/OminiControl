#!/usr/bin/env python3
"""
PCB Harmonization Inference Script
Load trained LoRA checkpoint and generate harmonized PCB images from composite inputs.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from diffusers.pipelines import FluxPipeline

# Add OminiControl to path
sys.path.insert(0, str(Path(__file__).parent))

from omini.pipeline.flux_omini import Condition, generate, seed_everything


def load_model(checkpoint_path, device="cuda"):
    """Load FLUX.1-dev base model and LoRA checkpoint."""
    print(f"Loading base model: black-forest-labs/FLUX.1-dev...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(device)
    
    print(f"Loading LoRA checkpoint from: {checkpoint_path}")
    pipe.load_lora_weights(
        checkpoint_path,
        weight_name="default.safetensors",
        adapter_name="pcb_harmonize",
    )
    pipe.set_adapters(["pcb_harmonize"])
    
    print("Model loaded successfully!")
    return pipe


def run_inference(pipe, composite_path, prompt, output_path, seed=42):
    """Run inference on a single composite image."""
    # Load composite image
    composite_img = Image.open(composite_path).convert("RGB")
    
    # Create condition
    condition = Condition(composite_img, "pcb_harmonize")
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    # Generate
    print(f"Generating harmonized image...")
    print(f"  Prompt: {prompt}")
    result_img = generate(
        pipe,
        prompt=prompt,
        conditions=[condition],
    ).images[0]
    
    # Save result
    result_img.save(output_path)
    print(f"  Saved to: {output_path}")
    
    return result_img


def main():
    parser = argparse.ArgumentParser(description="PCB Harmonization Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint directory (contains default.safetensors)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to composite manifest JSON")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Data root directory (for resolving relative paths in manifest)")
    parser.add_argument("--output_dir", type=str, default="./inference_output",
                        help="Output directory for generated images")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of samples to test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    print(f"Total samples in manifest: {len(manifest)}")
    print(f"Testing on first {args.num_samples} samples")
    
    # Load model
    pipe = load_model(args.checkpoint, device=args.device)
    
    # Run inference on samples
    data_root = Path(args.data_root)
    
    for i, sample in enumerate(manifest[:args.num_samples]):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{args.num_samples}")
        print(f"{'='*80}")
        
        # Get paths
        composite_rel = sample['composite']
        real_rel = sample['real']
        prompt = sample['prompt']
        
        composite_path = data_root / composite_rel
        real_path = data_root / real_rel
        
        # Generate output filename
        name = sample['name']
        output_path = output_dir / f"{name}_harmonized.png"
        
        # Run inference
        result_img = run_inference(
            pipe, composite_path, prompt, output_path, seed=args.seed + i
        )
        
        # Create side-by-side comparison
        composite_img = Image.open(composite_path).convert("RGB")
        real_img = Image.open(real_path).convert("RGB")
        
        w, h = composite_img.size
        concat_img = Image.new("RGB", (w * 3, h))
        concat_img.paste(composite_img, (0, 0))
        concat_img.paste(result_img, (w, 0))
        concat_img.paste(real_img, (w * 2, 0))
        
        concat_path = output_dir / f"{name}_comparison.png"
        concat_img.save(concat_path)
        print(f"  Comparison saved to: {concat_path}")
    
    print(f"\n{'='*80}")
    print("Inference complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
