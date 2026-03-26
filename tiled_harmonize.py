#!/usr/bin/env python3
"""
Tiled PCB Harmonization for 1280×720 boards.

Takes a composited image (white canvas + pasted components) at 1280×720,
splits into overlapping 512×512 tiles, runs OminiControl harmonization
on each tile, then stitches with linear blending in overlap regions.

Usage:
    CUDA_VISIBLE_DEVICES=2 python tiled_harmonize.py \
        --input composite.png \
        --output harmonized.png \
        --checkpoint runs/20260305-001915/ckpt/24000
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline

# OminiControl imports
OMINI_ROOT = "/home/xinrui/projects/OminiControl"
sys.path.insert(0, OMINI_ROOT)
from omini.pipeline.flux_omini import Condition, generate, seed_everything

# ---------------------------------------------------------------------------
# Tiling utilities
# ---------------------------------------------------------------------------

def compute_tile_positions(total: int, tile_size: int, min_overlap: int = 64) -> List[int]:
    """
    Compute 1D tile start positions to cover `total` pixels with `tile_size` tiles
    and at least `min_overlap` overlap between adjacent tiles.
    
    Returns list of start positions.
    """
    if total <= tile_size:
        return [0]
    
    # Number of tiles needed
    # With n tiles: covered = n * tile_size - (n-1) * overlap >= total
    # So overlap <= (n * tile_size - total) / (n - 1)
    # We want overlap >= min_overlap, so:
    # n >= (total - min_overlap) / (tile_size - min_overlap)
    n = 1
    while True:
        if n == 1:
            if tile_size >= total:
                break
            n += 1
            continue
        overlap = (n * tile_size - total) / (n - 1)
        if overlap >= min_overlap:
            break
        n += 1
    
    if n == 1:
        return [0]
    
    # Distribute tiles evenly
    stride = (total - tile_size) / (n - 1)
    positions = [round(i * stride) for i in range(n)]
    
    # Ensure last tile ends exactly at total
    positions[-1] = total - tile_size
    
    return positions


def compute_tile_grid(
    width: int, height: int, tile_size: int = 512, min_overlap: int = 64
) -> List[Tuple[int, int]]:
    """
    Returns list of (x, y) top-left positions for tiles covering the image.
    """
    x_positions = compute_tile_positions(width, tile_size, min_overlap)
    y_positions = compute_tile_positions(height, tile_size, min_overlap)
    
    grid = []
    for y in y_positions:
        for x in x_positions:
            grid.append((x, y))
    
    return grid


def create_blend_weights(
    width: int,
    height: int,
    tile_size: int,
    tiles: List[Tuple[int, int]],
) -> List[np.ndarray]:
    """
    Create per-tile blend weight maps. In overlap regions, weights ramp
    linearly so tiles blend smoothly. Each weight map is (tile_size, tile_size).
    """
    # For each tile, create a weight that ramps from 0 to 1 at borders
    # that overlap with other tiles
    weights = []
    
    for tx, ty in tiles:
        w = np.ones((tile_size, tile_size), dtype=np.float32)
        
        # For each edge, check if there's a neighboring tile
        # Left edge ramp
        for ox, oy in tiles:
            if ox == tx and oy == ty:
                continue
            
            # Horizontal overlap
            if oy == ty or (oy < ty + tile_size and oy + tile_size > ty):
                # Tile to the left
                if ox < tx and ox + tile_size > tx:
                    overlap = ox + tile_size - tx
                    ramp = np.linspace(0, 1, overlap)
                    w[:, :overlap] = np.minimum(w[:, :overlap], ramp[np.newaxis, :])
                
                # Tile to the right
                if ox > tx and tx + tile_size > ox:
                    overlap = tx + tile_size - ox
                    ramp = np.linspace(1, 0, overlap)
                    w[:, -overlap:] = np.minimum(w[:, -overlap:], ramp[np.newaxis, :])
            
            # Vertical overlap
            if tx == ox or (ox < tx + tile_size and ox + tile_size > tx):
                # Tile above
                if oy < ty and oy + tile_size > ty:
                    overlap = oy + tile_size - ty
                    ramp = np.linspace(0, 1, overlap)
                    w[:overlap, :] = np.minimum(w[:overlap, :], ramp[:, np.newaxis])
                
                # Tile below
                if oy > ty and ty + tile_size > oy:
                    overlap = ty + tile_size - oy
                    ramp = np.linspace(1, 0, overlap)
                    w[-overlap:, :] = np.minimum(w[-overlap:, :], ramp[:, np.newaxis])
        
        weights.append(w)
    
    return weights


def stitch_tiles(
    width: int,
    height: int,
    tile_size: int,
    tiles: List[Tuple[int, int]],
    tile_images: List[np.ndarray],
) -> np.ndarray:
    """
    Stitch tile images back together with linear blending in overlaps.
    
    Args:
        tile_images: list of (tile_size, tile_size, 3) uint8 arrays
    Returns:
        (height, width, 3) uint8 array
    """
    weights = create_blend_weights(width, height, tile_size, tiles)
    
    canvas = np.zeros((height, width, 3), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    
    for (tx, ty), tile_img, w in zip(tiles, tile_images, weights):
        tile_float = tile_img.astype(np.float64)
        
        for c in range(3):
            canvas[ty:ty+tile_size, tx:tx+tile_size, c] += tile_float[:, :, c] * w
        weight_sum[ty:ty+tile_size, tx:tx+tile_size] += w
    
    # Avoid division by zero
    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(3):
        canvas[:, :, c] /= weight_sum
    
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda") -> FluxPipeline:
    """Load FLUX.1-dev + LoRA checkpoint."""
    print(f"Loading FLUX.1-dev base model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    print(f"Loading LoRA: {checkpoint_path}")
    pipe.load_lora_weights(
        checkpoint_path,
        weight_name="default.safetensors",
        adapter_name="pcb_harmonize",
    )
    pipe.set_adapters(["pcb_harmonize"])
    print("Model ready.")
    return pipe


# ---------------------------------------------------------------------------
# Tiled harmonization
# ---------------------------------------------------------------------------

def tiled_harmonize(
    pipe: FluxPipeline,
    composite_img: Image.Image,
    prompt: str,
    tile_size: int = 512,
    min_overlap: int = 128,
    seed: int = 42,
) -> Image.Image:
    """
    Run tiled harmonization on a large composite image.
    
    1. Compute tile grid with overlap
    2. For each tile: crop condition patch → run OminiControl
    3. Stitch with linear blending
    """
    w, h = composite_img.size
    print(f"\nTiled harmonization: {w}×{h} → {tile_size}×{tile_size} tiles, min_overlap={min_overlap}")
    
    tiles = compute_tile_grid(w, h, tile_size, min_overlap)
    print(f"Tile grid: {len(tiles)} tiles")
    for i, (tx, ty) in enumerate(tiles):
        print(f"  Tile {i}: ({tx}, {ty}) → ({tx+tile_size}, {ty+tile_size})")
    
    tile_results = []
    
    for i, (tx, ty) in enumerate(tiles):
        print(f"\n--- Tile {i+1}/{len(tiles)} at ({tx}, {ty}) ---")
        
        # Crop the condition patch from the composite
        condition_crop = composite_img.crop((tx, ty, tx + tile_size, ty + tile_size))
        
        # Create condition
        condition = Condition(condition_crop, "pcb_harmonize")
        
        # Generate
        seed_everything(seed + i)
        result = generate(
            pipe,
            prompt=prompt,
            conditions=[condition],
            height=tile_size,
            width=tile_size,
        ).images[0]
        
        tile_results.append(np.array(result))
        print(f"  ✓ Tile {i+1} done")
    
    # Stitch
    print(f"\nStitching {len(tile_results)} tiles...")
    stitched = stitch_tiles(w, h, tile_size, tiles, tile_results)
    
    return Image.fromarray(stitched)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tiled PCB Harmonization (1280×720)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input composite image (1280×720 or any size)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Process all images in directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (single image mode)")
    parser.add_argument("--output_dir", type=str, default="./tiled_output",
                        help="Output directory (batch mode)")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/xinrui/projects/OminiControl/runs/20260305-001915/ckpt/24000",
                        help="LoRA checkpoint path")
    parser.add_argument("--prompt", type=str,
                        default="A high-quality photograph of a printed circuit board with green soldermask, copper traces, and electronic components",
                        help="Generation prompt")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--min_overlap", type=int, default=128,
                        help="Minimum overlap between tiles in pixels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_tiles", action="store_true",
                        help="Also save individual tile results")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save side-by-side comparison (input | output)")
    
    args = parser.parse_args()
    
    # Determine input files
    if args.input_dir:
        input_dir = Path(args.input_dir)
        input_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
        if not input_files:
            print(f"No images found in {input_dir}")
            return
    else:
        input_files = [Path(args.input)]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model once
    pipe = load_model(args.checkpoint, device=args.device)
    
    for img_path in input_files:
        print(f"\n{'='*70}")
        print(f"Processing: {img_path}")
        print(f"{'='*70}")
        
        composite = Image.open(img_path).convert("RGB")
        w, h = composite.size
        print(f"Input size: {w}×{h}")
        
        # If image is already tile_size or smaller, just run directly
        if w <= args.tile_size and h <= args.tile_size:
            print("Image fits in single tile, running directly...")
            condition = Condition(composite, "pcb_harmonize")
            seed_everything(args.seed)
            result = generate(
                pipe, prompt=args.prompt, conditions=[condition],
                height=h, width=w,
            ).images[0]
        else:
            result = tiled_harmonize(
                pipe, composite, args.prompt,
                tile_size=args.tile_size,
                min_overlap=args.min_overlap,
                seed=args.seed,
            )
        
        # Save output
        if args.output and len(input_files) == 1:
            out_path = Path(args.output)
        else:
            out_path = output_dir / f"{img_path.stem}_harmonized.png"
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        print(f"✓ Saved: {out_path}")
        
        # Save comparison
        if args.save_comparison:
            comp_img = Image.new("RGB", (w * 2, h))
            comp_img.paste(composite, (0, 0))
            comp_img.paste(result, (w, 0))
            comp_path = output_dir / f"{img_path.stem}_comparison.png"
            comp_img.save(comp_path)
            print(f"  Comparison: {comp_path}")
        
        # Save individual tiles
        if args.save_tiles and (w > args.tile_size or h > args.tile_size):
            tiles = compute_tile_grid(w, h, args.tile_size, args.min_overlap)
            tile_dir = output_dir / f"{img_path.stem}_tiles"
            tile_dir.mkdir(exist_ok=True)
            for i, (tx, ty) in enumerate(tiles):
                crop = composite.crop((tx, ty, tx + args.tile_size, ty + args.tile_size))
                crop.save(tile_dir / f"condition_{i:02d}_{tx}_{ty}.png")
            print(f"  Tile conditions saved to: {tile_dir}")
    
    print(f"\n✅ All done! Output in: {output_dir}")


if __name__ == "__main__":
    main()
