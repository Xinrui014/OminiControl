#!/usr/bin/env python3
"""
Compute FID and KID between real crops and generated (harmonized) images.
Uses clean-fid for reliable computation.

Usage:
    python eval_metrics/compute_fid.py \
        --real_dir eval_metrics/real_crops \
        --gen_dir runs/v3_newdata/eval_full/ckpt14k \
        --gen_suffix _harmonized.png \
        --output runs/v3_newdata/eval_full/ckpt14k/fid_kid_metrics.json

Note: install clean-fid first: pip install clean-fid
"""
import json
import argparse
import os
import shutil
import tempfile
from pathlib import Path


def collect_gen_images(gen_dir, gen_suffix, tmp_dir):
    """Copy generated images to a flat temp dir (clean-fid expects a dir of images).
    Only copies files matching the suffix, renamed to match real crop filenames."""
    gen_dir = Path(gen_dir)
    count = 0
    for f in gen_dir.iterdir():
        if f.name.endswith(gen_suffix):
            # e.g. patch_id_harmonized.png -> patch_id.png
            clean_name = f.name.replace(gen_suffix, ".png")
            shutil.copy2(f, Path(tmp_dir) / clean_name)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True, help="Dir of real crop PNGs")
    parser.add_argument("--gen_dir", required=True, help="Dir containing generated images")
    parser.add_argument("--gen_suffix", default="_harmonized.png", help="Suffix to filter generated images")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Verify real dir
    real_dir = Path(args.real_dir)
    n_real = len([f for f in real_dir.iterdir() if f.suffix == ".png"])
    print(f"Real images: {n_real} in {real_dir}")

    # Collect generated images into a temp dir (clean-fid needs a flat dir)
    tmp_gen = tempfile.mkdtemp(prefix="fid_gen_")
    try:
        n_gen = collect_gen_images(args.gen_dir, args.gen_suffix, tmp_gen)
        print(f"Generated images: {n_gen} in {args.gen_dir} (suffix: {args.gen_suffix})")

        if n_gen == 0:
            print("ERROR: No generated images found!")
            return

        # Check 1:1 correspondence
        real_names = set(f.name for f in real_dir.iterdir() if f.suffix == ".png")
        gen_names = set(f for f in os.listdir(tmp_gen) if f.endswith(".png"))
        matched = real_names & gen_names
        only_real = real_names - gen_names
        only_gen = gen_names - real_names

        print(f"Matched pairs: {len(matched)}")
        if only_real:
            print(f"  Only in real (missing gen): {len(only_real)}")
        if only_gen:
            print(f"  Only in gen (missing real): {len(only_gen)}")

        # If not all matched, filter to only matched images
        if only_real or only_gen:
            print("Filtering to matched pairs only...")
            # Remove unmatched from real (copy matched to new tmp)
            tmp_real = tempfile.mkdtemp(prefix="fid_real_")
            for name in matched:
                shutil.copy2(real_dir / name, Path(tmp_real) / name)
            # Remove unmatched from gen
            for name in only_gen:
                os.remove(Path(tmp_gen) / name)
            real_path = tmp_real
            n_matched = len(matched)
        else:
            real_path = str(real_dir)
            tmp_real = None
            n_matched = len(matched)

        print(f"\nComputing FID and KID on {n_matched} image pairs...")

        from cleanfid import fid as cleanfid

        # Compute FID
        print("Computing FID...")
        fid_score = cleanfid.compute_fid(real_path, tmp_gen, device=args.device)
        print(f"  FID: {fid_score:.4f}")

        # Compute KID
        print("Computing KID...")
        kid_score = cleanfid.compute_kid(real_path, tmp_gen, device=args.device)
        print(f"  KID: {kid_score:.6f}")

        # Save results
        results = {
            "fid": round(fid_score, 4),
            "kid": round(kid_score, 6),
            "n_real": n_matched,
            "n_gen": n_matched,
            "real_dir": str(args.real_dir),
            "gen_dir": str(args.gen_dir),
            "gen_suffix": args.gen_suffix,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_path}")
        print(json.dumps(results, indent=2))

    finally:
        # Cleanup temp dirs
        shutil.rmtree(tmp_gen, ignore_errors=True)
        if 'tmp_real' in locals() and tmp_real:
            shutil.rmtree(tmp_real, ignore_errors=True)


if __name__ == "__main__":
    main()
