"""Build class-balanced validation set for AlignProp eval.

Ensures each DINO class (0-8) appears at least MIN_PER_CLASS times across val
samples. Uses larger candidate pool then filters to balance classes.
"""
import os, sys, random
from collections import Counter
import torch
from pathlib import Path

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")
from train_pcb_v3_4 import PCBHarmonizeDatasetV3_4
from lib.component_bank_v2_1 import ComponentBankV2_1

DATA_DIR = "/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass"
OUT = "/projects/_ssd/xrssd/rewards/alignprop_val_balanced.pt"

PCB_TO_DINO = {"RESISTOR":0,"CAPACITOR":1,"INDUCTOR":2,"CONNECTOR":3,
               "DIODE":4,"SWITCH":5,"TRANSISTOR":6,"IC":7,"OSCILLATOR":8}
DINO_NAMES = ["Resistor","Capacitor","Inductor","Connector",
              "Diode","Switch","Transistor","IC","Oscillator"]


def extract_samples(item):
    """Filter annotations to DINO classes."""
    bboxes, classes = [], []
    for bb, nm in zip(item["bboxes_xyxy"], item["cat_names"]):
        ci = PCB_TO_DINO.get(nm.upper())
        if ci is not None:
            bboxes.append(bb); classes.append(ci)
    return bboxes, classes


def main():
    N_VAL_TARGET = int(os.environ.get("N_VAL", "30"))
    MIN_PER_CLASS = int(os.environ.get("MIN_PER_CLASS", "3"))
    MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "500"))

    anno = os.path.join(DATA_DIR, "annotation/train")
    img = os.path.join(DATA_DIR, "image/train")
    bank = ComponentBankV2_1(anno_dir=anno, image_dir=img)
    ds = PCBHarmonizeDatasetV3_4(
        anno_dir=anno, image_dir=img,
        condition_size=(1024, 1024), target_size=(1024, 1024),
        component_bank=bank, zoom_prob=0.4, zoom_crop_size=256,
        drop_text_prob=0.0, drop_image_prob=0.0,
        return_annotations=True,
    )
    print(f"dataset size: {len(ds)}")

    random.seed(42)

    # Phase 1: collect candidates, prioritizing those with rare classes
    candidates = []   # list of (idx, item, bboxes, classes, rare_score)
    seen_idx = set()
    tries = 0

    while len(candidates) < MAX_CANDIDATES and tries < MAX_CANDIDATES * 5:
        tries += 1
        idx = random.randrange(len(ds))
        if idx in seen_idx: continue
        seen_idx.add(idx)
        item = ds[idx]
        bboxes, classes = extract_samples(item)
        if len(bboxes) < 3 or len(bboxes) > 20:
            continue
        candidates.append({
            "idx": idx,
            "composite": item["composite_pil"],
            "prompt": item["prompt"],
            "bboxes": bboxes,
            "classes": classes,
        })

    print(f"collected {len(candidates)} candidates from {tries} tries")

    # Phase 2: greedy select to balance classes
    # Sort candidates by how many rare classes they contain
    selected = []
    class_count = Counter()

    # First pass: pick any candidate containing a rare class (Switch=5, Trans=6, Osc=8, Ind=2)
    RARE = {5, 6, 8, 2}
    candidates_by_rare = sorted(
        candidates,
        key=lambda c: -sum(1 for cls in c["classes"] if cls in RARE)
    )

    # Greedy: pick samples that most help under-represented classes
    target_count = {ci: MIN_PER_CLASS for ci in range(9)}

    def gain(sample_classes, class_count):
        g = 0
        for cls in sample_classes:
            need = target_count[cls] - class_count[cls]
            if need > 0:
                g += 1
        return g

    remaining = candidates_by_rare.copy()
    while len(selected) < N_VAL_TARGET and remaining:
        # Pick candidate that most adds to under-covered classes
        best = max(remaining, key=lambda c: gain(c["classes"], class_count))
        best_gain = gain(best["classes"], class_count)

        selected.append(best)
        for cls in best["classes"]:
            class_count[cls] += 1
        remaining.remove(best)

        # Stop adding once all classes covered AND we have enough samples
        if all(class_count[ci] >= MIN_PER_CLASS for ci in range(9)) \
           and len(selected) >= N_VAL_TARGET:
            break
        if best_gain == 0 and len(selected) >= N_VAL_TARGET:
            break

    print(f"\nselected {len(selected)} samples")
    print(f"class coverage (target ≥{MIN_PER_CLASS}):")
    for ci in range(9):
        n = class_count[ci]
        tag = "✓" if n >= MIN_PER_CLASS else "⚠"
        print(f"  {DINO_NAMES[ci]:<12} {n:>3}  {tag}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(selected, OUT)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
