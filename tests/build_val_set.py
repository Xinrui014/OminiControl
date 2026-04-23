"""Build fixed validation set for AlignProp eval tracking.

Picks N composites from v2.2 val split deterministically and saves to disk.
Each val sample has composite PIL, prompt, bboxes, classes — reused every eval.
"""
import os, sys, random
import torch
from pathlib import Path

sys.path.insert(0, "/projects/_ssd/xrssd/OminiControl")
from train_pcb_v3_4 import PCBHarmonizeDatasetV3_4
from lib.component_bank_v2_1 import ComponentBankV2_1

DATA_DIR = "/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass"
OUT = "/projects/_ssd/xrssd/rewards/alignprop_val_20.pt"

PCB_TO_DINO = {"RESISTOR":0,"CAPACITOR":1,"INDUCTOR":2,"CONNECTOR":3,
               "DIODE":4,"SWITCH":5,"TRANSISTOR":6,"IC":7,"OSCILLATOR":8}


def main():
    N_VAL = int(os.environ.get("N_VAL", "20"))
    MIN_COMP = 5
    MAX_COMP = 15

    # Use VAL split if it exists, else train
    for split in ["val", "train"]:
        anno = os.path.join(DATA_DIR, f"annotation/{split}")
        img  = os.path.join(DATA_DIR, f"image/{split}")
        if os.path.isdir(anno):
            print(f"using split='{split}'")
            break
    else:
        raise FileNotFoundError("no val or train dir found")

    bank = ComponentBankV2_1(anno_dir=anno, image_dir=img)
    ds = PCBHarmonizeDatasetV3_4(
        anno_dir=anno, image_dir=img,
        condition_size=(1024, 1024), target_size=(1024, 1024),
        component_bank=bank,
        zoom_prob=0.4, zoom_crop_size=256,
        drop_text_prob=0.0, drop_image_prob=0.0,
        return_annotations=True,
    )
    print(f"  dataset size: {len(ds)}")

    random.seed(42)
    samples = []
    max_tries = 10 * N_VAL
    tries = 0
    seen_idx = set()
    while len(samples) < N_VAL and tries < max_tries:
        tries += 1
        idx = random.randrange(len(ds))
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        item = ds[idx]
        bboxes, classes = [], []
        for bb, nm in zip(item["bboxes_xyxy"], item["cat_names"]):
            ci = PCB_TO_DINO.get(nm.upper())
            if ci is not None:
                bboxes.append(bb); classes.append(ci)
        if MIN_COMP <= len(bboxes) <= MAX_COMP:
            samples.append({
                "idx": idx,
                "composite": item["composite_pil"],
                "prompt": item["prompt"],
                "bboxes": bboxes,
                "classes": classes,
            })
            print(f"  [{len(samples)}/{N_VAL}] ds[{idx}]: {len(bboxes)} comps, classes={classes}")

    print(f"\nkept {len(samples)} samples after {tries} tries")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(samples, OUT)
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
