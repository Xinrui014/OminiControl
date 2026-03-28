#!/bin/bash
# PCB Harmonization v2.1 — Multi-scale crops + component-aware loss
# Fine-tune from v2 12k checkpoint
#
# Changes from v2:
#   - 40% zoom crops (256→512) for small component detail
#   - 3x loss weight on component regions
#   - Lower LR (1e-5) and shorter run (8k steps)

export OMINI_CONFIG="train/config/pcb_harmonize_v2_1.yaml"

torchrun --nproc_per_node=4 \
    -m omini.train_flux.train_pcb_v2 \
    2>&1 | tee runs/train_v2_1.log
