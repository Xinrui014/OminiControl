#!/bin/bash
# PCB Harmonization v2.1 — Multi-scale crops + component-aware loss
# Train from FLUX.1-dev scratch with Prodigy optimizer
#
# Changes from v2:
#   - 40% zoom crops (256→512) for small component detail
#   - 3x loss weight on component regions
#   - Prodigy optimizer, 20k steps

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMINI_CONFIG=train/config/pcb_harmonize_v2_1.yaml
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY=a5ebf533c17c677bcee36f66c91907b5fb102f7c

echo 'Starting PCB v2.1 training from scratch...'
echo 'Config:' $OMINI_CONFIG

/home/xinrui/miniconda3/envs/omini/bin/torchrun \
    --nproc_per_node=4 \
    --master_port=41357 \
    -m omini.train_flux.train_pcb_v2
