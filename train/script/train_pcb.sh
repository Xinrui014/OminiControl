#!/bin/bash
# PCB Harmonization Training — 2-GPU torchrun + wandb
export CUDA_VISIBLE_DEVICES=0,1
export OMINI_CONFIG=./train/config/pcb_harmonize.yaml
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY="a5ebf533c17c677bcee36f66c91907b5fb102f7c"

echo "Starting PCB harmonization training (2-GPU + wandb)..."
echo "Config: $OMINI_CONFIG"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

torchrun \
    --nproc_per_node=2 \
    --master_port=41354 \
    -m omini.train_flux.train_pcb
