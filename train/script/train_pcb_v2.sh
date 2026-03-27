#!/bin/bash
# PCB Harmonization v2 Training — 4-GPU torchrun + wandb
# On-the-fly composite pasting with color-matched components
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMINI_CONFIG=./train/config/pcb_harmonize_v2.yaml
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY="a5ebf533c17c677bcee36f66c91907b5fb102f7c"

echo "Starting PCB harmonization v2 training (4-GPU + wandb)..."
echo "Config: $OMINI_CONFIG"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Key changes: on-the-fly pasting, color-matched, resize jitter, random crops"

torchrun \
    --nproc_per_node=4 \
    --master_port=41355 \
    -m omini.train_flux.train_pcb_v2
