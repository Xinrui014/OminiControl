#!/bin/bash
# PCB Harmonization v2 Fine-tune — Resume from 12k ckpt with AdamW lr=2e-5
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMINI_CONFIG=./config/config_pcb_v2_finetune.yaml
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY=a5ebf533c17c677bcee36f66c91907b5fb102f7c

echo 'Starting PCB v2 fine-tune from 12k checkpoint...'
echo 'Config:' $OMINI_CONFIG
echo 'Optimizer: AdamW lr=2e-5'
echo 'Resume from: runs/20260327-022340/ckpt/12000'

/home/xinrui/miniconda3/envs/omini/bin/torchrun     --nproc_per_node=4     --master_port=41356     -m omini.train_flux.train_pcb_v2
