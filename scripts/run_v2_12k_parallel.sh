#!/bin/bash
cd /home/xinrui/projects/OminiControl

PYTHON="/home/xinrui/miniconda3/envs/omini/bin/python"
CKPT="runs/v2_pcb_harmonize/ckpt/12000"
OUT="output_fixed_eval/v2_12k"
SCRIPT="infer_fixed_eval.py"

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --omini_ckpt $CKPT --output_dir $OUT --start 0  --end 8  --seed 42 &
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT --omini_ckpt $CKPT --output_dir $OUT --start 8  --end 16 --seed 42 &
CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT --omini_ckpt $CKPT --output_dir $OUT --start 16 --end 24 --seed 42 &
CUDA_VISIBLE_DEVICES=3 $PYTHON $SCRIPT --omini_ckpt $CKPT --output_dir $OUT --start 24 --end 30 --seed 42 &

wait
echo "All 4 GPUs done at $(date)"
