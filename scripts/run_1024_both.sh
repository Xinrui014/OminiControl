#!/bin/bash
cd /home/xinrui/projects/OminiControl

PYTHON="/home/xinrui/miniconda3/envs/omini/bin/python"
EVAL_JSON="config/eval_patches_small_components.json"

V2_CKPT="runs/v2_pcb_harmonize/ckpt/12000"
V21_CKPT="runs/v2.1_pcb_harmonize/ckpt/12000"
V2_OUT="output_1024/v2_12k"
V21_OUT="output_1024/v2.1_12k"

# v2 on GPU 0,1 — v2.1 on GPU 2,3
CUDA_VISIBLE_DEVICES=0 $PYTHON infer_1024.py --eval_json $EVAL_JSON --omini_ckpt $V2_CKPT  --output_dir $V2_OUT  --start 0  --end 15 --seed 42 &
CUDA_VISIBLE_DEVICES=1 $PYTHON infer_1024.py --eval_json $EVAL_JSON --omini_ckpt $V2_CKPT  --output_dir $V2_OUT  --start 15 --end 30 --seed 42 &
CUDA_VISIBLE_DEVICES=2 $PYTHON infer_1024.py --eval_json $EVAL_JSON --omini_ckpt $V21_CKPT --output_dir $V21_OUT --start 0  --end 15 --seed 42 &
CUDA_VISIBLE_DEVICES=3 $PYTHON infer_1024.py --eval_json $EVAL_JSON --omini_ckpt $V21_CKPT --output_dir $V21_OUT --start 15 --end 30 --seed 42 &

wait
echo "All done at $(date)"
