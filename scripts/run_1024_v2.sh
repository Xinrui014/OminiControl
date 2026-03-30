#!/bin/bash
cd /home/xinrui/projects/OminiControl

PYTHON="/home/xinrui/miniconda3/envs/omini/bin/python"
CKPT="runs/v2_pcb_harmonize/ckpt/12000"
OUT="output_1024/v2_12k"
EVAL_JSON="config/eval_patches_small_components.json"

CUDA_VISIBLE_DEVICES=0 $PYTHON infer_1024.py --eval_json $EVAL_JSON --omini_ckpt $CKPT --output_dir $OUT --start 0  --end 8  --seed 42 &
CUDA_VISIBLE_DEVICES=1 $PYTHON infer_1024.py --eval_json $EVAL_JSON --omini_ckpt $CKPT --output_dir $OUT --start 8  --end 16 --seed 42 &
CUDA_VISIBLE_DEVICES=2 $PYTHON infer_1024.py --eval_json $EVAL_JSON --omini_ckpt $CKPT --output_dir $OUT --start 16 --end 24 --seed 42 &
CUDA_VISIBLE_DEVICES=3 $PYTHON infer_1024.py --eval_json $EVAL_JSON --omini_ckpt $CKPT --output_dir $OUT --start 24 --end 30 --seed 42 &

wait
echo "All done at $(date)"
