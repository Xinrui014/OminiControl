#!/bin/bash
# Launch 4-GPU parallel inference for v3.4 resumed ckpt (14k equiv) on pro6000.
set -e

CKPT="/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000"
EVAL_JSON="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json"
OUT_DIR="/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/eval_full/ckpt14k"
LOG_DIR="/projects/_ssd/xrssd/logs/infer_v3.4_ckpt14k"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd /projects/_ssd/xrssd/OminiControl

echo "=== v3.4 ckpt14k inference ==="
echo "Checkpoint: $CKPT"
echo "Output:     $OUT_DIR"

# 2186 patches / 4 GPUs = 547 per GPU (last gets 545)
# GPU 0: [0, 547)
# GPU 1: [547, 1094)
# GPU 2: [1094, 1641)
# GPU 3: [1641, 2186)
for i in 0 1 2 3; do
    START=$((i * 547))
    END=$(( (i + 1) * 547 ))
    if [ $END -gt 2186 ]; then END=2186; fi
    echo "GPU $i: patches [$START, $END)"
    CUDA_VISIBLE_DEVICES=$i python infer_v3_3_full_eval.py \
        --eval_json "$EVAL_JSON" \
        --omini_ckpt "$CKPT" \
        --output_dir "$OUT_DIR" \
        --start $START --end $END \
        > "$LOG_DIR/gpu${i}.log" 2>&1 &
    echo "  -> PID $!"
done
wait
echo "=== All 4 GPUs finished ==="
