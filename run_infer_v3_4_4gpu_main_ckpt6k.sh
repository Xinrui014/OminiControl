#!/bin/bash
# Launch 4-GPU parallel inference for v3.4 MAIN ckpt6k on pro6000.
set -e

CKPT="/projects/_ssd/xrssd/OminiControl/runs/v3.4_refined_1024/20260420-104031/ckpt/6000"
EVAL_JSON="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json"
OUT_DIR="/projects/_ssd/xrssd/OminiControl/runs/v3.4_refined_1024/eval_full/ckpt6k"
LOG_DIR="/projects/_ssd/xrssd/logs/infer_v3.4_main_ckpt6k"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd /projects/_ssd/xrssd/OminiControl

echo "=== v3.4 main ckpt6k inference ==="
echo "Checkpoint: $CKPT"
echo "Output:     $OUT_DIR"

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
