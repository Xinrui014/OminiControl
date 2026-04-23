#!/bin/bash
# 8-GPU parallel inference for v3.2
# Usage: bash run_infer_v3_2_8gpu.sh <ckpt_step>
set -e

STEP=${1:-10000}
RUN_DIR="/projects/_ssd/xrssd/OminiControl/runs/v4_subclass_1024/20260415-104945"
CKPT="$RUN_DIR/ckpt/$STEP"
EVAL_JSON="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json"
STEP_LABEL=$((STEP / 1000))k
OUT_DIR="$RUN_DIR/eval_full/ckpt${STEP_LABEL}"

echo "=== v3.2 inference (8 GPU) ==="
echo "Checkpoint: $CKPT"
echo "Output:     $OUT_DIR"

if [ ! -d "$CKPT" ]; then echo "ERROR: checkpoint not found: $CKPT"; exit 1; fi

mkdir -p "$OUT_DIR"
cd /projects/_ssd/xrssd/OminiControl

LOG_DIR="/projects/_ssd/xrssd/logs/infer_v3.2_ckpt${STEP_LABEL}"
mkdir -p "$LOG_DIR"

# Split 2186 patches across 8 GPUs (~274 each)
TOTAL=2186
PER_GPU=$(( (TOTAL + 7) / 8 ))

for i in $(seq 0 7); do
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU ))
    if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
    if [ $START -ge $TOTAL ]; then break; fi

    echo "GPU $i: patches [$START, $END)"
    CUDA_VISIBLE_DEVICES=$i nohup python infer_v3_2_full_eval.py \
        --eval_json "$EVAL_JSON" \
        --omini_ckpt "$CKPT" \
        --output_dir "$OUT_DIR" \
        --start $START --end $END \
        > "$LOG_DIR/gpu${i}.log" 2>&1 &
    echo "  -> PID $!"
done

echo ""
echo "All launched. Logs: $LOG_DIR/"
