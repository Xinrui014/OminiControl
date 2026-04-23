#!/bin/bash
# 4-GPU parallel inference for v3.2 using ComponentBankV2_new (no sub_class needed)
# Usage: bash run_infer_v3_2_oldbank_4gpu.sh <ckpt_step>

set -e

STEP=${1:-6000}
RUN_DIR="/projects/_ssd/xrssd/OminiControl/runs/v4_subclass_1024/20260415-104945"
CKPT="$RUN_DIR/ckpt/$STEP"
EVAL_JSON="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test.json"
STEP_LABEL=$((STEP / 1000))k
OUT_DIR="$RUN_DIR/eval_full_oldbank/ckpt${STEP_LABEL}"

echo "=== v3.2 inference (oldbank) ==="
echo "Checkpoint: $CKPT"
echo "Output:     $OUT_DIR"

if [ ! -d "$CKPT" ]; then echo "ERROR: checkpoint not found: $CKPT"; exit 1; fi

mkdir -p "$OUT_DIR"
cd /projects/_ssd/xrssd/OminiControl

LOG_DIR="/projects/_ssd/xrssd/logs/infer_v3.2_oldbank_ckpt${STEP_LABEL}"
mkdir -p "$LOG_DIR"

for i in 0 1 2 3; do
    START=$((i * 547))
    END=$(( (i + 1) * 547 ))
    if [ $END -gt 2186 ]; then END=2186; fi
    echo "GPU $i: patches [$START, $END)"
    CUDA_VISIBLE_DEVICES=$i nohup python infer_v3_2_oldbank.py \
        --eval_json "$EVAL_JSON" \
        --omini_ckpt "$CKPT" \
        --output_dir "$OUT_DIR" \
        --start $START --end $END \
        > "$LOG_DIR/gpu${i}.log" 2>&1 &
    echo "  -> PID $!"
done

echo ""
echo "All 4 launched. Logs: $LOG_DIR/gpu{0,1,2,3}.log"
