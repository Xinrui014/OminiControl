#!/bin/bash
# Launch 4-GPU parallel inference for v3.2 on the held pro6000 node.
# Usage: bash run_infer_v3_2_4gpu.sh <ckpt_step>
# Example: bash run_infer_v3_2_4gpu.sh 6000

set -e

STEP=${1:-6000}
RUN_DIR="/projects/_ssd/xrssd/OminiControl/runs/v4_subclass_1024/20260415-104945"
CKPT="$RUN_DIR/ckpt/$STEP"
EVAL_JSON="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json"

# Compute short step label (e.g. 6000 -> 6k)
STEP_LABEL=$((STEP / 1000))k
OUT_DIR="$RUN_DIR/eval_full/ckpt${STEP_LABEL}"

echo "=== v3.2 inference ==="
echo "Checkpoint: $CKPT"
echo "Output:     $OUT_DIR"

if [ ! -d "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

# Split 2186 patches across 4 GPUs
# 2186 / 4 = 547 (round up to 547)
# GPU 0: [0, 547)
# GPU 1: [547, 1094)
# GPU 2: [1094, 1641)
# GPU 3: [1641, 2186)

mkdir -p "$OUT_DIR"
cd /projects/_ssd/xrssd/OminiControl

LOG_DIR="/projects/_ssd/xrssd/logs/infer_v3.2_ckpt${STEP_LABEL}"
mkdir -p "$LOG_DIR"

for i in 0 1 2 3; do
    START=$((i * 547))
    END=$(( (i + 1) * 547 ))
    if [ $END -gt 2186 ]; then END=2186; fi

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
echo "All 4 processes launched. Logs: $LOG_DIR/gpu{0,1,2,3}.log"
echo "Monitor with: tail -f $LOG_DIR/gpu0.log"
