#!/bin/bash
# Wait for v2.1 training to finish, then run patch-level inference
# on multiple checkpoints for comparison.
#
# Usage:
#   bash scripts/run_after_train.sh              # sequential on GPU 0
#   bash scripts/run_after_train.sh --parallel   # parallel across GPUs

TRAIN_PID=781275
RUN_DIR="runs/v2.1_pcb_harmonize"
OUTPUT_BASE="output_v2_1"
PYTHON="/home/xinrui/miniconda3/envs/omini/bin/python"
PARALLEL=false
CHECKPOINTS=(6000 12000 20000)
GPUS=(0 1 2 3)

if [[ "$1" == "--parallel" ]]; then
    PARALLEL=true
fi

echo "Waiting for training PID $TRAIN_PID to finish..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
done
echo "Training finished at $(date)"

run_infer() {
    local STEP=$1
    local GPU=$2
    local CKPT="$RUN_DIR/ckpt/$STEP"
    local OUT="${OUTPUT_BASE}/ckpt_${STEP}"

    if [ ! -d "$CKPT" ]; then
        echo "[GPU $GPU] Checkpoint $CKPT not found, skipping"
        return
    fi

    echo "[GPU $GPU] Inference on checkpoint $STEP → $OUT"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON infer_patches.py \
        --omini_ckpt "$CKPT" \
        --output_dir "$OUT" \
        --num_boards 10 \
        --patches_per_board 3 \
        --seed 42 \
        --device cuda
    echo "[GPU $GPU] Done: $OUT/gallery.html"
}

if $PARALLEL; then
    echo "Running ${#CHECKPOINTS[@]} checkpoints in parallel across GPUs..."
    PIDS=()
    for i in "${!CHECKPOINTS[@]}"; do
        GPU=${GPUS[$i % ${#GPUS[@]}]}
        run_infer "${CHECKPOINTS[$i]}" "$GPU" &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
else
    echo "Running checkpoints sequentially on GPU 0..."
    for STEP in "${CHECKPOINTS[@]}"; do
        run_infer "$STEP" 0
    done
fi

echo ""
echo "All inference complete at $(date)"
