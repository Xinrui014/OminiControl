#!/bin/bash
# Run fixed eval on v2 and v2.1 checkpoints for comparison.
#
# Usage:
#   bash scripts/run_fixed_eval.sh              # sequential on GPU 0
#   bash scripts/run_fixed_eval.sh --parallel   # parallel across GPUs

PYTHON="/home/xinrui/miniconda3/envs/omini/bin/python"
EVAL_JSON="config/eval_patches_small.json"
OUTPUT_BASE="output_fixed_eval"
PARALLEL=false

if [[ "$1" == "--parallel" ]]; then
    PARALLEL=true
fi

declare -A CHECKPOINTS
CHECKPOINTS=(
    ["v2_12k"]="runs/v2_pcb_harmonize/ckpt/12000"
    ["v2.1_6k"]="runs/v2.1_pcb_harmonize/ckpt/6000"
    ["v2.1_12k"]="runs/v2.1_pcb_harmonize/ckpt/12000"
    ["v2.1_20k"]="runs/v2.1_pcb_harmonize/ckpt/20000"
)

GPUS=(0 1 2 3)

run_eval() {
    local NAME=$1
    local CKPT=$2
    local GPU=$3
    local OUT="${OUTPUT_BASE}/${NAME}"

    if [ ! -d "$CKPT" ]; then
        echo "[GPU $GPU] Checkpoint $CKPT not found, skipping"
        return
    fi

    echo "[GPU $GPU] Running fixed eval: $NAME → $OUT"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON infer_fixed_eval.py \
        --eval_json "$EVAL_JSON" \
        --omini_ckpt "$CKPT" \
        --output_dir "$OUT" \
        --seed 42 \
        --device cuda
    echo "[GPU $GPU] Done: $OUT/gallery.html"
}

# Sort keys for consistent ordering
NAMES=($(echo "${!CHECKPOINTS[@]}" | tr ' ' '\n' | sort))

if $PARALLEL; then
    echo "Running ${#NAMES[@]} checkpoints in parallel..."
    PIDS=()
    for i in "${!NAMES[@]}"; do
        NAME="${NAMES[$i]}"
        GPU=${GPUS[$i % ${#GPUS[@]}]}
        run_eval "$NAME" "${CHECKPOINTS[$NAME]}" "$GPU" &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
else
    echo "Running ${#NAMES[@]} checkpoints sequentially on GPU 0..."
    for NAME in "${NAMES[@]}"; do
        run_eval "$NAME" "${CHECKPOINTS[$NAME]}" 0
    done
fi

echo ""
echo "All fixed eval runs complete at $(date)"
echo "Results in: ${OUTPUT_BASE}/"
