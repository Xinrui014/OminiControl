#!/bin/bash
# Watch the training PID; when it exits, launch eval on same node.
# Args: $1 = train PID to wait for
set -e

TRAIN_PID="$1"
BEST_CKPT="/projects/_ssd/xrssd/OminiControl/runs/alignprop_prod_4gpu_500step/ckpt/best.safetensors"
FINAL_CKPT="/projects/_ssd/xrssd/OminiControl/runs/alignprop_prod_4gpu_500step/ckpt/final.safetensors"

echo "[watcher] waiting for train PID $TRAIN_PID to finish..."
while ps -p "$TRAIN_PID" > /dev/null 2>&1; do
    sleep 60
done
echo "[watcher] train PID $TRAIN_PID exited at $(date)"

# Wait a bit for any final writes to flush
sleep 30

# Pick best if available, else final
if [ -f "$BEST_CKPT" ]; then
    DELTA_CKPT="$BEST_CKPT"
    RUN_SUFFIX="best"
elif [ -f "$FINAL_CKPT" ]; then
    DELTA_CKPT="$FINAL_CKPT"
    RUN_SUFFIX="final"
else
    echo "[watcher] ERROR: no best or final ckpt found, exiting"
    exit 1
fi

echo "[watcher] using delta ckpt: $DELTA_CKPT"

# Setup env
source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit_flash
export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export TMPDIR=/projects/_ssd/xrssd/tmp

# Launch eval
export DELTA_CKPT OUT_DIR LOG_DIR
export OUT_DIR="/projects/_ssd/xrssd/OminiControl/runs/alignprop_prod_4gpu_500step/eval_full/$RUN_SUFFIX"
export LOG_DIR="/projects/_ssd/xrssd/logs/infer_alignprop_$RUN_SUFFIX"

cd /projects/_ssd/xrssd/OminiControl
echo "[watcher] launching eval at $(date)"
bash run_infer_alignprop_4gpu.sh
echo "[watcher] eval done at $(date)"
