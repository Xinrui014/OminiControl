#!/bin/bash
#SBATCH --job-name=pcb-v3-3-refined-1024
#SBATCH --partition=cluster02
#SBATCH --gpus=pro6000:4
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/pcb_v3_3_refined_1024_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/pcb_v3_3_refined_1024_%j.err

# OminiControl v3.3 training on 4x pro6000 with flash-attn.
#
# Uses dataset v2.2_subclass (human-refined over v2.1) with all 8 bug fixes
# (crop scheme 60/40, original_bbox matching, cardinal rotate via transpose,
#  self-fallback pastes real pixels, weighted-loss normalization,
#  worker_init_fn for RNG decorrelation, persistent_workers, etc).
#
# WANDB_API_KEY must be set in your shell env before `sbatch`, e.g.
#   export WANDB_API_KEY=$(pass wandb/api_key)   # or: wandb login, then copy
# The job forwards it into the worker processes.

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit_flash

export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export TMPDIR=/projects/_ssd/xrssd/tmp
export TOKENIZERS_PARALLELISM=true
export WANDB_DIR=/projects/_ssd/xrssd/runs
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Forward WANDB_API_KEY from sbatch submission env, if set.
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set — wandb logging will be disabled."
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p /projects/_ssd/xrssd/logs
cd /projects/_ssd/xrssd/OminiControl

echo "Starting v3.3 refined 1024 training at $(date)..."
echo "Env: qwen_edit_flash (flash_attn 2.8.3, PyTorch 2.11)"
echo "Data: v2.2_subclass (human-refined)"
echo "GPUs: pro6000 x4, bs=2/gpu, accum=2, global_bs=16"
torchrun --nproc_per_node=4 --master_port=41361 train_v3_3_1024.py
echo "Training finished at $(date) with exit code $?"
echo "Job holding GPUs — scancel to release, or wait for time wall"
sleep infinity
