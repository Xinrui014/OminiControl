#!/bin/bash
#SBATCH --job-name=pcb-omini
#SBATCH --partition=cluster02
#SBATCH --gpus pro6000:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/xrssd/logs/pcb_train_%j.out
#SBATCH --error=/projects/xrssd/logs/pcb_train_%j.err
#SBATCH --qos=override-limits-but-killable

# NOTE: Remove --qos above if you have enough GPU quota in the default fair-share queue.

set -e
echo "=== PCB OminiControl Training — 2x Pro6000 Blackwell ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"

# ── 0. Init Lmod ──────────────────────────────────────────────────────────────
source /etc/profile.d/z00-lmod.sh
source /etc/profile.d/eee_cluster.sh 2>/dev/null || true

# ── 1. Load modules ────────────────────────────────────────────────────────────
module load Miniforge3
module load CUDA/12.8.0
module load cuDNN/9.10.1.4-CUDA-12.8.0
source activate /projects/xrssd/envs/omini_cuda128

echo "Python:   $(python --version)"
echo "CUDA mod: $CUDA_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU count:', torch.cuda.device_count())"

# ── 2. Environment variables ───────────────────────────────────────────────────
export HF_HOME="/projects/xrssd/checkpoints/hf_cache"
export OMINI_CONFIG="./train/config/pcb_harmonize_cluster.yaml"
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY="a5ebf533c17c677bcee36f66c91907b5fb102f7c"

# NCCL / Blackwell settings
export NCCL_P2P_DISABLE=0
export TORCH_NCCL_ENABLE_MONITORING=0

# ── 3. Run DDP training ────────────────────────────────────────────────────────
mkdir -p /projects/xrssd/logs
cd /projects/xrssd/OminiControl

echo "Starting training at $(date)..."
torchrun \
    --nproc_per_node=2 \
    --master_port=41355 \
    -m omini.train_flux.train_pcb

echo "Training finished at $(date)."
