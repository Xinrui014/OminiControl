#!/bin/bash
#SBATCH --job-name=pcb-v3.1-newdata-1024
#SBATCH --partition=cluster02
#SBATCH --gpus=4
#SBATCH -C gpu_48g
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/pcb_v3_1_1024_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/pcb_v3_1_1024_%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate omini

export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export TMPDIR=/projects/_ssd/xrssd/tmp
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY=a5ebf533c17c677bcee36f66c91907b5fb102f7c
export WANDB_DIR=/projects/_ssd/xrssd/runs
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p /projects/_ssd/xrssd/logs
cd /projects/_ssd/xrssd/OminiControl

echo "Starting v3.1 1024 training at $(date)..."
torchrun --nproc_per_node=4 --master_port=41359 train_v3_1_1024.py
echo "Training finished at $(date)."
