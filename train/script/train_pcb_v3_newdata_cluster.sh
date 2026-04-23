#!/bin/bash
#SBATCH --job-name=pcb-v3-newdata
#SBATCH --partition=cluster02
#SBATCH --gpus=4
#SBATCH -C gpu_48g
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/pcb_v3_newdata_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/pcb_v3_newdata_%j.err

set -e
echo "=== PCB v3_newdata OminiControl Training — 4x 48GB GPU ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /home/xinrui004/.conda/envs/omini

export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export TMPDIR=/projects/_ssd/xrssd/tmp
export PIP_CACHE_DIR=/projects/_ssd/xrssd/cache/pip
export CONDA_PKGS_DIRS=/projects/_ssd/xrssd/cache/conda_pkgs
export PYTHONUSERBASE=/projects/_ssd/xrssd/python_user
export OMINI_CONFIG=train/config/pcb_harmonize_v3_newdata_cluster.yaml
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY=a5ebf533c17c677bcee36f66c91907b5fb102f7c
export WANDB_DIR=/projects/_ssd/xrssd/runs
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Python: $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch:', torch.__version__); print('GPUs:', torch.cuda.device_count())"

mkdir -p /projects/_ssd/xrssd/logs
cd /projects/_ssd/xrssd/OminiControl

echo "Starting training at $(date)..."
torchrun \
    --nproc_per_node=4 \
    --master_port=41358 \
    -m omini.train_flux.train_pcb_v3_newdata

echo "Training finished at $(date)."
