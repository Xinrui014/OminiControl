#!/bin/bash
#SBATCH --job-name=v3-4-bench-p10
#SBATCH --partition=cluster02
#SBATCH --reservation=pro6000-stability-test
#SBATCH --qos=override-limits-but-killable
#SBATCH --nodelist=gpu-pro6000-10
#SBATCH --gpus=pro6000:8
#SBATCH --time=1:30:00
#SBATCH --output=/projects/_ssd/xrssd/logs/v3_4_bench_p10_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/v3_4_bench_p10_%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate /projects/_ssd/xrssd/envs/qwen_edit_flash

export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export TMPDIR=/projects/_ssd/xrssd/tmp
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=disabled
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Host: $(hostname)"
echo "Date: $(date)"

cd /projects/_ssd/xrssd/OminiControl

# master_port=41363 so it doesn't collide with gpu-pro6000-9's 41362
torchrun --nproc_per_node=8 --master_port=41363 train_v3_4_1024_bench.py
echo "Exit code: $?"
