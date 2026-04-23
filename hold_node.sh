#!/bin/bash
#SBATCH --job-name=hold-pro6000
#SBATCH --partition=cluster02
#SBATCH --gpus=pro6000:4
#SBATCH --time=2-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/hold_pro6000_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/hold_pro6000_%j.err

echo "Node reserved at $(date) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Holding node — scancel to release"
sleep infinity
