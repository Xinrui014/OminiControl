#!/bin/bash
#SBATCH --job-name=hold-pro6000-midmem
#SBATCH --partition=cluster02
#SBATCH --reservation=pro6000-new-maint
#SBATCH --qos=override-limits-but-killable
#SBATCH --constraint=midmem
#SBATCH --gpus=pro6000:8
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/hold_pro6000_midmem_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/hold_pro6000_midmem_%j.err

echo "Node reserved at $(date) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Holding node — scancel to release"
sleep infinity
