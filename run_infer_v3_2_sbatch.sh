#!/bin/bash
#SBATCH --job-name=infer-v3.2
#SBATCH --partition=cluster02
#SBATCH --reservation=pro6000-new-maint
#SBATCH --qos=override-limits-but-killable
#SBATCH --constraint=midmem
#SBATCH --gpus=pro6000:8
#SBATCH --exclude=gpu-pro6000-10
#SBATCH --time=3-00:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/infer_v3.2_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/infer_v3.2_%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit_flash

export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export TMPDIR=/projects/_ssd/xrssd/tmp

cd /projects/_ssd/xrssd/OminiControl

echo "=== Starting ckpt6k inference at $(date) on $(hostname) ==="
bash run_infer_v3_2_8gpu.sh 6000
# Wait for all background processes
wait
echo "=== ckpt6k done at $(date) ==="

echo "=== Starting ckpt10k inference at $(date) ==="
bash run_infer_v3_2_8gpu.sh 10000
wait
echo "=== ckpt10k done at $(date) ==="

echo "=== ALL DONE ==="
sleep infinity
