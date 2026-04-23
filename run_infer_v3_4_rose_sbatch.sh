#!/bin/bash
#SBATCH --job-name=infer-v3.4-14k
#SBATCH --partition=cluster02
#SBATCH --qos=rose
#SBATCH --gpus=pro6000:4
#SBATCH --time=5:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/infer_v3.4_14k_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/infer_v3.4_14k_%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit_flash

export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export TMPDIR=/projects/_ssd/xrssd/tmp

cd /projects/_ssd/xrssd/OminiControl
echo "=== Starting v3.4 ckpt14k full inference at $(date) on $(hostname) ==="
bash run_infer_v3_4_4gpu.sh
echo "=== Done at $(date) ==="
