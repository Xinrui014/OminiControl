#!/bin/bash
#SBATCH --job-name=cache-pcb-SPLIT_IDX
#SBATCH --partition=cluster02
#SBATCH --gpus=1
#SBATCH --constraint=gpu_48g
#SBATCH --time=6:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/cache_pcb_split_SPLIT_IDX_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/cache_pcb_split_SPLIT_IDX_%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit
export HF_HOME=/projects/_ssd/xrssd/cache/huggingface
export TMPDIR=/projects/_ssd/xrssd/tmp

cd /projects/_ssd/xrssd/qwen-image-finetune

DATA_DIR=/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/PCB_harmonize
SPLIT_DIR=/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/PCB_harmonize_splits
CACHE_DIR=/projects/_ssd/xrssd/runs/pcb_harmonize_qwen_edit_v1/cache
SPLIT_IDX=SPLIT_IDX

echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== Patching attn to sdpa ==="
sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' src/qflux/models/load_model.py

CONFIG=/projects/_ssd/xrssd/cache_split_${SPLIT_IDX}.yaml
cp configs/pcb_harmonize_qwen_edit_2511.yaml ${CONFIG}
sed -i "s|${DATA_DIR}|${SPLIT_DIR}/split_${SPLIT_IDX}|g" ${CONFIG}
sed -i 's/use_cache: true/use_cache: false/' ${CONFIG}

echo "=== Starting cache for split ${SPLIT_IDX} ==="
python -m qflux.main --config ${CONFIG} --cache

echo "=== DONE split ${SPLIT_IDX} ==="
find ${CACHE_DIR}/metadata -type f 2>/dev/null | wc -l
echo "total metadata files"
