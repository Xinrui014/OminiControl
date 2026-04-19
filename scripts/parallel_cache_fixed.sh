#!/bin/bash
#SBATCH --job-name=cache-pcb-4gpu
#SBATCH --partition=cluster02
#SBATCH --gpus=4
#SBATCH --constraint=gpu_48g
#SBATCH --time=6:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/cache_pcb_4gpu_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/cache_pcb_4gpu_%j.err

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

echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== Splitting dataset ==="
rm -rf ${SPLIT_DIR}
python /projects/_ssd/xrssd/split_dataset.py ${DATA_DIR} ${SPLIT_DIR} 4

echo "=== Patching attn to sdpa ==="
sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' src/qflux/models/load_model.py

echo "=== Patching dataset.py: force cache_exists=False so it writes cache, not reads ==="
# Keep use_cache=True (needed for cache_manager + file_hashes), but force cache_exists=False
# so __getitem__ won't try to load nonexistent cache entries
sed -i 's/self.cache_exists = self.cache_manager.exist(self.cache_dir)/self.cache_exists = False  # patched for parallel cache writing/' src/qflux/data/dataset.py

echo "=== Launching 4 parallel cache processes ==="
for i in 0 1 2 3; do
    CONFIG=/projects/_ssd/xrssd/cache_split_${i}.yaml
    cp configs/pcb_harmonize_qwen_edit_2511.yaml ${CONFIG}
    # Point to this GPU's data split
    sed -i "s|${DATA_DIR}|${SPLIT_DIR}/split_${i}|g" ${CONFIG}
    # Keep use_cache: true — cache_manager needs it for hashing
    # Keep cuda:0 in config — CUDA_VISIBLE_DEVICES makes the assigned GPU appear as cuda:0
    echo "GPU ${i}: launching with CUDA_VISIBLE_DEVICES=${i}, config uses cuda:0"
    CUDA_VISIBLE_DEVICES=${i} python -m qflux.main --config ${CONFIG} --cache &
done

echo "=== Waiting for all 4 processes ==="
wait
echo "=== ALL DONE ==="
du -sh ${CACHE_DIR} 2>/dev/null
find ${CACHE_DIR}/metadata -type f 2>/dev/null | wc -l
echo "metadata files cached"
