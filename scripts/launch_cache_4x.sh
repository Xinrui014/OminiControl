#!/bin/bash
# Launch 4 independent cache jobs, one per split

DATA_DIR=/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/PCB_harmonize
SPLIT_DIR=/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/PCB_harmonize_splits
TEMPLATE=/projects/_ssd/xrssd/cache_split_template.sh

# Activate conda for python access on login node
source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit

# Split dataset
echo "=== Splitting dataset into 4 ==="
rm -rf ${SPLIT_DIR}
cd /projects/_ssd/xrssd/qwen-image-finetune
python /projects/_ssd/xrssd/split_dataset.py ${DATA_DIR} ${SPLIT_DIR} 4

# Submit 4 jobs
for i in 0 1 2 3; do
    JOB_SCRIPT=/projects/_ssd/xrssd/cache_split_${i}.sh
    sed "s/SPLIT_IDX/${i}/g" ${TEMPLATE} > ${JOB_SCRIPT}
    echo "Submitting split ${i}..."
    sbatch ${JOB_SCRIPT}
done

echo "=== All 4 jobs submitted ==="
squeue -u xinrui004
