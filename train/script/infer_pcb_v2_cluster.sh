#!/bin/bash
#SBATCH --job-name=pcb-infer-v2
#SBATCH --partition=cluster02
#SBATCH --gpus=6000ada:1
#SBATCH --time=4:00:00
#SBATCH --output=/projects/_ssd/xrssd/logs/pcb_infer_v2_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/pcb_infer_v2_%j.err

set -e
echo "=== PCB v2 Patch-Level Inference ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

# ── 1. Load modules ───────────────────────────────────────────────────────────
source /etc/profile.d/z00-lmod.sh
source /etc/profile.d/eee_cluster.sh 2>/dev/null || true
module load Miniforge3
source activate
conda activate omini

echo "Python:   $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0))"

# ── 2. Environment variables ──────────────────────────────────────────────────
# FLUX.1-dev downloads to SSD (not home dir — home has 50GB limit)
export HF_HOME="/projects/_ssd/xrssd/checkpoints/hf_cache"
export TOKENIZERS_PARALLELISM=true
mkdir -p "$HF_HOME"

# ── 3. Data paths (all on /projects/_ssd/xrssd) ──────────────────────────────
DATA_ROOT="/projects/_ssd/xrssd/data/ti_pcb"
ANNO_DIR="${DATA_ROOT}/coco_annotation_85"
IMAGE_DIR="${DATA_ROOT}/images_top"
LAYOUT_DIR="${DATA_ROOT}/layout_data/v2_Color_Res_Class_xywh"
CKPT_DIR="/projects/_ssd/xrssd/OminiControl/runs/20260327-022340/ckpt/12000"
OUTPUT_DIR="/projects/_ssd/xrssd/OminiControl/patch_v2_output"

# ── 4. Run patch-level inference ──────────────────────────────────────────────
cd /projects/_ssd/xrssd/OminiControl

echo ""
echo "Running patch-level inference (512x512, no tiling)..."
echo "  Annotations: ${ANNO_DIR}"
echo "  Images:      ${IMAGE_DIR}"
echo "  Test JSONL:  ${LAYOUT_DIR}/test.jsonl"
echo "  Checkpoint:  ${CKPT_DIR}"
echo "  Output:      ${OUTPUT_DIR}"

python infer_patches_v2.py \
    --anno_dir "${ANNO_DIR}" \
    --image_dir "${IMAGE_DIR}" \
    --train_jsonl "${LAYOUT_DIR}/train.jsonl" \
    --test_jsonl "${LAYOUT_DIR}/test.jsonl" \
    --omini_ckpt "${CKPT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_boards 5 \
    --patches_per_board 2 \
    --seed 42

echo ""
echo "=== Done at $(date) ==="
ls -lh "${OUTPUT_DIR}/"
