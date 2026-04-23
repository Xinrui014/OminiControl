#!/bin/bash
# Upload PCB v2.1_subclass dataset to HuggingFace
# Run from cluster: bash upload_dataset.sh

set -e

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate /projects/_ssd/xrssd/envs/qwen_edit_flash
pip install -q huggingface_hub[cli]

REPO="NothingSpecialSiri/pcb-v2.1-subclass"
ANNO_DIR="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/annotation"
IMAGE_DIR="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image"
REGISTRY="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/subclass_registry.json"

echo "=== Uploading PCB v2.1_subclass to $REPO ==="

# Login (will prompt for token if not already logged in)
huggingface-cli whoami || huggingface-cli login

# Create dataset repo (private by default)
huggingface-cli repo create "$REPO" --type dataset --private 2>/dev/null || echo "Repo already exists"

# Upload annotations (train + test)
echo "Uploading annotations..."
huggingface-cli upload "$REPO" "$ANNO_DIR/" annotation/ --repo-type dataset

# Upload subclass registry
echo "Uploading subclass_registry.json..."
huggingface-cli upload "$REPO" "$REGISTRY" subclass_registry.json --repo-type dataset

# Upload images (follow symlinks)
echo "Uploading images (this may take a while)..."
huggingface-cli upload "$REPO" "$IMAGE_DIR/" image/ --repo-type dataset

echo "=== Done! ==="
echo "Dataset at: https://huggingface.co/datasets/$REPO"
