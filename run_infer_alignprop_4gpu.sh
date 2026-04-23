#!/bin/bash
# Run v3.4+delta_step_300 inference on full 2186-patch test set using
# EXACT composites from v3.4 ckpt14k eval (read-only). 4-GPU parallel shards.
set -e
cd /projects/_ssd/xrssd/OminiControl
source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate /projects/_ssd/xrssd/envs/qwen_edit_flash
export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export TMPDIR=/projects/_ssd/xrssd/tmp

V34=/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/20260421-060156/ckpt/6000
DELTA=/projects/_ssd/xrssd/OminiControl/runs/alignprop_prod_4gpu_500step_v2/ckpt/step_300.safetensors
COMPDIR=/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024/eval_full/ckpt14k
EVAL=/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json
OUT=/projects/_ssd/xrssd/OminiControl/runs/alignprop_prod_4gpu_500step_v2/eval_full_step300
LOG=/projects/_ssd/xrssd/OminiControl/runs/alignprop_prod_4gpu_500step_v2/eval_full_step300.log
mkdir -p "$OUT"

# Patch counts: total 2186 → shards 547/546/547/546
N=$(python3 -c "import json; print(len(json.load(open('$EVAL'))))")
echo "[config] $N patches, 4 GPUs, shards of ~$((N/4))"

for rank in 0 1 2 3; do
    start=$(( rank * N / 4 ))
    end=$(( (rank + 1) * N / 4 ))
    [ $rank -eq 3 ] && end=$N
    echo "[launch] GPU $rank: patches [$start:$end]"
    CUDA_VISIBLE_DEVICES=$rank nohup python -u infer_alignprop_fromcomposite.py \
        --v34_ckpt "$V34" --delta_ckpt "$DELTA" \
        --composites_dir "$COMPDIR" --eval_json "$EVAL" \
        --output_dir "$OUT" --start $start --end $end \
        > "${LOG}.rank${rank}" 2>&1 &
done
wait
echo "[done] all 4 ranks completed at $(date)"
ls "$OUT" | wc -l
