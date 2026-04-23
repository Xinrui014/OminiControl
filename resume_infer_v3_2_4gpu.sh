#!/bin/bash
# Resume v3.2 inference — skip already-done patches by finding the first missing one per GPU range
set -e

STEP=${1:-10000}
RUN_DIR="/projects/_ssd/xrssd/OminiControl/runs/v4_subclass_1024/20260415-104945"
CKPT="$RUN_DIR/ckpt/$STEP"
EVAL_JSON="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/eval_patches_full_test_v2.1.json"
STEP_LABEL=$((STEP / 1000))k
OUT_DIR="$RUN_DIR/eval_full/ckpt${STEP_LABEL}"

echo "=== v3.2 inference RESUME ==="
echo "Checkpoint: $CKPT"
echo "Output:     $OUT_DIR"

DONE=$(ls "$OUT_DIR"/*_harmonized.png 2>/dev/null | wc -l)
echo "Already done: $DONE / 2186"

if [ "$DONE" -ge 2186 ]; then
    echo "All patches done!"
    exit 0
fi

mkdir -p "$OUT_DIR"
cd /projects/_ssd/xrssd/OminiControl

LOG_DIR="/projects/_ssd/xrssd/logs/infer_v3.2_resume_ckpt${STEP_LABEL}"
mkdir -p "$LOG_DIR"

# Find which patches are missing and split across 4 GPUs
python3 -c "
import json, os
with open('$EVAL_JSON') as f:
    patches = json.load(f)
done = set()
for f in os.listdir('$OUT_DIR'):
    if f.endswith('_harmonized.png'):
        done.add(f.replace('_harmonized.png',''))
missing = [i for i,p in enumerate(patches) if p['patch_id'] not in done]
print(f'Missing: {len(missing)} patches')
# Split missing indices into 4 roughly equal chunks
n = len(missing)
chunk = (n + 3) // 4
for gpu in range(4):
    start = gpu * chunk
    end = min((gpu+1) * chunk, n)
    if start >= n: break
    indices = missing[start:end]
    # Write index file for each GPU
    with open(f'$LOG_DIR/gpu{gpu}_indices.json', 'w') as f:
        json.dump(indices, f)
    print(f'GPU {gpu}: {len(indices)} patches (indices {indices[0]}..{indices[-1]})')
"

# Launch inference per GPU using the index-based approach
# Since the script uses --start/--end, we need a different approach:
# Just re-run full range but skip existing files
for i in 0 1 2 3; do
    START=$((i * 547))
    END=$(( (i + 1) * 547 ))
    if [ $END -gt 2186 ]; then END=2186; fi

    echo "GPU $i: patches [$START, $END) (skipping existing)"
    CUDA_VISIBLE_DEVICES=$i nohup python infer_v3_2_full_eval.py \
        --eval_json "$EVAL_JSON" \
        --omini_ckpt "$CKPT" \
        --output_dir "$OUT_DIR" \
        --start $START --end $END \
        > "$LOG_DIR/gpu${i}.log" 2>&1 &
    echo "  -> PID $!"
done

echo ""
echo "All launched. Logs: $LOG_DIR/gpu{0,1,2,3}.log"
