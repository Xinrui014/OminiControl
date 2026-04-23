#!/bin/bash
while true; do
  n=$(ls /projects/_ssd/xrssd/OminiControl/runs/alignprop_prod_4gpu_500step_v2/eval_full_step300 2>/dev/null | wc -l)
  alive=$(pgrep -c -f infer_alignprop_fromcomposite)
  echo "count=$n alive=$alive"
  if grep -q "all 4 ranks completed" /projects/_ssd/xrssd/OminiControl/runs/alignprop_prod_4gpu_500step_v2/eval_full_launcher.log 2>/dev/null; then
    echo FINISHED
    break
  fi
  if [ "$alive" -lt 4 ] && [ "$n" -lt 2186 ]; then
    echo WORKER-DIED
    break
  fi
  sleep 300
done
