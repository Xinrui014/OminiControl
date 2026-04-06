#!/bin/bash
# PCB Harmonization v3_newdata — ComponentBankV2_new + v2 annotations
#
# Same as v2.1 except:
#   - ComponentBankV2_new (orientation-first matching, resolution class, RGBA fix)
#   - Paste at native crop resolution, resize after
#   - v2 annotations (3515 boards with orientation/resolution_class)
#   - 8k steps

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMINI_CONFIG=train/config/pcb_harmonize_v3_newdata.yaml
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY=a5ebf533c17c677bcee36f66c91907b5fb102f7c

echo 'Starting PCB v3_newdata training...'
echo 'Config:' $OMINI_CONFIG

/home/xinrui/miniconda3/envs/omini/bin/torchrun \
    --nproc_per_node=4 \
    --master_port=41357 \
    -m omini.train_flux.train_pcb_v3_newdata
