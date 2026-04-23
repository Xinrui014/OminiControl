#!/bin/bash
#SBATCH --job-name=test-v3
#SBATCH --partition=cluster02
#SBATCH --gpus=pro6000:1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/_ssd/xrssd/logs/test_v3_%j.out
#SBATCH --error=/projects/_ssd/xrssd/logs/test_v3_%j.err

source /etc/profile.d/z00-lmod.sh
module load Miniforge3
source activate
conda activate omini

export HF_HOME=/projects/_ssd/xrssd/checkpoints/hf_cache
export OMINI_CONFIG=train/config/pcb_harmonize_v3_newdata_cluster.yaml
export TOKENIZERS_PARALLELISM=true
export TMPDIR=/projects/_ssd/xrssd/tmp

cd /projects/_ssd/xrssd/OminiControl

echo '=== Test 1: imports ==='
python -c "from omini.train_flux.train_pcb_v3_newdata import PCBHarmonizeDatasetV3; print('import OK')"

echo '=== Test 2: component bank ==='
python -c "
from lib.component_bank_v2_new import ComponentBankV2_new
bank = ComponentBankV2_new(
    anno_dir='/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/annotation/train',
    image_dir='/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/train',
)
print('Bank OK, components:', sum(len(v) for v in bank.by_cat.values()))
"

echo '=== Test 3: dataset sample ==='
python -c "
from omini.train_flux.train_pcb_v3_newdata import PCBHarmonizeDatasetV3
from lib.component_bank_v2_new import ComponentBankV2_new
bank = ComponentBankV2_new(
    anno_dir='/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/annotation/train',
    image_dir='/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/train',
)
ds = PCBHarmonizeDatasetV3(
    anno_dir='/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/annotation/train',
    image_dir='/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/train',
    component_bank=bank,
)
s = ds[0]
print('image:', s['image'].shape)
print('condition:', s['condition_0'].shape)
print('prompt:', s['description'][:100])
print('mask:', s['loss_weight_mask'].shape)
"

echo '=== Test 4: single-GPU training (10 steps) ==='
python -c "
import os, yaml, torch
os.environ['OMINI_CONFIG'] = 'train/config/pcb_harmonize_v3_newdata_cluster.yaml'
from omini.train_flux.train_pcb_v3_newdata import main
# Patch config to 10 steps
from omini.train_flux.trainer import get_config
config = get_config()
config['train']['max_steps'] = 10
config['train']['save_interval'] = 999
config['train']['sample_interval'] = 999
config['train']['save_path'] = '/projects/_ssd/xrssd/OminiControl/runs/test_v3_newdata'
config['train']['batch_size'] = 1
config['train']['accumulate_grad_batches'] = 1

from lib.component_bank_v2_new import ComponentBankV2_new
from omini.train_flux.train_pcb_v3_newdata import PCBHarmonizeDatasetV3
from omini.train_flux.trainer import OminiModel, train

dc = config['train']['dataset']
bank = ComponentBankV2_new(anno_dir=dc['anno_dir'], image_dir=dc['image_dir'])
ds = PCBHarmonizeDatasetV3(
    anno_dir=dc['anno_dir'], image_dir=dc['image_dir'], component_bank=bank,
    condition_size=tuple(dc['condition_size']), target_size=tuple(dc['target_size']),
)
model = OminiModel(
    flux_pipe_id=config['flux_path'], lora_config=config['train'].get('lora_config'),
    device='cuda', dtype=torch.bfloat16, optimizer_config=config['train']['optimizer'],
    model_config=config.get('model', {}), gradient_checkpointing=True,
)
train(ds, model, config, test_function=None)
print('Training 10 steps OK')
"

echo '=== ALL TESTS DONE ==='
