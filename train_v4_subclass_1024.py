"""
OminiControl v4_subclass_1024 — v3.1 + ComponentBankV2.1 + flash-attn + pro6000.

Changes from v3.1:
- annotation dir → v2.1_subclass, ComponentBank → V2.1
- bs=4/GPU (pro6000 98GB), accum=1, global bs=16 (same effective)
- qwen_edit_flash env with flash_attn 2.8.3 (auto-dispatched via SDPA)
"""
import os
import torch

os.environ.setdefault('OMINI_CONFIG', 'train/config/pcb_harmonize_v3_newdata_cluster.yaml')
os.environ.setdefault('WANDB_API_KEY', 'a5ebf533c17c677bcee36f66c91907b5fb102f7c')
os.environ.setdefault('WANDB_DIR', '/projects/_ssd/xrssd/runs')
os.environ.setdefault('HF_HOME', '/projects/_ssd/xrssd/checkpoints/hf_cache')

from omini.train_flux.trainer import OminiModel, get_config, train
from train_pcb_v4_subclass import PCBHarmonizeDatasetV4, test_function
from lib.component_bank_v2_1 import ComponentBankV2_1

config = get_config()

# --- Training params (global bs=16 same as v3.1) ---
config['train']['max_steps'] = 8000
config['train']['save_interval'] = 1000
config['train']['sample_interval'] = 1000
config['train']['batch_size'] = 2
config['train']['accumulate_grad_batches'] = 2  # effective global bs=16
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

# --- v4 changes ---
config['train']['dataset']['anno_dir'] = '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/annotation/train'
config['train']['save_path'] = '/projects/_ssd/xrssd/OminiControl/runs/v4_subclass_1024'

torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

dc = config['train']['dataset']
bank = ComponentBankV2_1(anno_dir=dc['anno_dir'], image_dir=dc['image_dir'])
ds = PCBHarmonizeDatasetV4(
    anno_dir=dc['anno_dir'], image_dir=dc['image_dir'], component_bank=bank,
    condition_size=(1024, 1024), target_size=(1024, 1024),
)
model = OminiModel(
    flux_pipe_id=config['flux_path'], lora_config=config['train'].get('lora_config'),
    device='cuda', dtype=torch.bfloat16, optimizer_config=config['train']['optimizer'],
    model_config=config.get('model', {}), gradient_checkpointing=True,
)
train(ds, model, config, test_function=test_function)
