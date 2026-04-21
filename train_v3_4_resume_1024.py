"""
OminiControl v3.4 RESUME — warm-start from v3.4 ckpt8000 + train 6000 more steps.

Identical to train_v3_4_1024.py except:
  - Loads LoRA weights from ckpt/8000 via `lora_path` (warm-start)
  - max_steps = 6000 (saves at +1k, +2k, +3k, +4k, +5k, +6k past ckpt8k)
  - save_path points to v3.4_resumed_from8k_1024 (new run dir)

Prodigy optimizer re-inits fresh (no saved optimizer state). `safeguard_warmup:
true` in config mitigates the ramp-up. Model is already converged at loss ~0.30,
so brief lr warm-up should not spike loss significantly.
"""
import os
import torch

os.environ.setdefault('OMINI_CONFIG', 'train/config/pcb_harmonize_v3_newdata_cluster.yaml')
os.environ.setdefault('WANDB_DIR', '/projects/_ssd/xrssd/runs')
os.environ.setdefault('HF_HOME', '/projects/_ssd/xrssd/checkpoints/hf_cache')

from omini.train_flux.trainer import OminiModel, get_config, train
from train_pcb_v3_4 import PCBHarmonizeDatasetV3_4, test_function
from lib.component_bank_v2_1 import ComponentBankV2_1

config = get_config()

# --- Training params (mostly same as v3.4) ---
config['train']['max_steps'] = 6000
config['train']['save_interval'] = 1000
config['train']['sample_interval'] = 100000
config['train']['batch_size'] = 2
config['train']['accumulate_grad_batches'] = 1
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

# --- v2.2_subclass data paths ---
config['train']['dataset']['anno_dir'] = (
    '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/annotation/train'
)
config['train']['dataset']['image_dir'] = (
    '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/image/train'
)

# --- NEW: resume save dir ---
config['train']['save_path'] = '/projects/_ssd/xrssd/OminiControl/runs/v3.4_resumed_from8k_1024'

# --- NEW: warm-start source ---
RESUME_LORA_PATH = '/projects/_ssd/xrssd/OminiControl/runs/v3.4_refined_1024/20260420-104031/ckpt/8000'

torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

dc = config['train']['dataset']
bank = ComponentBankV2_1(anno_dir=dc['anno_dir'], image_dir=dc['image_dir'])
ds = PCBHarmonizeDatasetV3_4(
    anno_dir=dc['anno_dir'], image_dir=dc['image_dir'], component_bank=bank,
    condition_size=(1024, 1024), target_size=(1024, 1024),
)

# NEW: pass lora_path to warm-start from v3.4 ckpt8k
model = OminiModel(
    flux_pipe_id=config['flux_path'],
    lora_path=RESUME_LORA_PATH,
    lora_config=config['train'].get('lora_config'),
    device='cuda', dtype=torch.bfloat16, optimizer_config=config['train']['optimizer'],
    model_config=config.get('model', {}), gradient_checkpointing=True,
)
# Selective gradient checkpointing (same as v3.4): stride_single=2 → 5.97 s/step.
model.flux_pipe.transformer.gc_stride_double = 1
model.flux_pipe.transformer.gc_stride_single = 2
print(f"[launcher] Resumed from {RESUME_LORA_PATH}")
print(f"[launcher] gc_stride_double=1, gc_stride_single=2")

train(ds, model, config, test_function=test_function)
