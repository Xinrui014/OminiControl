"""
OminiControl v3.4 — v3.1 crop scheme + ComponentBankV2.1 + v2.2 data + bug fixes.

Changes from v3.3:
  - Crop scheme reverted to v3.1 exactly (non-zoom = 1024 native,
    zoom = 256 → 1024, no native-pad branch)
  - Keeps all bank fixes (cardinal rotate via transpose, size floor,
    original_bbox matching, real-pixel self-fallback)
  - Data unchanged: v2.2_subclass

WANDB_API_KEY must be set via env (use `wandb login` or export before launching).
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

# --- Training params: 8 GPUs × bs=2 × accum=1 = effective bs=16 ---
# Plan C tuned: stride_single=2 (half single-stream blocks skip ckpt).
# Verified 5.97 s/step, 89% VRAM — safe headroom for no mid-training OOM.
# sample_interval > max_steps → skip on-the-fly val-sample generation entirely
# (not needed, and saves tiny bit of time per save event).
config['train']['max_steps'] = 8000
config['train']['save_interval'] = 1000
config['train']['sample_interval'] = 100000  # disabled (> max_steps)
config['train']['batch_size'] = 2
config['train']['accumulate_grad_batches'] = 1
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

# --- v3.4 changes: v2.2_subclass data paths ---
config['train']['dataset']['anno_dir'] = (
    '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/annotation/train'
)
config['train']['dataset']['image_dir'] = (
    '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/image/train'
)
config['train']['save_path'] = '/projects/_ssd/xrssd/OminiControl/runs/v3.4_refined_1024'

torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

dc = config['train']['dataset']
bank = ComponentBankV2_1(anno_dir=dc['anno_dir'], image_dir=dc['image_dir'])
ds = PCBHarmonizeDatasetV3_4(
    anno_dir=dc['anno_dir'], image_dir=dc['image_dir'], component_bank=bank,
    condition_size=(1024, 1024), target_size=(1024, 1024),
)
model = OminiModel(
    flux_pipe_id=config['flux_path'], lora_config=config['train'].get('lora_config'),
    device='cuda', dtype=torch.bfloat16, optimizer_config=config['train']['optimizer'],
    model_config=config.get('model', {}), gradient_checkpointing=True,
)
# Selective gradient checkpointing (Plan C tuned):
#   double-stream blocks (19, big): ckpt all     → gc_stride_double = 1
#   single-stream blocks (38): ckpt every 2nd    → gc_stride_single = 2
# stride=3 fit tightly (99% VRAM, OOM risk at sample gen); stride=2 is safe
# with 11 GB headroom, gives 5.97 s/step (6.4% faster than full ckpt 6.38).
model.flux_pipe.transformer.gc_stride_double = 1
model.flux_pipe.transformer.gc_stride_single = 2
print(f"[launcher] gc_stride_double=1, gc_stride_single=2")

train(ds, model, config, test_function=test_function)
