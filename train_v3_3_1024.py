"""
OminiControl v3.3 refined — 1024×1024 training with dataset v2.2.

Refinements over v3.2 (v4_subclass):
- annotation dir → v2.2_subclass (human-refined over v2.1), ComponentBank → V2.1
- dataset class → PCBHarmonizeDatasetV3_3 (see train_pcb_v3_3.py):
    * 60% crops at 512 native → LANCZOS → 1024 (matches inference)
    * 40% crops at 1024 native on WHITE-padded 1280×1280 board (random placement)
    * match via original_bbox (not clipped), visible-portion paste
    * self-fallback pastes real board pixels instead of squeezing
- trainer.py: weighted loss is sum-normalized; DataLoader has worker_init_fn
- component_bank_v2_1.py: cardinal rotations via Image.transpose (no BILINEAR blur)

WANDB_API_KEY must be set via env (use `wandb login` or export before launching).
"""
import os
import torch

os.environ.setdefault('OMINI_CONFIG', 'train/config/pcb_harmonize_v3_newdata_cluster.yaml')
os.environ.setdefault('WANDB_DIR', '/projects/_ssd/xrssd/runs')
os.environ.setdefault('HF_HOME', '/projects/_ssd/xrssd/checkpoints/hf_cache')

from omini.train_flux.trainer import OminiModel, get_config, train
from train_pcb_v3_3 import PCBHarmonizeDatasetV3_3, test_function
from lib.component_bank_v2_1 import ComponentBankV2_1

config = get_config()

# --- Training params ---
# 8 GPUs × bs=2 × accum=1 = effective bs=16. Full gradient checkpointing (stride=1).
# Benchmarked fastest config:
#   bs=1/accum=2/8GPU/stride=2: 7.75 s/step
#   bs=2/accum=1/8GPU/stride=1: 6.38 s/step  ← winner (14.2h for 8k steps)
#   bs=4/accum=1/4GPU/stride=1: 12.76 s/step
config['train']['max_steps'] = 8000
config['train']['save_interval'] = 1000
config['train']['sample_interval'] = 1000
config['train']['batch_size'] = 2
config['train']['accumulate_grad_batches'] = 1
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

# --- v3.3 changes ---
# Dataset v2.2 (human-refined annotations, same 3515 boards as v2.1)
config['train']['dataset']['anno_dir'] = (
    '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/annotation/train'
)
config['train']['dataset']['image_dir'] = (
    '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/image/train'
)
config['train']['save_path'] = '/projects/_ssd/xrssd/OminiControl/runs/v3.3_refined_1024'

torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

dc = config['train']['dataset']
bank = ComponentBankV2_1(anno_dir=dc['anno_dir'], image_dir=dc['image_dir'])
ds = PCBHarmonizeDatasetV3_3(
    anno_dir=dc['anno_dir'], image_dir=dc['image_dir'], component_bank=bank,
    condition_size=(1024, 1024), target_size=(1024, 1024),
)
model = OminiModel(
    flux_pipe_id=config['flux_path'], lora_config=config['train'].get('lora_config'),
    device='cuda', dtype=torch.bfloat16, optimizer_config=config['train']['optimizer'],
    model_config=config.get('model', {}), gradient_checkpointing=True,
)
# Partial gradient checkpointing: stride=1 means checkpoint EVERY block
# (full checkpointing, v3.2 baseline). Slowest but safest memory-wise.
model.flux_pipe.transformer.gc_stride = 1
print(f"[launcher] gc_stride={model.flux_pipe.transformer.gc_stride}")

train(ds, model, config, test_function=test_function)
