"""
Benchmark launcher — same v3.4 config as train_v3_4_1024.py but max_steps=30
and separate save_path so it doesn't collide with the live v3.4 run.
"""
import os
import torch

os.environ.setdefault('OMINI_CONFIG', 'train/config/pcb_harmonize_v3_newdata_cluster.yaml')
os.environ.setdefault('WANDB_DIR', '/projects/_ssd/xrssd/runs')
os.environ.setdefault('HF_HOME', '/projects/_ssd/xrssd/checkpoints/hf_cache')
# Disable wandb for the benchmark run (avoid polluting the real project)
os.environ['WANDB_MODE'] = 'disabled'

from omini.train_flux.trainer import OminiModel, get_config, train
from train_pcb_v3_4 import PCBHarmonizeDatasetV3_4, test_function
from lib.component_bank_v2_1 import ComponentBankV2_1

config = get_config()

# --- BENCH: 500 steps, per-step logging ---
config['train']['max_steps'] = 500
config['train']['print_every_n_steps'] = 1
config['train']['save_interval'] = 100000   # don't save ckpts
config['train']['sample_interval'] = 100000 # no sampling
config['train']['batch_size'] = 2
config['train']['accumulate_grad_batches'] = 1
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

config['train']['dataset']['anno_dir'] = (
    '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/annotation/train'
)
config['train']['dataset']['image_dir'] = (
    '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.2_subclass/image/train'
)
# Separate save dir so it doesn't overwrite the live v3.4 run
config['train']['save_path'] = '/projects/_ssd/xrssd/OminiControl/runs/v3.4_bench_p10'

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
model.flux_pipe.transformer.gc_stride_double = 1
model.flux_pipe.transformer.gc_stride_single = 2
print(f"[bench] gc_stride_double=1, gc_stride_single=2 — gpu-pro6000-10 (baremetal)")

train(ds, model, config, test_function=test_function)
