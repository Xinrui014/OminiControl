"""Profile one training step to find the bottleneck."""
import os
import time
import torch

os.environ.setdefault('OMINI_CONFIG', 'train/config/pcb_harmonize_v3_newdata_cluster.yaml')
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('HF_HOME', '/projects/_ssd/xrssd/checkpoints/hf_cache')
os.environ.setdefault('TMPDIR', '/projects/_ssd/xrssd/tmp')

from omini.train_flux.trainer import OminiModel, get_config
from train_pcb_v4_subclass import PCBHarmonizeDatasetV4
from lib.component_bank_v2_1 import ComponentBankV2_1
from torch.utils.data import DataLoader

config = get_config()
config['train']['dataset']['anno_dir'] = '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/annotation/train'
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

dc = config['train']['dataset']

# 1. Profile dataset/bank loading
t0 = time.time()
bank = ComponentBankV2_1(anno_dir=dc['anno_dir'], image_dir=dc['image_dir'])
t1 = time.time()
print(f"[1] Bank loading: {t1-t0:.1f}s")

ds = PCBHarmonizeDatasetV4(
    anno_dir=dc['anno_dir'], image_dir=dc['image_dir'], component_bank=bank,
    condition_size=(1024, 1024), target_size=(1024, 1024),
)
t2 = time.time()
print(f"[2] Dataset init: {t2-t1:.1f}s")

# 2. Profile single sample fetch
times_fetch = []
for i in range(20):
    t = time.time()
    sample = ds[i]
    times_fetch.append(time.time() - t)
avg_fetch = sum(times_fetch) / len(times_fetch)
print(f"[3] Single sample fetch (avg of 20): {avg_fetch:.3f}s")
print(f"    At bs=4, 4 workers: theoretical data time = {avg_fetch * 4 / 4:.3f}s/step")

# 3. Profile dataloader batch
loader = DataLoader(ds, batch_size=4, num_workers=8, shuffle=True, pin_memory=True)
loader_iter = iter(loader)

# Warm up
batch = next(loader_iter)
times_batch = []
for i in range(5):
    t = time.time()
    batch = next(loader_iter)
    times_batch.append(time.time() - t)
avg_batch = sum(times_batch) / len(times_batch)
print(f"[4] DataLoader batch fetch (avg of 5, bs=4, workers=4): {avg_batch:.3f}s")

# 4. Profile model loading
t3 = time.time()
model = OminiModel(
    flux_pipe_id=config['flux_path'], lora_config=config['train'].get('lora_config'),
    device='cuda', dtype=torch.bfloat16, optimizer_config=config['train']['optimizer'],
    model_config=config.get('model', {}), gradient_checkpointing=True,
)
t4 = time.time()
print(f"[5] Model loading: {t4-t3:.1f}s")

# 5. Profile forward + backward
from omini.pipeline.flux_omini import Condition
import numpy as np

# Warm up GPU
torch.cuda.synchronize()
for warmup in range(2):
    batch = next(loader_iter)
    image = batch['image'].to('cuda', dtype=torch.bfloat16)
    cond = batch['condition_0'].to('cuda', dtype=torch.bfloat16)
    desc = batch['description']

    # Simulate what trainer does (simplified)
    torch.cuda.synchronize()

# Profile forward
times_fwd = []
times_bwd = []
for step in range(5):
    batch = next(loader_iter)
    image = batch['image'].to('cuda', dtype=torch.bfloat16)
    cond = batch['condition_0'].to('cuda', dtype=torch.bfloat16)

    torch.cuda.synchronize()
    t_fwd_start = time.time()

    # Forward pass (use model's training_step if available)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = model.training_step(batch, step)

    torch.cuda.synchronize()
    t_fwd_end = time.time()
    times_fwd.append(t_fwd_end - t_fwd_start)

    # Backward
    t_bwd_start = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd_end = time.time()
    times_bwd.append(t_bwd_end - t_bwd_start)

    model.flux_pipe.transformer.zero_grad()

avg_fwd = sum(times_fwd) / len(times_fwd)
avg_bwd = sum(times_bwd) / len(times_bwd)

print(f"[6] Forward pass (avg of 5, bs=4): {avg_fwd:.3f}s")
print(f"[7] Backward pass (avg of 5, bs=4): {avg_bwd:.3f}s")
print(f"[8] Total compute per step: {avg_fwd + avg_bwd:.3f}s")
print()
print(f"=== SUMMARY ===")
print(f"Data loading:  {avg_batch:.3f}s")
print(f"Forward:       {avg_fwd:.3f}s")
print(f"Backward:      {avg_bwd:.3f}s")
print(f"Total:         {avg_batch + avg_fwd + avg_bwd:.3f}s")
print(f"Observed:      ~26s/step")
print(f"GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f}GB peak")
