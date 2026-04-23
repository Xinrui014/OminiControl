"""Profile with LoRA on last half of blocks only."""
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

# Last half: joint blocks 10-18 (9 blocks), single blocks 19-37 (19 blocks) + x_embedder
# Joint blocks: [0-9]+ matches 0-9, need 1[0-8] for 10-18
# Single blocks: need (19|2[0-9]|3[0-7]) for 19-37
HALF_LORA = (
    "(.*x_embedder"
    "|.*(?<!single_)transformer_blocks\\.1[0-8]\\.norm1\\.linear"
    "|.*(?<!single_)transformer_blocks\\.1[0-8]\\.attn\\.to_k"
    "|.*(?<!single_)transformer_blocks\\.1[0-8]\\.attn\\.to_q"
    "|.*(?<!single_)transformer_blocks\\.1[0-8]\\.attn\\.to_v"
    "|.*(?<!single_)transformer_blocks\\.1[0-8]\\.attn\\.to_out\\.0"
    "|.*(?<!single_)transformer_blocks\\.1[0-8]\\.ff\\.net\\.2"
    "|.*single_transformer_blocks\\.(19|2[0-9]|3[0-7])\\.norm\\.linear"
    "|.*single_transformer_blocks\\.(19|2[0-9]|3[0-7])\\.proj_mlp"
    "|.*single_transformer_blocks\\.(19|2[0-9]|3[0-7])\\.proj_out"
    "|.*single_transformer_blocks\\.(19|2[0-9]|3[0-7])\\.attn\\.to_k"
    "|.*single_transformer_blocks\\.(19|2[0-9]|3[0-7])\\.attn\\.to_q"
    "|.*single_transformer_blocks\\.(19|2[0-9]|3[0-7])\\.attn\\.to_v"
    "|.*single_transformer_blocks\\.(19|2[0-9]|3[0-7])\\.attn\\.to_out)"
)

config['train']['lora_config']['target_modules'] = HALF_LORA

dc = config['train']['dataset']
bank = ComponentBankV2_1(anno_dir=dc['anno_dir'], image_dir=dc['image_dir'])
ds = PCBHarmonizeDatasetV4(
    anno_dir=dc['anno_dir'], image_dir=dc['image_dir'], component_bank=bank,
    condition_size=(1024, 1024), target_size=(1024, 1024),
)
loader = DataLoader(ds, batch_size=2, num_workers=8, shuffle=True, pin_memory=True)
loader_iter = iter(loader)

model = OminiModel(
    flux_pipe_id=config['flux_path'], lora_config=config['train'].get('lora_config'),
    device='cuda', dtype=torch.bfloat16, optimizer_config=config['train']['optimizer'],
    model_config=config.get('model', {}), gradient_checkpointing=True,
)

# Count params
total_params = sum(p.numel() for p in model.flux_pipe.transformer.parameters())
trainable = sum(p.numel() for p in model.flux_pipe.transformer.parameters() if p.requires_grad)
print(f"\nTotal params: {total_params/1e6:.1f}M")
print(f"Trainable (LoRA last-half): {trainable/1e6:.1f}M ({100*trainable/total_params:.2f}%)")
print(f"vs full LoRA: 58.0M")

# Warm up
for _ in range(2):
    batch = next(loader_iter)
    loss = model.training_step(batch, 0)
    loss.backward()
    model.flux_pipe.transformer.zero_grad()
torch.cuda.synchronize()

# Profile
print(f"\n=== STEP TIMING (bs=2, ckpt=True, last-half LoRA) ===")
for step in range(5):
    torch.cuda.synchronize()
    t0 = time.time()
    batch = next(loader_iter)
    t1 = time.time()
    loss = model.training_step(batch, step)
    torch.cuda.synchronize()
    t2 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t3 = time.time()
    model.flux_pipe.transformer.zero_grad()
    print(f"  Step {step}: data={t1-t0:.3f}s, fwd={t2-t1:.3f}s, bwd={t3-t2:.3f}s, total={t3-t0:.3f}s")

print(f"\nGPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f}GB peak")

# Gradient size
grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.flux_pipe.transformer.parameters() if p.grad is not None)
print(f"Gradient size: {grad_bytes/1e6:.1f}MB")
