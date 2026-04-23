"""Deep profiling — check flash attention, DDP overhead, optimizer step, etc."""
import os
import time
import torch
import torch.nn.functional as F

os.environ.setdefault('OMINI_CONFIG', 'train/config/pcb_harmonize_v3_newdata_cluster.yaml')
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('HF_HOME', '/projects/_ssd/xrssd/checkpoints/hf_cache')
os.environ.setdefault('TMPDIR', '/projects/_ssd/xrssd/tmp')

# 1. Check if flash attention is available
print("=== ATTENTION BACKEND CHECK ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"SM capability: {torch.cuda.get_device_capability(0)}")

try:
    import flash_attn
    print(f"flash_attn package: {flash_attn.__version__}")
except ImportError:
    print("flash_attn package: NOT INSTALLED")

# Check which SDPA backend gets used
print("\nSDPA backend test (bf16, head_dim=128):")
with torch.no_grad():
    q = torch.randn(1, 24, 4096, 128, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(1, 24, 4096, 128, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(1, 24, 4096, 128, dtype=torch.bfloat16, device='cuda')

    # Time SDPA
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"  SDPA (auto): {(t1-t0)/10*1000:.1f}ms")

    # Check which backend was used
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for backend_name, backend in [("FLASH", SDPBackend.FLASH_ATTENTION),
                                   ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                                   ("MATH", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    out = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                t1 = time.time()
                print(f"  {backend_name}: {(t1-t0)/10*1000:.1f}ms")
        except Exception as e:
            print(f"  {backend_name}: FAILED ({e})")

del q, k, v, out
torch.cuda.empty_cache()

# 2. Load model and dataset
print("\n=== MODEL + DATA SETUP ===")
from omini.train_flux.trainer import OminiModel, get_config
from train_pcb_v4_subclass import PCBHarmonizeDatasetV4
from lib.component_bank_v2_1 import ComponentBankV2_1
from torch.utils.data import DataLoader

config = get_config()
config['train']['dataset']['anno_dir'] = '/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2.1_subclass/annotation/train'
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

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

# 3. Count trainable params
total_params = sum(p.numel() for p in model.flux_pipe.transformer.parameters())
trainable_params = sum(p.numel() for p in model.flux_pipe.transformer.parameters() if p.requires_grad)
print(f"Total params: {total_params/1e6:.1f}M")
print(f"Trainable (LoRA): {trainable_params/1e6:.1f}M ({100*trainable_params/total_params:.2f}%)")

# 4. Profile full training step (forward + backward + optimizer)
print("\n=== FULL STEP PROFILING (bs=2) ===")

# Warm up
for _ in range(2):
    batch = next(loader_iter)
    loss = model.training_step(batch, 0)
    loss.backward()
    model.flux_pipe.transformer.zero_grad()
torch.cuda.synchronize()

# Detailed timing
for step in range(3):
    torch.cuda.synchronize()

    t_data = time.time()
    batch = next(loader_iter)
    t_data_end = time.time()

    t_fwd = time.time()
    loss = model.training_step(batch, step)
    torch.cuda.synchronize()
    t_fwd_end = time.time()

    t_bwd = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd_end = time.time()

    # Simulate optimizer step (Prodigy)
    t_opt = time.time()
    # Just zero grad to measure overhead (real optimizer would update)
    model.flux_pipe.transformer.zero_grad()
    torch.cuda.synchronize()
    t_opt_end = time.time()

    print(f"  Step {step}: data={t_data_end-t_data:.3f}s, fwd={t_fwd_end-t_fwd:.3f}s, "
          f"bwd={t_bwd_end-t_bwd:.3f}s, zero_grad={t_opt_end-t_opt:.3f}s, "
          f"total={t_opt_end-t_data:.3f}s")

# 5. Check gradient sizes
print("\n=== GRADIENT STATS ===")
batch = next(loader_iter)
loss = model.training_step(batch, 0)
loss.backward()
grad_sizes = []
for name, p in model.flux_pipe.transformer.named_parameters():
    if p.grad is not None:
        grad_sizes.append((name, p.grad.numel(), p.grad.element_size()))
total_grad_bytes = sum(n * s for _, n, s in grad_sizes)
print(f"Gradient tensors: {len(grad_sizes)}")
print(f"Total gradient size: {total_grad_bytes/1e6:.1f}MB")
print(f"(This is what DDP all-reduces each step)")
model.flux_pipe.transformer.zero_grad()

# 6. Memory breakdown
print("\n=== MEMORY ===")
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.1f}GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

# 7. Estimate DDP overhead
print("\n=== DDP OVERHEAD ESTIMATE ===")
print(f"Gradient size: {total_grad_bytes/1e6:.1f}MB")
print(f"4-GPU all-reduce (ring): ~2x gradient size = {2*total_grad_bytes/1e6:.1f}MB transferred")
print(f"At ~50GB/s NVLink: {2*total_grad_bytes/50e9:.3f}s")
print(f"At ~25GB/s PCIe: {2*total_grad_bytes/25e9:.3f}s")
print(f"At ~12GB/s InfiniBand: {2*total_grad_bytes/12e9:.3f}s")
print(f"Note: pro6000 nodes likely use PCIe, not NVLink")
