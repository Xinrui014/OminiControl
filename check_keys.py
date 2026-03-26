
import sys
sys.path.insert(0, "/home/xinrui/projects/OminiControl")
import torch
import yaml
from omini.train_flux.trainer import OminiModel
from safetensors.torch import load_file

# Load config
with open("runs/20260305-001915/config.yaml") as f:
    config = yaml.safe_load(f)
training_config = config["train"]

# Create model (same as my inference)
model = OminiModel(
    flux_pipe_id=config["flux_path"],
    lora_config=training_config["lora_config"],
    device="cpu",  # Use CPU for faster init
    dtype=torch.bfloat16,
    optimizer_config=training_config["optimizer"],
    model_config=config.get("model", {}),
    gradient_checkpointing=False,
)

# Get model keys
model_keys = set(model.state_dict().keys())
print(f"Model has {len(model_keys)} keys in state_dict")

# Load checkpoint
ckpt = load_file("runs/20260305-001915/ckpt/20000/default.safetensors")
ckpt_keys = set(ckpt.keys())
print(f"Checkpoint has {len(ckpt_keys)} keys")

# Check overlap
common = model_keys & ckpt_keys
only_model = model_keys - ckpt_keys  
only_ckpt = ckpt_keys - model_keys

print(f"\nCommon keys: {len(common)}")
print(f"Only in model: {len(only_model)}")
print(f"Only in checkpoint: {len(only_ckpt)}")

if only_model:
    print(f"\nSample keys only in model (first 5):")
    for k in list(only_model)[:5]:
        print(f"  {k}")

if only_ckpt:
    print(f"\nSample keys only in checkpoint (first 5):")
    for k in list(only_ckpt)[:5]:
        print(f"  {k}")

if common:
    print(f"\nSample common keys (first 5):")
    for k in list(common)[:5]:
        print(f"  {k}")
