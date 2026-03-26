
import sys
sys.path.insert(0, "/home/xinrui/projects/OminiControl")
from safetensors.torch import load_file

ckpt = load_file("runs/20260305-001915/ckpt/20000/default.safetensors")
print(f"Checkpoint has {len(ckpt)} keys")
print(f"\nFirst 10 keys:")
for i, key in enumerate(list(ckpt.keys())[:10]):
    print(f"  {key}: {ckpt[key].shape}")
    
print(f"\nLast 5 keys:")
for key in list(ckpt.keys())[-5:]:
    print(f"  {key}: {ckpt[key].shape}")
    
# Check if these are LoRA keys
lora_keys = [k for k in ckpt.keys() if 'lora' in k.lower()]
print(f"\nLoRA-related keys: {len(lora_keys)}")
if lora_keys:
    print("Sample LoRA keys:")
    for k in lora_keys[:5]:
        print(f"  {k}")
