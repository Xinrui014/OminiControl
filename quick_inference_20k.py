#!/usr/bin/env python3
"""Quick inference test on checkpoint 20000 - 1 train sample + 1 test sample"""
import os
import sys
import json
import numpy as np
import torch
import yaml
from PIL import Image
from pathlib import Path

sys.path.insert(0, "/home/xinrui/projects/OminiControl")
from omini.pipeline.flux_omini import Condition, generate
from omini.train_flux.trainer import OminiModel
from safetensors.torch import load_file

checkpoint_path = "runs/20260305-001915/ckpt/20000"
output_dir = "inference_output_ckpt20k_comparison"
os.makedirs(output_dir, exist_ok=True)

# Load config
config_path = "runs/20260305-001915/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
training_config = config["train"]

print("Loading model...")
model = OminiModel(
    flux_pipe_id=config["flux_path"],
    lora_config=training_config["lora_config"],
    device="cuda:2",
    dtype=getattr(torch, config["dtype"]),
    optimizer_config=training_config["optimizer"],
    model_config=config.get("model", {}),
    gradient_checkpointing=False,
)

# Load checkpoint 20000
ckpt_file = os.path.join(checkpoint_path, "default.safetensors")
print(f"Loading checkpoint: {ckpt_file}")
state_dict = load_file(ckpt_file)
model.load_state_dict(state_dict, strict=False)
model.eval()

adapter = model.adapter_names[2]
target_size = (512, 512)
condition_size = (512, 512)

# Define samples
samples = [
    {
        "name": "TRAIN_sample100",
        "manifest": "/home/xinrui/projects/data/ti_pcb/COCO_label/cropped_512/composite_manifest_train.json",
        "index": 100,
        "type": "TRAINING"
    },
    {
        "name": "TEST_sample0",
        "manifest": "/home/xinrui/projects/data/ti_pcb/COCO_label/cropped_512/composite_manifest_test.json",
        "index": 0,
        "type": "TESTING"
    }
]

data_root = "/home/xinrui/projects/data/ti_pcb/COCO_label/cropped_512"

for s in samples:
    print(f"\n{'='*60}")
    print(f"Processing {s['type']} SET: {s['name']}")
    
    with open(s['manifest']) as f:
        manifest = json.load(f)
    
    sample = manifest[s['index']]
    
    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(data_root, p)
    
    comp_img = Image.open(_abs(sample["composite"])).convert("RGB").resize(condition_size)
    real_img = Image.open(_abs(sample["real"])).convert("RGB").resize(target_size)
    prompt = sample.get("prompt", "PCB board")
    
    print(f"Name: {sample['name']}")
    print(f"Prompt: {prompt}")
    print(f"Generating with 28 steps...")
    
    condition = Condition(comp_img, adapter, position_delta=np.array([0, 0]))
    generator = torch.Generator(device=model.device)
    generator.manual_seed(42)
    
    with torch.no_grad():
        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
        )
    
    gen_img = res.images[0]
    
    # Save comparison
    W, H = target_size
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(comp_img.resize((W, H)), (0, 0))
    canvas.paste(gen_img, (W, 0))
    canvas.paste(real_img, (W * 2, 0))
    
    output_path = os.path.join(output_dir, f"{s['name']}_comparison.jpg")
    canvas.save(output_path, quality=95)
    print(f"✓ Saved: {output_path}")

print(f"\n{'='*60}")
print(f"✅ Complete! Results in: {output_dir}/")
