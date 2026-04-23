import os
import torch

os.environ['OMINI_CONFIG'] = 'train/config/pcb_harmonize_v3_newdata_cluster.yaml'

from omini.train_flux.trainer import OminiModel, get_config, train
from omini.train_flux.train_pcb_v3_newdata import PCBHarmonizeDatasetV3
from lib.component_bank_v2_new import ComponentBankV2_new

config = get_config()
config['train']['max_steps'] = 5
config['train']['save_interval'] = 999
config['train']['sample_interval'] = 999
config['train']['batch_size'] = 1
config['train']['accumulate_grad_batches'] = 1
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

dc = config['train']['dataset']
bank = ComponentBankV2_new(anno_dir=dc['anno_dir'], image_dir=dc['image_dir'])
ds = PCBHarmonizeDatasetV3(
    anno_dir=dc['anno_dir'], image_dir=dc['image_dir'], component_bank=bank,
    condition_size=(1024, 1024), target_size=(1024, 1024),
)
model = OminiModel(
    flux_pipe_id=config['flux_path'], lora_config=config['train'].get('lora_config'),
    device='cuda', dtype=torch.bfloat16, optimizer_config=config['train']['optimizer'],
    model_config=config.get('model', {}), gradient_checkpointing=True,
)
train(ds, model, config, test_function=None)
