import os
import torch

os.environ.setdefault('OMINI_CONFIG', 'train/config/pcb_harmonize_v3_newdata_cluster.yaml')
os.environ.setdefault('WANDB_API_KEY', 'a5ebf533c17c677bcee36f66c91907b5fb102f7c')
os.environ.setdefault('WANDB_DIR', '/projects/_ssd/xrssd/runs')
os.environ.setdefault('HF_HOME', '/projects/_ssd/xrssd/checkpoints/hf_cache')

from omini.train_flux.trainer import OminiModel, get_config, train
from omini.train_flux.train_pcb_v3_newdata import PCBHarmonizeDatasetV3, test_function
from lib.component_bank_v2_new import ComponentBankV2_new

config = get_config()
config['train']['max_steps'] = 8000
config['train']['save_interval'] = 1000
config['train']['sample_interval'] = 1000
config['train']['batch_size'] = 2
config['train']['accumulate_grad_batches'] = 2
config['train']['save_path'] = '/projects/_ssd/xrssd/OminiControl/runs/v3.1_newdata_1024'
config['train']['dataset']['target_size'] = [1024, 1024]
config['train']['dataset']['condition_size'] = [1024, 1024]

torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

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
train(ds, model, config, test_function=test_function)
