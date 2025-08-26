import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import numpy as np

from PIL import Image

# from datasets import load_dataset

from trainer_instance_control import OminiModel, get_config, train
from omini.pipeline.flux_omini import Condition, generate
from Dataloader.PCB_dataloader import TIPCBDataset
from torch.utils.data import DataLoader
import lightning as L

# @torch.no_grad()
# def test_function(model, save_path, file_name):
#     condition_size = model.training_config["dataset"]["condition_size"]
#     target_size = model.training_config["dataset"]["target_size"]
#
#     # More details about position delta can be found in the documentation.
#     position_delta = [0, -condition_size[0] // 16]
#
#     # Set adapters
#     adapter = model.adapter_names[2]
#     condition_type = model.training_config["condition_type"]
#     test_list = []
#
#     # Test case1 (in-distribution test case)
#     image = Image.open("assets/test_in.jpg")
#     image = image.resize(condition_size)
#     prompt = "Resting on the picnic table at a lakeside campsite, it's caught in the golden glow of early morning, with mist rising from the water and tall pines casting long shadows behind the scene."
#     condition = Condition(image, adapter, position_delta)
#     test_list.append((condition, prompt))
#
#     # Test case2 (out-of-distribution test case)
#     image = Image.open("assets/test_out.jpg")
#     image = image.resize(condition_size)
#     prompt = "In a bright room. It is placed on a table."
#     condition = Condition(image, adapter, position_delta)
#     test_list.append((condition, prompt))
#
#     # Generate images
#     os.makedirs(save_path, exist_ok=True)
#     for i, (condition, prompt) in enumerate(test_list):
#         generator = torch.Generator(device=model.device)
#         generator.manual_seed(42)
#
#         res = generate(
#             model.flux_pipe,
#             prompt=prompt,
#             conditions=[condition],
#             height=target_size[1],
#             width=target_size[0],
#             generator=generator,
#             model_config=model.model_config,
#             kv_cache=model.model_config.get("independent_condition", False),
#         )
#         file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
#         res.images[0].save(file_path)


def main():
    seed = int(os.environ.get("SEED", 42))
    L.seed_everything(seed, workers=True)
    # 1) Load config + set device
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # 2) Build the PCB dataset (patches + natural captions + boxes)
    ds_cfg = training_config["dataset"]
    dataset = TIPCBDataset(
        img_root=ds_cfg["img_root"],                 # e.g. "/path/to/images_top"
        csv_root=ds_cfg["csv_root"],                 # e.g. "/path/to/ann_csv_top"
        image_size=ds_cfg.get("image_size", 1024),   # base square size (pre-patch)
        max_boxes_per_data=ds_cfg.get("max_boxes_per_data", 200),
        random_crop=ds_cfg.get("random_crop", False),
        random_flip=ds_cfg.get("random_flip", True),
        grid_size=ds_cfg.get("grid_size", 16),
        use_patches=ds_cfg.get("use_patches", True), # turn on 256×256 random patches
        patch_size=ds_cfg.get("patch_size", 256),
        save_debug=ds_cfg.get("save_debug", False),  # set True to dump test.png overlays
    )

    print("Dataset length:", len(dataset))

    # 3) Init model (Flux + LoRA like OminiControl)
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    # 4) Train — pass test_function=None for now
    train(dataset, trainable_model, config, test_function=None)

if __name__ == "__main__":
    main()
