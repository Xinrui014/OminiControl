import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
import time

from typing import List


import prodigyopt

from omini.pipeline.flux_omini import transformer_forward, encode_images
from omini.utils.layout import bbox_to_latent_mask, build_group_mask


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer


    # inside class OminiModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        imgs = batch["image"]                         # (B=1,3,H,W), already normalized
        prompt_global = batch["description"]          # str or list[str] from your dataset
        boxes = batch.get("boxes", torch.zeros((0, 4), dtype=torch.float32))          # (K,4)
        box_prompts = batch.get("box_prompts", [])    # List[str], len K
        box_prompts = [bp[0] if isinstance(bp, (tuple, list)) else bp for bp in box_prompts]

        with torch.no_grad():
            # --- image latents & noisy mixture ---
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            x_t = ((1 - t.view(-1,1,1)) * x_0 + t.view(-1,1,1) * x_1).to(self.dtype)

            # --- global text branch ---
            prompt_embeds, pooled_prompt_embeds, text_ids = self.flux_pipe.encode_prompt(
                prompt=[prompt_global] if isinstance(prompt_global, str) else prompt_global,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )

        # --- per-box condition branches (masked tokens) ---
        num_boxes = int(boxes.shape[1])
        cond_latents_list, cond_ids_list = [], []
        if num_boxes > 0:
            blank = torch.zeros_like(imgs)  # placeholder; swap with real modality map if you have one
            with torch.no_grad():
                blank_latents, blank_ids = encode_images(self.flux_pipe, blank)

            for k in range(num_boxes):
                lm = bbox_to_latent_mask(boxes[0, k], img_ids)  # Bool (T,)
                cond_latents_list.append(blank_latents[:, lm])  # (B, Tk, C)
                cond_ids_list.append(blank_ids[lm])             # (Tk, 3)

        # --- per-box text branches ---
        box_text_embeds, box_pooled_embeds, box_text_ids = [], [], []
        for p in box_prompts:
            pe, pool, tids = self.flux_pipe.encode_prompt(
                prompt=p,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )
            box_text_embeds.append(pe)
            box_pooled_embeds.append(pool)
            box_text_ids.append(tids)

        # --- assemble branch lists ---
        text_features = [prompt_embeds] + box_text_embeds        # (1 + K)
        image_features = [x_t] + cond_latents_list               # (1 + K)
        txt_ids = [text_ids] + box_text_ids
        img_ids_list = [img_ids] + cond_ids_list

        # time/guidance/pool/adapters: lengths must match text+image branches
        timesteps = [t] * len(text_features) + [t] + [torch.zeros_like(t)] * num_boxes
        pooled = [pooled_prompt_embeds] + box_pooled_embeds + [pooled_prompt_embeds] * (1 + num_boxes)
        if self.transformer.config.guidance_embeds:
            guidances = [torch.ones_like(t).to(self.device)] * (len(text_features) + len(image_features))
        else:
            guidances = [None] * (len(text_features) + len(image_features))

        # adapters: text branches (None), image (None), conditions ("default")
        adapters = [None] * len(text_features) + [None] + ["default"] * num_boxes
        self.adapter_names = adapters
        self.adapter_set = set([a for a in adapters if a is not None])  # ensures "default" exists

        # --- attention mask across branches ---
        group_mask = build_group_mask(
            num_text=len(text_features),
            num_image=len(image_features),
            num_boxes=num_boxes,
            device=self.device,
            independent_condition=self.model_config.get("independent_condition", True),
        )

        # --- forward ---
        pred = transformer_forward(
            self.transformer,
            image_features=image_features,
            text_features=text_features,
            img_ids=img_ids_list,
            txt_ids=txt_ids,
            timesteps=timesteps,
            pooled_projections=pooled,
            guidances=guidances,
            adapters=adapters,
            return_dict=False,
            group_mask=group_mask,
        )[0]

        # --- loss ---
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        self.log_loss = loss.item() if not hasattr(self, "log_loss") else self.log_loss * 0.95 + loss.item() * 0.05
        return loss


    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0 and self.test_function:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            pl_module.eval()
            self.test_function(
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
            )
            pl_module.train()


def train(dataset, trainable_model, config, test_function):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    # run_name = time.strftime("%Y%m%d-%H%M%S")
    run_name = config.get("run_name", time.strftime("%Y%m%d-%H%M%S"))

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        callbacks = [TrainingCallback(run_name, training_config, test_function)]

    # Initialize trainer
    trainer = L.Trainer(
        # accelerator="gpu",                   # use GPUs
        # devices=training_config.get("gpus", -1),  # -1 means "all available GPUs"
        # strategy="ddp",                      # DistributedDataParallel
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)
