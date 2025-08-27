import os, yaml, argparse
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from diffusers.pipelines import FluxPipeline
from omini.pipeline.flux_omini import Condition, generate, encode_images
from omini.utils.layout import bbox_to_latent_mask
from omini.train_flux.Dataloader.PCB_dataloader import TIPCBDataset
import lightning as L


def build_dataset(config, img_root_override=None):
    ds_cfg = config["train"]["dataset"].copy()
    # if img_root_override is not None:
    #     ds_cfg["img_root"] = img_root_override

    dataset = TIPCBDataset(
        img_root=ds_cfg["img_root"],
        csv_root=ds_cfg["csv_root"],
        image_size=ds_cfg.get("image_size", 1024),
        max_boxes_per_data=ds_cfg.get("max_boxes_per_data", 200),
        random_crop=ds_cfg.get("random_crop", False),
        random_flip=ds_cfg.get("random_flip", True),
        grid_size=ds_cfg.get("grid_size", 16),
        use_patches=ds_cfg.get("use_patches", True),
        patch_size=ds_cfg.get("patch_size", 256),
        save_debug=False,
    )
    return dataset, ds_cfg


@torch.no_grad()
def sample_to_io(sample):
    """
    Convert a TIPCBDataset sample to (boxes, box_prompts, prompt, real_patch PIL).
    """
    # sample["image"]: tensor normalized to [-1, 1] -> convert back to [0,1] and PIL
    img = (sample["image"].clamp(-1, 1) + 1.0) / 2.0
    real_patch = T.ToPILImage()(img.cpu()).convert("RGB")

    boxes = sample.get("boxes", torch.empty(0, 4))
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu()
    box_prompts = sample.get("box_prompts", [])
    prompt = sample.get("description", "")
    return boxes, box_prompts, prompt, real_patch


def build_conditions_from_boxes(pipe, boxes, target_size):
    """
    Create latent masks & Condition objects for each box.
    """
    if boxes is None or len(boxes) == 0:
        return None  # no constraints

    # We need encoder ids for the latent grid space
    w, h = target_size
    blank_latents = torch.zeros(1, 3, h, w)
    _, ids = encode_images(pipe, blank_latents)

    blank_img = Image.new("RGB", (w, h))
    conditions = []
    for box in boxes:
        # box is normalized [0,1]: (x0, y0, x1, y1)
        b = torch.tensor(box)
        mask = bbox_to_latent_mask(b, ids)
        conditions.append(Condition(blank_img, "default", latent_mask=mask))
    return conditions


def draw_boxes_on_image(pil_img, boxes, color=(255, 0, 0), width=2):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for b in boxes:
        x0, y0, x1, y1 = (b.tolist() if hasattr(b, "tolist") else list(b))
        draw.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], outline=color, width=width)
    return img


def infer_folder(
        # input_dir: str,
        save_dir: str,
        config_path: str,
        lora_ckpt: str,
        device: str = "cuda:0",
        seed: int = 42,
        target_size=(256, 256),
        limit: int | None = None,
):
    os.makedirs(save_dir, exist_ok=True)

    # Seed
    # L.seed_everything(int(os.environ.get("SEED", seed)), workers=True)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Dataset bound to input_dir
    dataset, ds_cfg = build_dataset(config)

    # Load FLUX + LoRA
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to(device)
    pipe.load_lora_weights(lora_ckpt, adapter_name="default")

    total = len(dataset) if limit is None else min(limit, len(dataset))
    # print(f"[Info] Running inference on {total} items from: {input_dir}")

    for idx in range(total):
        sample = dataset[idx]
        # Prefer basename from sample if provided, else index
        # TIPCBDataset usually includes path; adjust if your key name differs
        img_path = sample.get("img_path") or sample.get("path") or None
        stem = (
            os.path.splitext(os.path.basename(img_path))[0]
            if img_path
            else f"item_{idx:05d}"
        )

        boxes, box_prompts, prompt, real_patch = sample_to_io(sample)
        boxes       = [[0.10, 0.12, 0.30, 0.35], [0.55, 0.60, 0.80, 0.90]]
        box_prompts = ["red Resistor", "blue Capacitor"]
        prompt      = "PCB board with components: " + ", ".join(box_prompts)
        conditions = build_conditions_from_boxes(pipe, boxes, target_size=target_size)

        # Generate
        result = generate(
            pipe,
            prompt=prompt,
            conditions=conditions,
            height=target_size[1],
            width=target_size[0],
            # model_config={'independent_condition': True},
        )
        synth = result.images[0]

        # Save outputs
        synth_path = os.path.join(save_dir, f"{stem}_synth.jpg")
        # real_path = os.path.join(save_dir, f"{stem}_real.jpg")
        overlay_path = os.path.join(save_dir, f"{stem}_real_with_boxes.jpg")

        synth.save(synth_path)
        # real_patch.save(real_path)
        overlay_img = draw_boxes_on_image(real_patch, boxes if boxes is not None else [])
        overlay_img.save(overlay_path)

        print(
            f"[OK] {idx+1}/{total}  Saved:\n"
            f"     synth:   {synth_path}\n"
            # f"     real:    {real_path}\n"
            f"     overlay: {overlay_path}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference for a folder of images.")
    # p.add_argument("--input_dir", required=True, help="Folder of input images.")
    p.add_argument("--save_dir", default="/home/xinrui/projects/OminiControl/omini/train_flux/runs/before_debug/test_natural", help="Where to save results.")
    p.add_argument(
        "--config_path",
        default="/home/xinrui/projects/OminiControl/train/config/instance_control.yaml",
        help="Path to training/inference YAML config.",
    )
    p.add_argument(
        "--lora_ckpt",
        default="/home/xinrui/projects/OminiControl/omini/train_flux/runs/before_debug/ckpt/12000",
        help="Path to LoRA weights directory (adapter).",
    )
    p.add_argument("--device", default="cuda:3", help="CUDA device string, e.g. cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--limit", type=int, default=1, help="Max number of items.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_folder(
        # input_dir=args.input_dir,
        save_dir=args.save_dir,
        config_path=args.config_path,
        lora_ckpt=args.lora_ckpt,
        device=args.device,
        seed=args.seed,
        target_size=(args.width, args.height),
        limit=args.limit,
    )
