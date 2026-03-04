import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline
from omini.pipeline.flux_omini import Condition, encode_images, generate_instance
from omini.utils.layout import bbox_to_latent_mask

# ------------ Config ------------
CUDA_DEVICE = "cuda:3"
DTYPE       = torch.bfloat16
BASE_ID     = "black-forest-labs/FLUX.1-dev"
LORA_DIR    = "/home/xinrui/projects/OminiControl/omini/train_flux/runs/before_debug/ckpt/12000"
W, H        = 256, 256                    # a bit larger canvas often helps placement
STEPS       = 28                            # matches your default
GUIDE       = 3.5
IMG_GUIDE   = 1.0                           # >1.0 if you want stronger image-cond guidance
SEED        = 42

# Boxes (x0,y0,x1,y1 in [0,1])
boxes       = [[0.10, 0.12, 0.30, 0.35], [0.55, 0.60, 0.80, 0.90]]
box_texts   = ["red resistor", "blue capacitor"]
global_text = "A PCB with components placed exactly in the designated boxes."

# ------------ Load pipeline + LoRA ------------
pipe = FluxPipeline.from_pretrained(BASE_ID, torch_dtype=DTYPE).to(CUDA_DEVICE)
pipe.load_lora_weights(LORA_DIR, adapter_name="default")
g = torch.Generator(device=CUDA_DEVICE).manual_seed(SEED)

# ------------ Build condition masks ------------
# We create a blank image of the *target size* so the latent ids match.
blank_img = Image.new("RGB", (W, H), (0, 0, 0))
# Get ids from encoding a blank tensor of the same (H,W)
blank_tensor = torch.zeros(1, 3, H, W, device=CUDA_DEVICE, dtype=DTYPE)
_, ids = encode_images(pipe, blank_tensor)   # ids: (T,3)

conditions = []
for b in boxes:
    mask = bbox_to_latent_mask(torch.tensor(b, device=CUDA_DEVICE, dtype=torch.float32), ids)
    # Each condition carries the adapter + its latent mask; image content is blank (we only need the tokens)
    conditions.append(Condition(blank_img,"default", latent_mask=mask))

# ------------ Generate ------------
out = generate_instance(
    pipe,
    prompt=global_text,
    height=H,
    width=W,
    num_inference_steps=STEPS,
    guidance_scale=GUIDE,
    image_guidance_scale=IMG_GUIDE,
    main_adapter="default",
    conditions=conditions,
    box_texts=box_texts,              # NEW: per-box texts
    independent_condition=True,       # matches training mask behavior
    generator=g,
)
img = out.images[0]
img.save("pcb_boxes_placed.png")
print("Saved to pcb_boxes_placed.png")
