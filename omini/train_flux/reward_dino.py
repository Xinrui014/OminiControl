"""DINOv2-based local classification reward for AlignProp.

Loads pad20_transfix best.pt (DINOv2 ViT-B/14 + linear head for 9 PCB classes).
For each GT (bbox, class) crop of the generated image, compute -CE loss.
Reward = mean of -CE across components (higher = better).
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


CAT_NAMES = ["Resistor","Capacitor","Inductor","Connector","Diode","Switch","Transistor","IC","Oscillator"]
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


class DinoClassifier(nn.Module):
    """Matches train_dino_cls_transfix.Classifier structure (DINOv2 ViT-B/14 + Linear(768, 9))."""
    def __init__(self, backbone_name: str = "dinov2_vitb14", num_cat: int = 9, unfreeze_last: int = 4):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, trust_repo=True)
        dim = self.backbone.embed_dim
        for p in self.backbone.parameters(): p.requires_grad = False
        if unfreeze_last > 0:
            for blk in self.backbone.blocks[-unfreeze_last:]:
                for p in blk.parameters(): p.requires_grad = True
            for p in self.backbone.norm.parameters(): p.requires_grad = True
        self.cat_head = nn.Linear(dim, num_cat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)["x_norm_clstoken"]
        return self.cat_head(feat)


class DinoLocalReward(nn.Module):
    """DINOv2 classifier on known bbox crops of the generated image.

    Usage:
        reward_model = DinoLocalReward("path/to/best.pt", device="cuda")
        # image: (1,3,H,W) in [-1, 1] from VAE (grad-enabled)
        # bboxes: list of (x1,y1,x2,y2) in pixel coords at (H, W)
        # classes: list of int class indices (0..8)
        reward, per_class = reward_model(image, bboxes, classes)  # scalar, dict
    """
    def __init__(self, ckpt_path: str, device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model = DinoClassifier()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inner = sd["state_dict"] if "state_dict" in sd else sd
        missing, unexpected = self.model.load_state_dict(inner, strict=False)
        if missing: print(f"[DinoLocalReward] missing keys: {len(missing)}")
        if unexpected: print(f"[DinoLocalReward] unexpected keys: {len(unexpected)}")
        self.model.to(device=device, dtype=dtype).eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.dtype = dtype
        self.device = device
        self.register_buffer("mean", torch.tensor(IMG_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMG_STD,  device=device, dtype=dtype).view(1, 3, 1, 1))
        self.register_buffer("grey_pad", torch.tensor(114.0 / 127.5 - 1.0,
                                                      device=device, dtype=dtype))  # PIL (114,114,114) in [-1,1]

    def _square_pad(self, crop: torch.Tensor) -> torch.Tensor:
        """Pad crop to square with grey fill (matches training preprocessing)."""
        _, _, ch, cw = crop.shape
        side = max(cw, ch)
        pad_h_top  = (side - ch) // 2
        pad_h_bot  = side - ch - pad_h_top
        pad_w_left = (side - cw) // 2
        pad_w_right = side - cw - pad_w_left
        if pad_h_top or pad_h_bot or pad_w_left or pad_w_right:
            crop = F.pad(crop, (pad_w_left, pad_w_right, pad_h_top, pad_h_bot),
                         value=self.grey_pad.item())
        return crop

    def _prepare_crop(self, image_m1p1: torch.Tensor, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        """Crop image at bbox, square-pad, resize to 224, normalize for DINOv2."""
        x1, y1, x2, y2 = bbox
        crop = image_m1p1[:, :, y1:y2, x1:x2]  # (1, 3, h, w), grad flows through
        crop = self._square_pad(crop)
        crop = F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False)
        # [-1, 1] -> [0, 1] -> ImageNet norm
        crop01 = (crop + 1.0) / 2.0
        crop_norm = (crop01 - self.mean) / self.std
        return crop_norm.to(self.dtype)

    def forward(
        self,
        image_m1p1: torch.Tensor,
        bboxes: List[Tuple[int, int, int, int]],
        classes: List[int],
        class_weights: "Optional[Dict[str, float]]" = None,
        return_per_class: bool = False,
        return_per_component: bool = False,
    ):
        """Returns (reward, per_class_dict, per_component_dict).

        reward:    scalar tensor — weighted mean log-p over components (higher = better)
        per_class: dict {class_name: mean log-p for that class} (logging, detached)
        per_component: dict with {'logp': List[float], 'classes': List[int], 'weights': List[float]}
                       (logging, detached) — None unless return_per_component=True

        class_weights: {class_name: float}. If provided, reward =
                       Σ(w_i * logp_i) / Σ(w_i) (weighted mean).
                       If None, uniform weighting (= current behavior).
        """
        assert len(bboxes) == len(classes), f"{len(bboxes)} bboxes vs {len(classes)} classes"
        if len(bboxes) == 0:
            zero = torch.zeros(1, device=self.device, dtype=self.dtype)
            return zero, {}, ({"logp": [], "classes": [], "weights": []} if return_per_component else None)

        crops = [self._prepare_crop(image_m1p1, b) for b in bboxes]
        batch = torch.cat(crops, dim=0)                     # (N, 3, 224, 224)
        logits = self.model(batch)                           # (N, 9)
        logp = F.log_softmax(logits, dim=-1)                 # (N, 9)
        class_idx = torch.tensor(classes, device=self.device, dtype=torch.long)
        chosen = logp.gather(1, class_idx.unsqueeze(1)).squeeze(1)   # (N,)

        # Per-component weights (B)
        if class_weights is not None:
            w_list = [float(class_weights.get(CAT_NAMES[c], 1.0)) for c in classes]
            w = torch.tensor(w_list, device=self.device, dtype=chosen.dtype)
            reward = (w * chosen).sum() / w.sum()
        else:
            w_list = [1.0] * len(classes)
            reward = chosen.mean()

        per_class = {}
        if return_per_class:
            for ci in set(classes):
                mask = class_idx == ci
                if mask.any():
                    per_class[CAT_NAMES[ci]] = chosen[mask].mean().item()

        per_component = None
        if return_per_component:
            per_component = {
                "logp": chosen.detach().cpu().tolist(),
                "classes": list(classes),
                "weights": w_list,
            }
        return reward, per_class, per_component
