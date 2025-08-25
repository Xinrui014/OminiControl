import torch

def bbox_to_latent_mask(box_xyxy_norm: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """
    box_xyxy_norm: [x0,y0,x1,y1] in [0,1] for the *current* image/patch.
    ids: (T,3) token indices from encode_images (cols=[b,y,x]).
    returns: Bool mask over T (tokens inside the box).
    """
    device = ids.device
    b = torch.as_tensor(box_xyxy_norm, dtype=torch.float32, device=device).flatten()
    if b.numel() == 0:
        return torch.zeros(ids.shape[0], dtype=torch.bool, device=device)

    # wrong size? -> hard error so you can catch it
    if b.numel() != 4:
        raise ValueError(f"bbox_to_latent_mask expected 4 numbers, got shape {tuple(b.shape)} / value={b}")

    H = ids[:, 1].amax().add(1).to(torch.float32)  # tensor scalar
    W = ids[:, 2].amax().add(1).to(torch.float32)

    x0 = torch.floor(b[0] * W).clamp(min=0)
    y0 = torch.floor(b[1] * H).clamp(min=0)
    x1 = torch.ceil (b[2] * W).clamp(min=0, max=W)
    y1 = torch.ceil (b[3] * H).clamp(min=0, max=H)

    x0 = x0.to(torch.long); y0 = y0.to(torch.long)
    x1 = x1.to(torch.long); y1 = y1.to(torch.long)

    x = ids[:, 2].to(torch.long)
    y = ids[:, 1].to(torch.long)

    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

def build_group_mask(num_text: int, num_image: int, num_boxes: int, device, independent_condition: bool) -> torch.Tensor:
    """
    Branch order we’ll use:
      text:  [ global_text, box_text_1..box_text_K ]   -> len = 1 + K
      image: [ image_xt,  cond_box_1..cond_box_K ]     -> len = 1 + K
    """
    n = num_text + num_image
    gm = torch.ones((n, n), dtype=torch.bool, device=device)
    start_img = num_text
    # cond<->cond diagonal only
    for i in range(num_boxes):
        r = start_img + 1 + i
        for j in range(num_boxes):
            c = start_img + 1 + j
            gm[r, c] = (i == j)
    # bind each box text to its own cond branch
    for i in range(num_boxes):
        t_idx = 1 + i
        for j in range(num_boxes):
            c_idx = start_img + 1 + j
            gm[t_idx, c_idx] = (i == j)
    if independent_condition:
        gm[start_img + 1 :, :num_text] = False   # cond -> any text blocked
    # print(gm)
    return gm
