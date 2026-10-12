import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

def make_transforms(
    split="train",
    input_size=224,
    color_jitter=0.4,
    horizontal_flip=True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    transform_list = []
    
    if split == "train":
        # Random Resize and crop
        # transform_list.append(T.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3))  # 3 is bicubic
        transform_list.append(T.Resize((input_size, input_size), interpolation=3))
        
        if horizontal_flip:
            transform_list.append(T.RandomHorizontalFlip())
            
        if color_jitter > 0:
            transform_list.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=color_jitter))
    
    else:
        # Validation/Test: Resize to exact square
        transform_list.append(T.Resize((input_size, input_size), interpolation=3))

    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(mean=mean, std=std))
    
    return T.Compose(transform_list)

@dataclass
class InferenceResult:
    patch_tokens: torch.Tensor    # [N, D] raw patch tokens (CPU)
    normed_tokens: np.ndarray     # [N, D] L2-normalized
    attn_cls: Optional[np.ndarray]# [num_heads, h_feat, w_feat] CLS attention
    img_display: np.ndarray       # [H, W, 3] float 0-1
    h_feat: int
    w_feat: int
    filename: str
    pca_threshold: float = 0.0
    pca_inverted: bool = False

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    img_display = tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.clip(std * img_display + mean, 0, 1)


def _extract_patch_tokens(model, tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract patch tokens of shape [B, N, D] from `model`, supporting both
    `BaseViT.forward_features(tensor)` (returning a dict or Tensor) and
    `model(tensor, is_training=True)`.
    """
    if hasattr(model, "forward_features"):
        out = model.forward_features(tensor)
    else:
        out = model(tensor, is_training=True)

    if isinstance(out, dict):
        if "x_norm_patchtokens" in out:
            return out["x_norm_patchtokens"]
        if "patch_tokens" in out:
            return out["patch_tokens"]
        raise KeyError(
            "Model forward_features dict must contain 'x_norm_patchtokens' or 'patch_tokens'."
        )
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"Unsupported output type from model.forward_features: {type(out).__name__}")


def _extract_cls_attention(
    attn: Optional[torch.Tensor], meta: dict, h_feat: int, w_feat: int
) -> Optional[np.ndarray]:
    """
    Extract per-head CLS-to-patch attention maps of shape [num_heads, h_feat, w_feat].
    Supports full [num_heads, Seq, Seq] matrices as well as pre-sliced [num_heads, H, W]
    or [num_heads, H*W] attention tensors from custom models.
    """
    if attn is None:
        return None
    if attn.ndim == 4:
        attn = attn[0]  # [num_heads, ...]
    if attn.ndim == 3 and attn.shape[1] == h_feat and attn.shape[2] == w_feat:
        return attn.detach().cpu().numpy()
    if attn.ndim == 2 and attn.shape[1] == h_feat * w_feat:
        return attn.detach().cpu().numpy().reshape(-1, h_feat, w_feat)
    if attn.ndim == 3 and meta.get("use_cls", True):
        offset = 1 + int(meta.get("n_storage", 0))
        raw_attn = attn[:, 0, offset:].detach().cpu().numpy()
        if raw_attn.shape[-1] == h_feat * w_feat:
            return raw_attn.reshape(-1, h_feat, w_feat)
    return None


def inference_standard(model, pil_img: Image.Image, meta: dict, device: str, filename: str) -> InferenceResult:
    """Resize to training img_size, single forward pass."""
    transform = make_transforms(split="val", color_jitter=0, horizontal_flip=False,
                                input_size=meta["img_size"])
    tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        patch_tokens = _extract_patch_tokens(model, tensor)[0]
        try:
            attn = model.get_attention_map(tensor, layer_num=-1)
        except Exception:
            attn = None
            
    h_feat = tensor.shape[2] // meta["patch_size"]
    w_feat = tensor.shape[3] // meta["patch_size"]
    
    attn_cls = _extract_cls_attention(attn, meta, h_feat, w_feat)
        
    patch_tokens_cpu = patch_tokens.cpu()
    normed = F.normalize(patch_tokens_cpu.float(), p=2, dim=-1).numpy()
    
    return InferenceResult(
        patch_tokens=patch_tokens_cpu,
        normed_tokens=normed,
        attn_cls=attn_cls,
        img_display=_denormalize(tensor[0]),
        h_feat=h_feat,
        w_feat=w_feat,
        filename=filename
    )

def inference_native(model, pil_img: Image.Image, meta: dict, device: str, filename: str, max_res: Optional[int] = None) -> InferenceResult:
    """Pass full image. Resize only to ensure divisibility by patch_size."""
    native_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    W, H = pil_img.size
    patch_size = meta["patch_size"]
    
    if max_res and max(W, H) > max_res:
        scale = max_res / float(max(W, H))
        W, H = int(W * scale), int(H * scale)
        
    W = max(patch_size, (W // patch_size) * patch_size)
    H = max(patch_size, (H // patch_size) * patch_size)
    pil_img = pil_img.resize((W, H), Image.Resampling.LANCZOS)
    tensor = native_transform(pil_img).unsqueeze(0).to(device)
    
    has_patch_embed_img_size = hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size")
    if has_patch_embed_img_size:
        orig_img_size = model.patch_embed.img_size
        model.patch_embed.img_size = (H, W)
    
    try:
        with torch.no_grad():
            patch_tokens = _extract_patch_tokens(model, tensor)[0]
            try:
                attn = model.get_attention_map(tensor, layer_num=-1)
            except Exception:
                attn = None
    finally:
        if has_patch_embed_img_size:
            model.patch_embed.img_size = orig_img_size
        
    h_feat = H // patch_size
    w_feat = W // patch_size
    
    attn_cls = _extract_cls_attention(attn, meta, h_feat, w_feat)
        
    patch_tokens_cpu = patch_tokens.cpu()
    normed = F.normalize(patch_tokens_cpu.float(), p=2, dim=-1).numpy()
    
    return InferenceResult(
        patch_tokens=patch_tokens_cpu,
        normed_tokens=normed,
        attn_cls=attn_cls,
        img_display=_denormalize(tensor[0]),
        h_feat=h_feat,
        w_feat=w_feat,
        filename=filename
    )

def inference_sliding_window(model, pil_img: Image.Image, meta: dict, device: str, filename: str, target_res: Optional[int] = None) -> InferenceResult:
    """Overlapping crops. Stitching. High res support."""
    native_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    img_size = meta["img_size"]
    patch_size = meta["patch_size"]
    W, H = pil_img.size

    if target_res and max(W, H) > target_res:
        scale = target_res / float(max(W, H))
        W, H = int(W * scale), int(H * scale)

    W = max(img_size, (W // patch_size) * patch_size)
    H = max(img_size, (H // patch_size) * patch_size)
    pil_img = pil_img.resize((W, H), Image.Resampling.LANCZOS)
    tensor_img = native_transform(pil_img).unsqueeze(0).to(device)

    stride = img_size // 2
    _, _, Hp, Wp = tensor_img.shape

    pad_h = (math.ceil((Hp - img_size) / stride) * stride + img_size) - Hp if Hp > img_size else max(0, img_size - Hp)
    pad_w = (math.ceil((Wp - img_size) / stride) * stride + img_size) - Wp if Wp > img_size else max(0, img_size - Wp)
    padded = F.pad(tensor_img, (0, pad_w, 0, pad_h))
    _, _, H_pad, W_pad = padded.shape

    crops = F.unfold(padded, kernel_size=img_size, stride=stride)
    crops = crops.transpose(1, 2).reshape(-1, 3, img_size, img_size)

    crop_tokens_list = []
    with torch.no_grad():
        batch_max = 64
        for i in range(0, crops.shape[0], batch_max):
            c_batch = crops[i:i + batch_max].to(device)
            crop_tokens_list.append(_extract_patch_tokens(model, c_batch).cpu())

    crop_tokens = torch.cat(crop_tokens_list, dim=0)
    D = crop_tokens.shape[-1]

    global_h_tokens = H_pad // patch_size
    global_w_tokens = W_pad // patch_size
    global_tokens = torch.zeros(global_h_tokens, global_w_tokens, D)
    count_map = torch.zeros(global_h_tokens, global_w_tokens, 1)

    h_crops = (H_pad - img_size) // stride + 1
    w_crops = (W_pad - img_size) // stride + 1
    tokens_per_side = img_size // patch_size
    stride_tokens = stride // patch_size

    idx = 0
    for y in range(h_crops):
        for x in range(w_crops):
            ty = y * stride_tokens
            tx = x * stride_tokens
            p_tok = crop_tokens[idx].view(tokens_per_side, tokens_per_side, D)
            global_tokens[ty:ty + tokens_per_side, tx:tx + tokens_per_side] += p_tok
            count_map[ty:ty + tokens_per_side, tx:tx + tokens_per_side] += 1
            idx += 1

    global_tokens /= count_map
    valid_h = Hp // patch_size
    valid_w = Wp // patch_size
    global_tokens = global_tokens[:valid_h, :valid_w].reshape(valid_h * valid_w, D)
    
    normed = F.normalize(global_tokens.float(), p=2, dim=-1).numpy()
    
    # Attention is too complex to reconstruct for sliding window correctly without weird artifacts,
    # so we leave it as None for now
    
    return InferenceResult(
        patch_tokens=global_tokens,
        normed_tokens=normed,
        attn_cls=None,
        img_display=_denormalize(tensor_img[0]),
        h_feat=valid_h,
        w_feat=valid_w,
        filename=filename
    )
