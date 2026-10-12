# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
import re
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Callable

import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn
from models.base import BaseViT
from models.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN

logger: logging.Logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    n_prefix: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate standard 2D sine-cosine positional embeddings (as used in original ViT / MAE / MoCo-v3).
    Returns tensor of shape [1, n_prefix + grid_h * grid_w, embed_dim].
    """
    assert embed_dim % 2 == 0, "embed_dim must be divisible by 2 for sinusoidal positional embedding"
    half_dim = embed_dim // 2

    def _get_1d_sincos(dim: int, pos: Tensor) -> Tensor:
        omega = torch.arange(dim // 2, device=device, dtype=torch.float32)
        omega /= dim / 2.0
        omega = 1.0 / (10000.0**omega)  # [dim // 2]
        out = torch.einsum("m,d->md", pos.reshape(-1), omega)  # [M, dim // 2]
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [M, dim]
        if emb.shape[1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[1]))
        return emb

    grid_y = torch.arange(grid_h, device=device, dtype=torch.float32)
    grid_x = torch.arange(grid_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    emb_h = _get_1d_sincos(half_dim, yy)
    emb_w = _get_1d_sincos(embed_dim - half_dim, xx)
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # [grid_h * grid_w, embed_dim]

    if n_prefix > 0:
        prefix = torch.zeros(n_prefix, embed_dim, device=device, dtype=torch.float32)
        pos_embed = torch.cat([prefix, pos_embed], dim=0)

    return pos_embed.unsqueeze(0).to(dtype=dtype)


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class VisionTransformer(BaseViT):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_type: Literal["rope", "learnable", "sinusoidal", "none"] = "rope",
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        use_cls_token: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        patch_embed_norm: bool = True,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        verbose: int = ignored_kwargs.get("verbose", 0)
        self.logger = logger

        if len(ignored_kwargs) > 0 and verbose > 0:
            logger.log(msg=f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.pos_embed_type = pos_embed_type

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
            norm_layer=norm_layer_cls if patch_embed_norm else None,
        )

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, device=device))
        else:
            self.cls_token = None
            

        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, embed_dim, device=device))

        num_prefix = (1 if self.use_cls_token else 0) + self.n_storage_tokens
        num_patches = self.patch_embed.num_patches
        grid_h, grid_w = self.patch_embed.patches_resolution

        if pos_embed_type == "rope":
            self.pos_embed = None
            if verbose > 0:
                logger.log(msg=f"using base={pos_embed_rope_base} for rope new")
                logger.log(msg=f"using min_period={pos_embed_rope_min_period} for rope new")
                logger.log(msg=f"using max_period={pos_embed_rope_max_period} for rope new")
                logger.log(msg=f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
                logger.log(msg=f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
                logger.log(msg=f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
                logger.log(msg=f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
                logger.log(msg=f"using dtype={pos_embed_rope_dtype} for rope new")
            self.rope_embed = RopePositionEmbedding(
                embed_dim=embed_dim,
                num_heads=num_heads,
                base=pos_embed_rope_base,
                min_period=pos_embed_rope_min_period,
                max_period=pos_embed_rope_max_period,
                normalize_coords=pos_embed_rope_normalize_coords,
                shift_coords=pos_embed_rope_shift_coords,
                jitter_coords=pos_embed_rope_jitter_coords,
                rescale_coords=pos_embed_rope_rescale_coords,
                dtype=dtype_dict[pos_embed_rope_dtype],
                device=device,
            )
        elif pos_embed_type == "learnable":
            self.rope_embed = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_prefix + num_patches, embed_dim, device=device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal":
            self.rope_embed = None
            sincos = build_2d_sincos_pos_embed(
                embed_dim=embed_dim,
                grid_h=grid_h,
                grid_w=grid_w,
                n_prefix=num_prefix,
                device=device,
            )
            self.pos_embed = nn.Parameter(sincos, requires_grad=False)
        else:
            self.rope_embed = None
            self.pos_embed = None

        if verbose > 0:
            logger.log(msg=f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, device=device))

    def init_weights(self):
        if self.rope_embed is not None:
            self.rope_embed._init_weights()
        if self.pos_embed is not None and self.pos_embed_type == "learnable":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def interpolate_pos_encoding(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Interpolate or generate 2D positional encodings for a patch grid of size (H, W).
        Handles prefix tokens (CLS + storage/register tokens) and supports both
        learnable and 2D sinusoidal positional embeddings.
        """
        num_prefix = (1 if self.use_cls_token else 0) + self.n_storage_tokens
        N = H * W

        if self.pos_embed_type == "sinusoidal" and (
            self.pos_embed is None or self.pos_embed.shape[1] != num_prefix + N
        ):
            return build_2d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                grid_h=H,
                grid_w=W,
                n_prefix=num_prefix,
                device=x.device,
                dtype=x.dtype,
            )

        if self.pos_embed is None:
            return torch.zeros(1, num_prefix + N, self.embed_dim, device=x.device, dtype=x.dtype)

        pos_embed = self.pos_embed.float()
        L = pos_embed.shape[1]
        D = pos_embed.shape[2]

        # Determine how many prefix tokens are stored inside self.pos_embed
        if L >= num_prefix and int(math.isqrt(L - num_prefix)) ** 2 == (L - num_prefix):
            n_pos_prefix = num_prefix
        elif L >= 1 and int(math.isqrt(L - 1)) ** 2 == (L - 1):
            n_pos_prefix = 1
        elif int(math.isqrt(L)) ** 2 == L:
            n_pos_prefix = 0
        else:
            n_pos_prefix = min(num_prefix, L)

        prefix_pos = pos_embed[:, :n_pos_prefix, :]
        patch_pos = pos_embed[:, n_pos_prefix:, :]

        if patch_pos.shape[1] != N:
            M = int(math.isqrt(patch_pos.shape[1]))
            patch_pos = patch_pos.reshape(1, M, M, D).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(H, W), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, N, D)

        if n_pos_prefix < num_prefix:
            pad = torch.zeros(1, num_prefix - n_pos_prefix, D, device=pos_embed.device, dtype=pos_embed.dtype)
            prefix_pos = torch.cat([prefix_pos, pad], dim=1)
        elif n_pos_prefix > num_prefix:
            prefix_pos = prefix_pos[:, :num_prefix, :]

        full_pos = torch.cat([prefix_pos, patch_pos], dim=1)
        return full_pos.to(device=x.device, dtype=x.dtype)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x) # Conv + Norm to convert into patch embeddings -> (B, H, W, C) 
        B, H, W, _ = x.shape
        x = x.flatten(1, 2) # Flattening the spatial dimensions -> (B, HW, C)

        if masks is not None:
            # If masks are provided, replace the masked patches with mask tokens
            x = torch.where(
                # When condition is True, replace with input, otherwise keep other
                condition = masks.unsqueeze(-1), # (B, HW, 1)
                input = self.mask_token.to(x.dtype).unsqueeze(0), # (1, 1, C) where C is the embed_dim
                other = x # (B, HW, C)
            )
            if self.use_cls_token:
                cls_token = self.cls_token
        else:
            if self.use_cls_token:
                # This is a common PyTorch trick for Distributed Data Parallel (DDP) training. It ensures that the self.mask_token is still included in the autograd computation graph even when masking is disabled. Without this trick, DDP throws an error because there would be a learnable parameter with no gradients attached to it during the forward pass.
                cls_token = self.cls_token + 0 * self.mask_token
                
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            # If no storage tokens, create empty tensor to simplify concat logic
            ref_token = self.cls_token if self.use_cls_token else self.mask_token
            storage_tokens = torch.empty(
                1,
                0,
                x.shape[-1],
                dtype=ref_token.dtype,
                device=ref_token.device,
            )

        # Build list of token blocks to prepend to patch tokens
        prefix_tokens = []
        if self.use_cls_token:
            prefix_tokens.append(cls_token.expand(B, -1, -1))
        prefix_tokens.append(storage_tokens.expand(B, -1, -1))
        
        x = torch.cat(prefix_tokens + [x], dim=1)

        if self.pos_embed_type in ("learnable", "sinusoidal"):
            x = x + self.interpolate_pos_encoding(x, H, W)

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        offset = 1 if self.use_cls_token else 0
        
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + offset])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + offset])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + offset])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + offset :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + offset]
                x_norm_patch = x_norm[:, self.n_storage_tokens + offset :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0] if self.use_cls_token else None,
                    "x_storage_tokens": x_norm_cls_reg[:, offset:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "x_cls_token": x[:, 0] if self.use_cls_token else None,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
            
    def forward_jepa_encoder(self, x: Tensor, masks_enc: List[Tensor], H: int, W: int) -> Tensor:
        """
        Native JEPA Context Encoder forward pass.
        Expects a raw image, extracts only the visible patches specified by masks_enc,
        and pushes ONLY the visible tokens through the transformer to compute self-attention.
        
        Args:
            x: [B, C, H_img, W_img] raw image.
            masks_enc: list containing a [B, K_enc] tensor with indices of visible patches.
            H, W: Dimensions of the original patch grid (e.g., 14, 14).
            
        Returns:
            Dict containing:
                "x_norm_patchtokens": [B, K_enc, D] visible encodings
                "x_norm_clstoken": [B, 1, D] cls token
                "x_storage_tokens": [B, S, D] storage tokens
        """
        
        # 1. Patchify the full image (just the conv projection, no attention)
        x = self.patch_embed(x)  # [B, H, W, D]
        B = x.shape[0]
        x = x.flatten(1, 2)      # [B, N, D]
        
        # 2. Extract only the visible tokens
        # apply_masks expects a list of masks, and returns the gathered sequence
        x_visible = apply_masks(x, masks_enc)  # [B, K_enc, D]
        
        # 3. Get full [HW, D] RoPE embeddings
        if self.rope_embed is not None:
            sin, cos = self.rope_embed(H=H, W=W)
            
            # RoPE returns shape [HW, D] -- Expand to [B, HW, D]
            sin = sin.unsqueeze(0).expand(B, -1, -1)
            cos = cos.unsqueeze(0).expand(B, -1, -1)
            
            # Gather the exact sin/cos vectors matching the visible patches
            visible_indices = masks_enc[0].unsqueeze(-1).expand(-1, -1, sin.shape[-1])
            sin_gathered = torch.gather(sin, 1, visible_indices)
            cos_gathered = torch.gather(cos, 1, visible_indices)
            
            # Unsqueeze the heads dimension so it broadcasts across [B, num_heads, seq_len, D_head]
            rope_generated = (sin_gathered.unsqueeze(1), cos_gathered.unsqueeze(1))
        else:
            rope_generated = None
            
        # 4. Standard structural tokens setup
        if self.use_cls_token:
            cls_token = self.cls_token + 0 * self.mask_token
            
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            ref_token = self.cls_token if self.use_cls_token else self.mask_token
            storage_tokens = torch.empty(
                1, 0, ref_token.shape[-1],
                dtype=ref_token.dtype, device=ref_token.device,
            )
            
        # Prepend struct tokens to sequence
        prefix_tokens = []
        if self.use_cls_token:
            prefix_tokens.append(cls_token.expand(B, -1, -1))
        prefix_tokens.append(storage_tokens.expand(B, -1, -1))
        
        x_visible = torch.cat(prefix_tokens + [x_visible], dim=1)
        
        # Struct tokens get zero RoPE embeddings (handled natively in apply_rope inside SelfAttention)
        rope = rope_generated        
        
        # 5. Run blocks
        x_list = [x_visible]
        for blk in self.blocks:
            x_list = blk(x_list, [rope])
        x_out = x_list[0]
        
        # 6. Normalize
        x_norm = self.norm(x_out)
        
        # 7. Isolate the structure and patch tokens
        offset = 1 if self.use_cls_token else 0
        cls_token_norm = x_norm[:, 0:1, :] if self.use_cls_token else None
        cls_token = x_out[:, 0:1, :] if self.use_cls_token else None
        storage_tokens_norm = x_norm[:, offset : self.n_storage_tokens + offset, :]
        storage_tokens = x_out[:, offset : self.n_storage_tokens + offset, :]
        x_enc_visible = x_norm[:, self.n_storage_tokens + offset :, :]
        
        return {
            "x_norm_patchtokens": x_enc_visible,
            "x_norm_clstoken": cls_token_norm,
            "x_norm_storage_tokens": storage_tokens_norm,
            "x_cls_token": cls_token,
            "x_storage_tokens": storage_tokens,
        }
    
    def forward_jepa_predictor(self, x_enc_visible: Tensor, masks_enc: Tensor, masks_pred: List[Tensor], H: int, W: int, return_all_patches: bool = False) -> Tensor:
        """
        Native JEPA Predictor forward pass.
        Takes visible encoder tokens and appends TARGET MASK tokens.
        Correctly gathers Rotary Position Embeddings (RoPE) for all scattered patches.
        
        Args:
            x_enc_visible: [B, K_enc, D] visible tokens from context encoder.
            masks_enc: [B, K_enc] indices of the visible patches.
            masks_pred: list of [B, K_pred] indices for the target patches (can be empty).
            H, W: Dimensions of the original patch grid (e.g., 14, 14).
            return_all_patches: If True, reorders all predicted tokens back to the full [B, H*W, D] grid.
            
        Returns:
            Dict containing:
                "x_norm_patchtokens": [B, H*W, D] if return_all_patches else [B, sum(K_pred), D]
                "x_norm_clstoken": [B, 1, D] cls token
                "x_storage_tokens": [B, S, D] storage tokens
        """
        B, K_enc, D = x_enc_visible.shape
        
        # Combine all prediction masks into one tensor if non-empty
        valid_masks_pred = [m for m in masks_pred if m is not None and m.shape[1] > 0]
        if len(valid_masks_pred) > 0:
            masks_pred_cat = torch.cat(valid_masks_pred, dim=1)  # [B, total_K_pred]
            K_pred_total = masks_pred_cat.shape[1]
            mask_tokens = self.mask_token.expand(B, K_pred_total, -1)  # [B, total_K_pred, D]
            sequence = torch.cat([x_enc_visible, mask_tokens], dim=1)  # [B, K_enc + total_K_pred, D]
            all_indices = torch.cat([masks_enc, masks_pred_cat], dim=1)  # [B, K_enc + total_K_pred]
        else:
            K_pred_total = 0
            sequence = x_enc_visible
            all_indices = masks_enc
        
        # 3. Get full [HW, D] RoPE embeddings
        if self.rope_embed is not None:
            sin, cos = self.rope_embed(H=H, W=W)
            
            # RoPE returns shape [HW, D] -- Expand to [B, HW, D]
            sin = sin.unsqueeze(0).expand(B, -1, -1)
            cos = cos.unsqueeze(0).expand(B, -1, -1)
            
            # Gather the exact sin/cos vectors using the provided patch indices
            indices_expanded = all_indices.unsqueeze(-1).expand(-1, -1, sin.shape[-1])
            sin_gathered = torch.gather(sin, 1, indices_expanded)
            cos_gathered = torch.gather(cos, 1, indices_expanded)
            
            # Unsqueeze the heads dimension so it broadcasts across [B, num_heads, seq_len, D_head]
            rope_generated = (sin_gathered.unsqueeze(1), cos_gathered.unsqueeze(1))
        else:
            rope_generated = None
            
        # 4. Standard structural tokens setup
        if self.use_cls_token:
            cls_token = self.cls_token + 0 * self.mask_token
            
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            ref_token = self.cls_token if self.use_cls_token else self.mask_token
            storage_tokens = torch.empty(
                1, 0, ref_token.shape[-1],
                dtype=ref_token.dtype, device=ref_token.device,
            )
            
        # Prepend struct tokens to sequence
        prefix_tokens = []
        if self.use_cls_token:
            prefix_tokens.append(cls_token.expand(B, -1, -1))
        prefix_tokens.append(storage_tokens.expand(B, -1, -1))
        
        x = torch.cat(prefix_tokens + [sequence], dim=1)
        
        # Struct tokens get zero RoPE embeddings (handled natively in apply_rope inside SelfAttention)
        rope = rope_generated        # 5. Run blocks
        x = [x]
        for blk in self.blocks:
            x = blk(x, [rope])
        x = x[0]
        
        # 6. Normalize
        x_norm = self.norm(x)
        
        # 7. Isolate the structure and target token predictions and return them!
        offset = 1 if self.use_cls_token else 0
        cls_token_norm = x_norm[:, 0:1, :] if self.use_cls_token else None
        cls_token = x[:, 0:1, :] if self.use_cls_token else None
        storage_tokens_norm = x_norm[:, offset : self.n_storage_tokens + offset, :]
        storage_tokens = x[:, offset : self.n_storage_tokens + offset, :]
        
        if return_all_patches:
            all_patches = x_norm[:, offset + self.n_storage_tokens :]
            # Reconstruct original spatial order [B, H*W, D]
            full_patches = torch.empty(B, H * W, all_patches.shape[-1], device=all_patches.device, dtype=all_patches.dtype)
            full_patches.scatter_(1, all_indices.unsqueeze(-1).expand(-1, -1, all_patches.shape[-1]), all_patches)
            h_pred = full_patches
        else:
            h_pred = x_norm[:, -K_pred_total:, :] if K_pred_total > 0 else x_norm[:, offset + self.n_storage_tokens :]
        
        return {
            "x_norm_patchtokens": h_pred,
            "x_norm_clstoken": cls_token_norm,
            "x_norm_storage_tokens": storage_tokens_norm,
            "x_cls_token": cls_token,
            "x_storage_tokens": storage_tokens,
        }
    
    def get_attention_map(self, x, layer_num:int = -1):
        """
        Get the attention map for a specific layer. -1 means the last attention layer.
        
        Returns:
            attention_map: [B, num_heads, N, N] attention weights (post-softmax)
        """
        
        # Adjust layer_num for negative indexing
        num_blocks = len(self.blocks)
        if layer_num < 0:
            layer_num += num_blocks
        
        if not (0 <= layer_num < num_blocks):
             raise ValueError(f"layer_num {layer_num} is out of bounds for model with {num_blocks} blocks")

        qkv_output = []
        
        def hook(module, input, output):
            # output of qkv linear: [B, N, 3 * num_heads * head_dim]
            qkv_output.append(output)

        # Hook on the qkv projection to capture Q, K, V
        target_layer = self.blocks[layer_num].attn.qkv
        handle = target_layer.register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                self(x, is_training=True)
        finally:
            handle.remove()
            
        if not qkv_output:
            raise RuntimeError("Failed to capture QKV output. Check model structure.")
        
        # Compute attention from Q and K
        qkv = qkv_output[0]  # [B, N, 3 * C]
        B, N, _ = qkv.shape
        num_heads = self.blocks[layer_num].attn.num_heads
        head_dim = qkv.shape[-1] // (3 * num_heads)
        
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
        q, k, _ = torch.unbind(qkv, 2)  # each [B, N, num_heads, head_dim]
        q = q.transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.transpose(1, 2)  # [B, num_heads, N, head_dim]
        
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        
        return attn

    @classmethod
    def infer_config_from_state_dict(cls, state_dict: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Automatically infer VisionTransformer initialization parameters directly
        from a cleaned checkpoint state_dict.
        """
        cfg: Dict[str, Any] = {}

        # 1. embed_dim, in_chans, patch_size from patch_embed.proj.weight
        if "patch_embed.proj.weight" in state_dict:
            w = state_dict["patch_embed.proj.weight"]
            cfg["embed_dim"] = int(w.shape[0])
            if w.ndim == 4:
                cfg["in_chans"] = int(w.shape[1])
                cfg["patch_size"] = int(w.shape[2])
        elif "cls_token" in state_dict:
            cfg["embed_dim"] = int(state_dict["cls_token"].shape[-1])
        elif "norm.weight" in state_dict:
            cfg["embed_dim"] = int(state_dict["norm.weight"].shape[0])

        embed_dim = cfg.get("embed_dim", 768)
        patch_size = cfg.get("patch_size", 16)

        # 2. depth from blocks.<idx>.*
        block_indices = []
        for k in state_dict:
            m = re.match(r"^blocks\.(\d+)\.", k)
            if m:
                block_indices.append(int(m.group(1)))
        cfg["depth"] = (max(block_indices) + 1) if block_indices else 12

        # 3. Structural tokens (CLS and storage/register tokens)
        cfg["use_cls_token"] = "cls_token" in state_dict
        if "storage_tokens" in state_dict:
            cfg["n_storage_tokens"] = int(state_dict["storage_tokens"].shape[1])
        elif "register_tokens" in state_dict:
            cfg["n_storage_tokens"] = int(state_dict["register_tokens"].shape[1])
        else:
            cfg["n_storage_tokens"] = 0

        # 4. Positional embedding type, img_size, and num_heads
        if "pos_embed" in state_dict and "rope_embed.periods" not in state_dict:
            cfg["pos_embed_type"] = "learnable"
            L = int(state_dict["pos_embed"].shape[1])
            grid_size = int(math.isqrt(L))
            if grid_size > 0:
                cfg["img_size"] = int(grid_size * patch_size)
            else:
                cfg["img_size"] = 224
        elif "rope_embed.periods" in state_dict:
            cfg["pos_embed_type"] = "rope"
            cfg["img_size"] = 224
            p_dtype = state_dict["rope_embed.periods"].dtype
            if p_dtype == torch.float32:
                cfg["pos_embed_rope_dtype"] = "fp32"
            elif p_dtype == torch.float16:
                cfg["pos_embed_rope_dtype"] = "fp16"
            elif p_dtype == torch.bfloat16:
                cfg["pos_embed_rope_dtype"] = "bf16"
            periods_len = int(state_dict["rope_embed.periods"].shape[0])
            if periods_len > 0:
                d_head = periods_len * 4
                if embed_dim % d_head == 0:
                    cfg["num_heads"] = embed_dim // d_head
        else:
            cfg["pos_embed_type"] = "rope"
            cfg["img_size"] = 224

        if "num_heads" not in cfg:
            if embed_dim >= 192 and embed_dim % 64 == 0:
                cfg["num_heads"] = embed_dim // 64
            elif embed_dim < 192 and embed_dim % 16 == 0:
                cfg["num_heads"] = embed_dim // 16
            elif embed_dim == 4096:
                cfg["num_heads"] = 32
            else:
                for hd in (64, 48, 80, 32, 96, 128, 16):
                    if embed_dim % hd == 0:
                        cfg["num_heads"] = embed_dim // hd
                        break
                else:
                    cfg["num_heads"] = 12

        # 5. FFN layer & ratio
        if "blocks.0.mlp.w1.weight" in state_dict:
            w1_out = int(state_dict["blocks.0.mlp.w1.weight"].shape[0])
            cfg["ffn_layer"] = "swiglu"
            cfg["ffn_bias"] = "blocks.0.mlp.w1.bias" in state_dict
            raw_ratio = (w1_out * 3.0 / 2.0) / embed_dim
            if abs(raw_ratio - 4.0) < 0.15:
                cfg["ffn_ratio"] = 4.0
            else:
                cfg["ffn_ratio"] = round(raw_ratio, 6)
        elif "blocks.0.mlp.fc1.weight" in state_dict:
            fc1_out = int(state_dict["blocks.0.mlp.fc1.weight"].shape[0])
            cfg["ffn_layer"] = "mlp"
            cfg["ffn_bias"] = "blocks.0.mlp.fc1.bias" in state_dict
            cfg["ffn_ratio"] = float(fc1_out) / embed_dim
        else:
            cfg["ffn_layer"] = "mlp"
            cfg["ffn_ratio"] = 4.0
            cfg["ffn_bias"] = True

        # 6. Attention biases & LayerScale
        cfg["qkv_bias"] = "blocks.0.attn.qkv.bias" in state_dict
        cfg["proj_bias"] = "blocks.0.attn.proj.bias" in state_dict
        cfg["mask_k_bias"] = "blocks.0.attn.qkv.bias_mask" in state_dict
        cfg["layerscale_init"] = 1e-5 if "blocks.0.ls1.gamma" in state_dict else None

        # 7. Normalization layers
        if "blocks.0.norm1.weight" in state_dict and "blocks.0.norm1.bias" not in state_dict:
            cfg["norm_layer"] = "rmsnorm"
        else:
            cfg["norm_layer"] = "layernorm"

        cfg["patch_embed_norm"] = "patch_embed.norm.weight" in state_dict
        cfg["untie_cls_and_patch_norms"] = "cls_norm.weight" in state_dict
        cfg["untie_global_and_local_cls_norm"] = "local_cls_norm.weight" in state_dict

        return cfg


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
