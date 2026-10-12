from typing import Any, Dict, Optional, Union
import torch
from torch import Tensor, nn


class BaseViT(nn.Module):
    """
    Base interface for all Vision Transformer models in ViT Feature Analyzer.

    To add support for a custom Vision Transformer architecture:
    1. Create a new Python file inside the `models/` directory (e.g., `models/my_custom_vit.py`).
    2. Define your model class inheriting from `BaseViT`.
    3. Implement `forward_features(self, x: Tensor)` (required) and optionally
       `get_attention_map(self, x: Tensor, layer_num: int = -1)`.

    Your custom model class will be automatically discovered and listed in the
    application's Architecture dropdown and YAML configuration loader.
    """

    def forward_features(self, x: Tensor) -> Union[Dict[str, Tensor], Tensor]:
        """
        Extract patch features from an input batch of images.

        Args:
            x: Input image tensor of shape `[B, C, H, W]`.

        Returns:
            Either:
            - A dictionary containing `"x_norm_patchtokens"` of shape `[B, N_patches, D]`
              (and optionally `"x_norm_clstoken"` of shape `[B, D]`), OR
            - A `Tensor` of patch tokens directly with shape `[B, N_patches, D]`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `forward_features(self, x: Tensor)` "
            "returning a dict with 'x_norm_patchtokens' [B, N, D] or a Tensor [B, N, D]."
        )

    def get_attention_map(self, x: Tensor, layer_num: int = -1) -> Optional[Tensor]:
        """
        Extract self-attention weights from a specific transformer block.

        Args:
            x: Input image tensor of shape `[B, C, H, W]`.
            layer_num: Transformer layer index (`-1` for the last layer).

        Returns:
            Post-softmax attention tensor of shape `[B, num_heads, Seq, Seq]`,
            or `None` if attention extraction is not implemented for this model.
        """
        return None

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Return architectural metadata required by the inference and visualization pipeline.
        Subclasses can either set `self.img_size`, `self.patch_size`, `self.num_heads`,
        `self.n_storage_tokens`, and `self.use_cls_token` attributes in `__init__`,
        or override this method directly.
        """
        img_size = getattr(self, "img_size", 224)
        if isinstance(img_size, (tuple, list)):
            img_size = img_size[0]
        elif hasattr(self, "patch_embed") and hasattr(self.patch_embed, "img_size"):
            ims = self.patch_embed.img_size
            img_size = ims[0] if isinstance(ims, (tuple, list)) else int(ims)

        patch_size = getattr(self, "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        elif hasattr(self, "patch_embed") and hasattr(self.patch_embed, "patch_size"):
            ps = self.patch_embed.patch_size
            patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)

        return {
            "img_size": int(img_size),
            "patch_size": int(patch_size),
            "num_heads": int(getattr(self, "num_heads", 12)),
            "n_storage": int(getattr(self, "n_storage_tokens", 0)),
            "use_cls": bool(getattr(self, "use_cls_token", True)),
        }

    @classmethod
    def infer_config_from_state_dict(cls, state_dict: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Optional classmethod to automatically infer `__init__` keyword arguments
        from a checkpoint `state_dict`.
        """
        return {}

