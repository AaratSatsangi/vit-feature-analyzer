import importlib
import inspect
import os
from typing import Dict, Type

from models.base import BaseViT
from models.vision_transformer import VisionTransformer


def discover_model_classes() -> Dict[str, Type[BaseViT]]:
    """
    Automatically discover all `BaseViT` subclasses defined in `models/*.py`.
    `VisionTransformer` is always listed first as the default standard architecture.
    """
    registry: Dict[str, Type[BaseViT]] = {"VisionTransformer": VisionTransformer}
    models_dir = os.path.dirname(os.path.abspath(__file__))

    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".py") or fname in ("__init__.py", "base.py"):
            continue
        mod_name = f"models.{fname[:-3]}"
        try:
            module = importlib.import_module(mod_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseViT) and obj is not BaseViT and name not in registry:
                    registry[name] = obj
        except Exception:
            # Skip broken user modules during discovery so the app still starts
            continue

    return registry


__all__ = ["BaseViT", "VisionTransformer", "discover_model_classes"]

