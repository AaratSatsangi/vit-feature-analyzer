import os
import re
import glob
import json
from typing import Any, Dict, List, Optional, Tuple
import yaml
import torch
from PyQt5.QtWidgets import QFileDialog

from models import BaseViT, VisionTransformer, discover_model_classes
from app.widgets.config_dialog import LoadStrategyDialog, ModelConfigDialog


SUPPORTED_CKPT_EXTENSIONS = (".pth", ".pt", ".bin", ".safetensors")


def discover_models_in_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Scan a directory recursively for PyTorch / SafeTensors model checkpoints.
    Supports both flat folders of checkpoints and structured experiment directories.
    """
    available = []
    seen_paths = set()

    for ext in SUPPORTED_CKPT_EXTENSIONS:
        pattern = os.path.join(folder_path, "**", f"*{ext}")
        for pth in glob.glob(pattern, recursive=True):
            abs_pth = os.path.abspath(pth)
            if abs_pth in seen_paths:
                continue
            seen_paths.add(abs_pth)

            exp_name = os.path.splitext(os.path.basename(pth))[0]
            parent_dir = os.path.basename(os.path.dirname(pth))
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(pth)))

            # Optional: check if an experiment log with accuracy exists nearby
            max_acc = 0.0
            for candidate_log_dir in (
                os.path.join(os.path.dirname(os.path.dirname(pth)), "logs"),
                os.path.join(os.path.dirname(pth), "logs"),
            ):
                log_file = os.path.join(candidate_log_dir, f"{exp_name}.log")
                if os.path.isfile(log_file):
                    try:
                        with open(log_file, "r", errors="ignore") as lf:
                            matches = re.findall(r"Linear Probe Test Acc:\s+(\d+\.?\d*)%", lf.read())
                            if matches:
                                max_acc = max(float(m) for m in matches)
                    except Exception:
                        pass

            if parent_dir.lower() in ("checkpoints", "ckpt", "weights") and grandparent_dir:
                display_name = f"{grandparent_dir} - {exp_name}"
            elif parent_dir and parent_dir != os.path.basename(os.path.normpath(folder_path)):
                display_name = f"{parent_dir} / {exp_name}"
            else:
                display_name = exp_name

            available.append({
                "name": display_name,
                "path": pth,
                "best_acc": max_acc,
            })

    available.sort(key=lambda x: x["name"])
    return available


def _unwrap_wandb_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unwrap a WandB-style configuration dict where parameters are stored as
    `{"param_name": {"desc": null, "value": actual_value}}`.
    """
    unwrapped = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and "value" in v and set(v.keys()).issubset({"value", "desc"}):
            unwrapped[k] = v["value"]
        else:
            unwrapped[k] = v
    return unwrapped


def _try_merge_local_wandb_config(model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    If a local `wandb/` directory exists alongside the experiment, merge any
    logged encoder parameters as a fallback without requiring external helpers.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(model_path))
    exp_dir = os.path.dirname(ckpt_dir)
    for root_candidate in (ckpt_dir, exp_dir):
        wandb_dir = os.path.join(root_candidate, "wandb")
        if os.path.isdir(wandb_dir):
            cfg_files = glob.glob(os.path.join(wandb_dir, "**", "files", "config.yaml"), recursive=True)
            for wf in cfg_files:
                try:
                    with open(wf, "r") as f:
                        w_raw = yaml.safe_load(f)
                    if isinstance(w_raw, dict):
                        w_clean = _unwrap_wandb_dict(w_raw)
                        for k, v in w_clean.items():
                            config.setdefault(k, v)
                except Exception:
                    continue
    return config


def parse_config_file(config_path: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a YAML or JSON configuration file into a flat dictionary of ViT parameters.
    Supports:
    - Flat YAML/JSON files (`embed_dim: 768`, `depth: 12`, ...)
    - Nested YAML/JSON files (`encoder_config`, `model_config`, `model`, `backbone`, `vit`, `arch`)
    - WandB `config.yaml` files (`param: {desc: null, value: ...}`)
    """
    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            raw = json.load(f)
        else:
            raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML/JSON dictionary in {config_path}, got {type(raw).__name__}.")

    config = _unwrap_wandb_dict(raw)
    if model_path:
        config = _try_merge_local_wandb_config(model_path, config)

    # Check common nested section names
    for nested_key in ("encoder_config", "model_config", "vit_config", "backbone", "model", "arch"):
        if nested_key in config and isinstance(config[nested_key], dict):
            nested = _unwrap_wandb_dict(config[nested_key])
            # Preserve top-level model_class if specified
            if "model_class" in config and "model_class" not in nested:
                nested["model_class"] = config["model_class"]
            return nested

    return config


def find_companion_config(model_path: str) -> Optional[str]:
    """
    Search for a companion YAML/JSON config file near `model_path`.
    Returns the file path if found, or `None` if no config is automatically detected.
    """
    ckpt_dir = os.path.dirname(os.path.abspath(model_path))
    exp_dir = os.path.dirname(ckpt_dir)
    stem = os.path.splitext(os.path.basename(model_path))[0]

    candidates = [
        os.path.join(ckpt_dir, f"{stem}.yaml"),
        os.path.join(ckpt_dir, f"{stem}.yml"),
        os.path.join(ckpt_dir, f"{stem}.json"),
        os.path.join(ckpt_dir, f"{stem}_config.yaml"),
        os.path.join(ckpt_dir, f"{stem}_config.yml"),
        os.path.join(ckpt_dir, "config.yaml"),
        os.path.join(ckpt_dir, "config.yml"),
        os.path.join(ckpt_dir, "config.json"),
        os.path.join(exp_dir, "config", f"{stem}_config.yaml"),
        os.path.join(exp_dir, "config", f"{stem}.yaml"),
        os.path.join(exp_dir, "config.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def resolve_config_path(model_path: str, parent_widget=None) -> Optional[str]:
    """Find the config YAML for a checkpoint if present."""
    return find_companion_config(model_path)


def extract_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint from disk, unwrap nested state_dict containers, and strip
    common wrapper prefixes (DDP, torch.compile, teacher/student/backbone wrappers).
    """
    if model_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file

            raw_ckpt = load_file(model_path, device="cpu")
        except ImportError as e:
            raise RuntimeError(
                "Loading .safetensors files requires the `safetensors` package (`pip install safetensors`)."
            ) from e
    else:
        try:
            raw_ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            raw_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(raw_ckpt, torch.nn.Module):
        raw_ckpt = raw_ckpt.state_dict()

    if not isinstance(raw_ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: expected dict or nn.Module, got {type(raw_ckpt).__name__}")

    # Unwrap nested state dictionary keys
    state_dict_full = raw_ckpt
    for wrapper_key in (
        "model_state_dict",
        "state_dict",
        "model",
        "teacher",
        "student",
        "encoder",
        "x_encoder",
        "backbone",
    ):
        if wrapper_key in state_dict_full and isinstance(state_dict_full[wrapper_key], dict):
            candidate = state_dict_full[wrapper_key]
            if any(isinstance(v, torch.Tensor) for v in candidate.values()):
                state_dict_full = candidate
                break

    # Smart prefix stripping
    prefixes_to_check = [
        "x_encoder._orig_mod.",
        "x_encoder.",
        "teacher.backbone.",
        "student.backbone.",
        "backbone._orig_mod.",
        "backbone.",
        "encoder._orig_mod.",
        "encoder.",
    ]

    prefix = ""
    for p in prefixes_to_check:
        for k in state_dict_full:
            if p in k:
                prefix = k[: k.find(p) + len(p)]
                break
        if prefix:
            break

    if prefix:
        cleaned = {k[len(prefix) :]: v for k, v in state_dict_full.items() if k.startswith(prefix)}
    else:
        cleaned = dict(state_dict_full)

    # Iteratively strip leading wrapper prefixes like `module.` or `_orig_mod.`
    changed = True
    while changed:
        changed = False
        for mod_prefix in ("module.", "_orig_mod.", "model."):
            if cleaned and all(k.startswith(mod_prefix) for k in cleaned):
                cleaned = {k[len(mod_prefix) :]: v for k, v in cleaned.items()}
                changed = True

    # Normalize common token key aliases (e.g., DINOv2 `register_tokens` -> `storage_tokens`)
    if "register_tokens" in cleaned and "storage_tokens" not in cleaned:
        cleaned["storage_tokens"] = cleaned.pop("register_tokens")

    return cleaned


def instantiate_and_load_model(
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: str,
    model_class_name: str = "VisionTransformer",
) -> Tuple[BaseViT, Dict[str, Any]]:
    """
    Instantiate the requested `BaseViT` subclass using `config` and load `state_dict`.
    """
    registry = discover_model_classes()
    target_cls_name = config.get("model_class", model_class_name)
    model_cls = registry.get(target_cls_name, VisionTransformer)

    init_kwargs = {k: v for k, v in config.items() if k != "model_class"}
    model = model_cls(**init_kwargs)

    sd_to_load = dict(state_dict)
    model_sd = model.state_dict()

    # Fill optional structural buffers/parameters if omitted in standard ViT checkpoints
    if "mask_token" in model_sd:
        if "mask_token" not in sd_to_load:
            sd_to_load["mask_token"] = model_sd["mask_token"]
        elif sd_to_load["mask_token"].shape != model_sd["mask_token"].shape and sd_to_load["mask_token"].numel() == model_sd["mask_token"].numel():
            sd_to_load["mask_token"] = sd_to_load["mask_token"].reshape(model_sd["mask_token"].shape)

    if (
        getattr(model, "pos_embed_type", None) == "sinusoidal"
        and "pos_embed" in model_sd
        and "pos_embed" not in sd_to_load
    ):
        sd_to_load["pos_embed"] = model_sd["pos_embed"]

    # If learnable pos_embed has a different spatial grid size in the checkpoint than the
    # configured img_size, adapt model.pos_embed shape so load_state_dict succeeds and
    # interpolate_pos_encoding dynamically interpolates it during inference.
    if (
        hasattr(model, "pos_embed")
        and getattr(model, "pos_embed", None) is not None
        and "pos_embed" in sd_to_load
        and model.pos_embed.shape != sd_to_load["pos_embed"].shape
        and model.pos_embed.shape[-1] == sd_to_load["pos_embed"].shape[-1]
    ):
        model.pos_embed = torch.nn.Parameter(torch.empty_like(sd_to_load["pos_embed"]))

    # Ignore unused classification head weights when model.head is Identity
    if isinstance(getattr(model, "head", None), torch.nn.Identity):
        for head_key in list(sd_to_load.keys()):
            if head_key.startswith(("head.", "fc.", "fc_norm.")):
                sd_to_load.pop(head_key, None)

    model.load_state_dict(sd_to_load, strict=True)
    model.to(device)
    model.eval()

    meta = model.get_model_metadata() if hasattr(model, "get_model_metadata") else {
        "img_size": config.get("img_size", 224),
        "patch_size": getattr(getattr(model, "patch_embed", None), "patch_size", [16])[0],
        "num_heads": getattr(model, "num_heads", 12),
        "n_storage": getattr(model, "n_storage_tokens", 0),
        "use_cls": getattr(model, "use_cls_token", True),
    }
    meta["raw_config"] = config
    meta["model_class"] = model_cls.__name__
    return model, meta


def load_model(
    model_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
    model_class_name: str = "VisionTransformer",
) -> Tuple[BaseViT, Dict[str, Any]]:
    """
    Programmatic model loader. Uses `config_path` if provided, otherwise auto-infers
    configuration from the checkpoint `state_dict`.
    """
    state_dict = extract_state_dict(model_path)
    registry = discover_model_classes()
    model_cls = registry.get(model_class_name, VisionTransformer)

    config = model_cls.infer_config_from_state_dict(state_dict)
    if config_path and os.path.isfile(config_path):
        file_cfg = parse_config_file(config_path, model_path=model_path)
        config.update(file_cfg)

    return instantiate_and_load_model(state_dict, config, device, model_class_name=model_class_name)


def interactive_load_model(
    model_path: str,
    device: str = "cpu",
    model_class_name: str = "VisionTransformer",
    parent_widget=None,
) -> Tuple[Optional[BaseViT], Optional[Dict[str, Any]]]:
    """
    Interactive GUI model loader with automatic inference, manual configuration dialog,
    and retry loop until the model loads cleanly or the user cancels.
    """
    state_dict = extract_state_dict(model_path)
    registry = discover_model_classes()
    available_classes = list(registry.keys())
    model_cls = registry.get(model_class_name, VisionTransformer)

    # Always compute best-effort inferred config to seed defaults
    current_cfg = model_cls.infer_config_from_state_dict(state_dict)
    current_cfg["model_class"] = model_class_name

    companion_path = find_companion_config(model_path)
    open_manual_first = False
    last_error: Optional[str] = None

    if companion_path:
        try:
            file_cfg = parse_config_file(companion_path, model_path=model_path)
            current_cfg.update(file_cfg)
        except Exception as e:
            last_error = f"Failed to parse companion config ({companion_path}):\n{e}"
            open_manual_first = True
    elif parent_widget is not None:
        strat_dlg = LoadStrategyDialog(os.path.basename(model_path), parent=parent_widget)
        if strat_dlg.exec_() != LoadStrategyDialog.Accepted or not strat_dlg.choice:
            return None, None

        if strat_dlg.choice == "manual":
            open_manual_first = True
        elif strat_dlg.choice == "file":
            cfg_path, _ = QFileDialog.getOpenFileName(
                parent_widget,
                "Select YAML or JSON Config",
                os.path.dirname(model_path),
                "Config Files (*.yaml *.yml *.json)",
            )
            if not cfg_path:
                return None, None
            try:
                file_cfg = parse_config_file(cfg_path, model_path=model_path)
                current_cfg.update(file_cfg)
            except Exception as e:
                last_error = f"Failed to parse selected config ({cfg_path}):\n{e}"
                open_manual_first = True

    while True:
        if not open_manual_first:
            try:
                selected_cls = current_cfg.get("model_class", model_class_name)
                return instantiate_and_load_model(
                    state_dict, current_cfg, device, model_class_name=selected_cls
                )
            except Exception as e:
                last_error = (
                    f"{type(e).__name__}: {e}\n\n"
                    "Tip: Verify embed_dim, depth, num_heads, patch_size, pos_embed_type "
                    "(rope / learnable / sinusoidal), and ffn_layer below.\n"
                    "If this is a non-standard ViT architecture, add a custom BaseViT subclass in `models/`."
                )
                if parent_widget is None:
                    raise

        open_manual_first = False
        cfg_dlg = ModelConfigDialog(
            initial_config=current_cfg,
            available_classes=available_classes,
            selected_class=current_cfg.get("model_class", model_class_name),
            error_message=last_error,
            parent=parent_widget,
        )
        if cfg_dlg.exec_() != ModelConfigDialog.Accepted:
            return None, None

        current_cfg = cfg_dlg.get_config()
        model_class_name = current_cfg.get("model_class", model_class_name)
