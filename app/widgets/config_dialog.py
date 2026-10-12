import json
from typing import Any, Dict, List, Optional
import yaml

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from PyQt5.QtCore import Qt


CUSTOM_VIT_TEMPLATE = '''import torch
from torch import Tensor, nn
from typing import Dict, Optional, Union
from models.base import BaseViT


class MyCustomViT(BaseViT):
    """
    Example custom Vision Transformer for ViT Feature Analyzer.
    Place this file inside the `models/` directory (e.g., `models/my_custom_vit.py`).
    It will be automatically discovered and listed in the Architecture dropdown!
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        use_cls_token: bool = True,
        n_storage_tokens: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_cls_token = use_cls_token
        self.n_storage_tokens = n_storage_tokens

        # TODO: Initialize your custom layers here
        # self.backbone = ...

    def forward_features(self, x: Tensor) -> Union[Dict[str, Tensor], Tensor]:
        """
        REQUIRED: Given input image tensor `x` of shape [B, C, H, W],
        return patch tokens of shape [B, N_patches, D] (excluding CLS/register tokens),
        or a dict {"x_norm_patchtokens": patch_tokens}.
        """
        raise NotImplementedError

    def get_attention_map(self, x: Tensor, layer_num: int = -1) -> Optional[Tensor]:
        """
        OPTIONAL: Return post-softmax attention weights of shape
        [B, num_heads, Seq, Seq] for `layer_num` (-1 = last block).
        Return None if attention visualization is not needed.
        """
        return None
'''


class CustomViTGuideDialog(QDialog):
    """Developer guide explaining how to add a custom ViT model in `models/`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Developer Guide: Adding a Custom ViT Architecture")
        self.setMinimumSize(760, 620)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QLabel(
            "<h3>Plugging in a Custom Vision Transformer</h3>"
            "<p><b>ViT Feature Analyzer</b> supports standard ViTs (RoPE, Learnable, and 2D Sinusoidal "
            "positional embeddings) out of the box. If your checkpoint uses a custom architecture "
            "with different layer names or forward logic, you can add it in 3 steps:</p>"
            "<ol>"
            "<li>Create a new Python file inside the <code>models/</code> folder "
            "(e.g., <code>models/my_custom_vit.py</code>).</li>"
            "<li>Subclass <code>BaseViT</code> from <code>models.base</code>.</li>"
            "<li>Implement <code>forward_features(self, x)</code> (required) and optionally "
            "<code>get_attention_map(self, x, layer_num=-1)</code>.</li>"
            "</ol>"
            "<p>The app automatically scans <code>models/*.py</code> on startup and adds your class "
            "to the <b>Architecture</b> selector and config dialog.</p>"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.code_box = QPlainTextEdit()
        self.code_box.setPlainText(CUSTOM_VIT_TEMPLATE)
        self.code_box.setReadOnly(True)
        self.code_box.setStyleSheet(
            "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; "
            "font-size: 12px; background: #0f172a; color: #e2e8f0; border: 1px solid #334155; "
            "border-radius: 6px; padding: 8px;"
        )
        layout.addWidget(self.code_box, stretch=1)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("📋 Copy Template to Clipboard")
        btn_copy.clicked.connect(self._copy_template)
        btn_row.addWidget(btn_copy)

        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)

        layout.addLayout(btn_row)

    def _copy_template(self):
        QApplication.clipboard().setText(CUSTOM_VIT_TEMPLATE)
        QMessageBox.information(self, "Copied", "Custom ViT template copied to clipboard!")


class LoadStrategyDialog(QDialog):
    """
    Prompts the user how to configure a checkpoint when no companion config file
    was automatically found.
    """

    def __init__(self, checkpoint_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Model Checkpoint")
        self.setMinimumWidth(460)
        self.choice: Optional[str] = None
        self._build_ui(checkpoint_name)

    def _build_ui(self, checkpoint_name: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl = QLabel(
            f"<b>Checkpoint:</b> <code>{checkpoint_name}</code><br><br>"
            "No companion YAML/JSON config file was automatically detected for this checkpoint. "
            "How would you like to initialize the Vision Transformer?"
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        btn_auto = QPushButton("✨ Auto-Infer Configuration from Checkpoint (Recommended)")
        btn_auto.setStyleSheet("background: #00bcd4; color: #000; font-weight: bold; padding: 10px;")
        btn_auto.clicked.connect(lambda: self._select("auto"))
        layout.addWidget(btn_auto)

        btn_manual = QPushButton("⚙️ Enter / Review Model Parameters Manually")
        btn_manual.setStyleSheet("padding: 8px;")
        btn_manual.clicked.connect(lambda: self._select("manual"))
        layout.addWidget(btn_manual)

        btn_file = QPushButton("📂 Load Existing YAML / JSON Config File...")
        btn_file.setStyleSheet("padding: 8px;")
        btn_file.clicked.connect(lambda: self._select("file"))
        layout.addWidget(btn_file)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_cancel)

    def _select(self, choice: str):
        self.choice = choice
        self.accept()


class ModelConfigDialog(QDialog):
    """
    Interactive dialog allowing users to inspect and edit all ViT configuration
    parameters, switch model classes, or load/save YAML configs.
    """

    def __init__(
        self,
        initial_config: Dict[str, Any],
        available_classes: List[str],
        selected_class: str = "VisionTransformer",
        error_message: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("ViT Model Configuration")
        self.setMinimumWidth(620)
        self._config = dict(initial_config)
        self._available_classes = available_classes or ["VisionTransformer"]
        self._selected_class = selected_class if selected_class in self._available_classes else self._available_classes[0]
        self._error_message = error_message

        self._build_ui()
        self._populate_from_dict(self._config)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        if self._error_message:
            err_box = QGroupBox("⚠️ Previous Load Attempt Failed")
            err_box.setStyleSheet(
                "QGroupBox { border: 1px solid #ef4444; background: rgba(239, 68, 68, 0.1); "
                "border-radius: 6px; margin-top: 8px; padding-top: 16px; color: #fca5a5; }"
                "QGroupBox::title { color: #f87171; }"
            )
            err_layout = QVBoxLayout(err_box)
            err_lbl = QLabel(
                "Please adjust the model parameters below to match your checkpoint and try again.\n"
                "If this checkpoint comes from a non-standard custom ViT architecture, click "
                "'Custom ViT Guide' below to see how to add your model class to `models/`."
            )
            err_lbl.setWordWrap(True)
            err_layout.addWidget(err_lbl)

            err_txt = QPlainTextEdit()
            err_txt.setPlainText(str(self._error_message))
            err_txt.setReadOnly(True)
            err_txt.setMaximumHeight(110)
            err_txt.setStyleSheet(
                "font-family: monospace; font-size: 11px; background: #0f172a; "
                "color: #fca5a5; border: 1px solid #7f1d1d;"
            )
            err_layout.addWidget(err_txt)
            layout.addWidget(err_box)
        else:
            info_lbl = QLabel(
                "Configure the Vision Transformer architecture parameters below. "
                "Fields have been pre-filled with best-effort values inferred from the checkpoint."
            )
            info_lbl.setWordWrap(True)
            info_lbl.setStyleSheet("color: #94a3b8;")
            layout.addWidget(info_lbl)

        # Core Architecture Form
        grp_arch = QGroupBox("Architecture & Dimensions")
        form = QGridLayout(grp_arch)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        form.addWidget(QLabel("Model Class (`models/`):"), 0, 0)
        self.combo_class = QComboBox()
        self.combo_class.addItems(self._available_classes)
        self.combo_class.setCurrentText(self._selected_class)
        form.addWidget(self.combo_class, 0, 1)

        form.addWidget(QLabel("Positional Embedding:"), 0, 2)
        self.combo_pos = QComboBox()
        self.combo_pos.addItems(["rope", "learnable", "sinusoidal", "none"])
        form.addWidget(self.combo_pos, 0, 3)

        form.addWidget(QLabel("Image Size (`img_size`):"), 1, 0)
        self.spin_img_size = QSpinBox()
        self.spin_img_size.setRange(16, 4096)
        self.spin_img_size.setSingleStep(16)
        form.addWidget(self.spin_img_size, 1, 1)

        form.addWidget(QLabel("Patch Size (`patch_size`):"), 1, 2)
        self.spin_patch_size = QSpinBox()
        self.spin_patch_size.setRange(1, 256)
        form.addWidget(self.spin_patch_size, 1, 3)

        form.addWidget(QLabel("Embed Dim (`embed_dim`):"), 2, 0)
        self.spin_embed_dim = QSpinBox()
        self.spin_embed_dim.setRange(8, 16384)
        self.spin_embed_dim.setSingleStep(64)
        form.addWidget(self.spin_embed_dim, 2, 1)

        form.addWidget(QLabel("Depth / Blocks (`depth`):"), 2, 2)
        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(1, 256)
        form.addWidget(self.spin_depth, 2, 3)

        form.addWidget(QLabel("Attention Heads (`num_heads`):"), 3, 0)
        self.spin_heads = QSpinBox()
        self.spin_heads.setRange(1, 256)
        form.addWidget(self.spin_heads, 3, 1)

        form.addWidget(QLabel("Input Channels (`in_chans`):"), 3, 2)
        self.spin_chans = QSpinBox()
        self.spin_chans.setRange(1, 64)
        form.addWidget(self.spin_chans, 3, 3)

        form.addWidget(QLabel("FFN Layer (`ffn_layer`):"), 4, 0)
        self.combo_ffn = QComboBox()
        self.combo_ffn.addItems(["mlp", "swiglu", "swiglu32", "swiglu64", "swiglu128"])
        form.addWidget(self.combo_ffn, 4, 1)

        form.addWidget(QLabel("FFN Ratio (`ffn_ratio`):"), 4, 2)
        self.spin_ffn_ratio = QDoubleSpinBox()
        self.spin_ffn_ratio.setRange(0.25, 32.0)
        self.spin_ffn_ratio.setDecimals(6)
        self.spin_ffn_ratio.setSingleStep(0.5)
        form.addWidget(self.spin_ffn_ratio, 4, 3)

        form.addWidget(QLabel("Norm Layer (`norm_layer`):"), 5, 0)
        self.combo_norm = QComboBox()
        self.combo_norm.addItems(["layernorm", "layernormbf16", "rmsnorm"])
        form.addWidget(self.combo_norm, 5, 1)

        form.addWidget(QLabel("Register/Storage Tokens:"), 5, 2)
        self.spin_storage = QSpinBox()
        self.spin_storage.setRange(0, 256)
        form.addWidget(self.spin_storage, 5, 3)

        form.addWidget(QLabel("RoPE Base Period:"), 6, 0)
        self.spin_rope_base = QDoubleSpinBox()
        self.spin_rope_base.setRange(1.0, 1000000.0)
        self.spin_rope_base.setDecimals(1)
        self.spin_rope_base.setValue(100.0)
        form.addWidget(self.spin_rope_base, 6, 1)

        layout.addWidget(grp_arch)

        # Options / Toggles
        grp_opts = QGroupBox("Structural Tokens, Biases & Normalization Options")
        opts_grid = QGridLayout(grp_opts)

        self.cb_cls = QCheckBox("Use CLS Token (`use_cls_token`)")
        self.cb_qkv_bias = QCheckBox("QKV Bias (`qkv_bias`)")
        self.cb_proj_bias = QCheckBox("Proj Bias (`proj_bias`)")
        self.cb_ffn_bias = QCheckBox("FFN Bias (`ffn_bias`)")
        self.cb_patch_norm = QCheckBox("PatchEmbed Norm (`patch_embed_norm`)")
        self.cb_layerscale = QCheckBox("LayerScale (`layerscale_init`)")
        self.cb_mask_k = QCheckBox("Mask K Bias (`mask_k_bias`)")
        self.cb_untie_norm = QCheckBox("Untie CLS/Patch Norms (`untie_cls_and_patch_norms`)")

        opts_grid.addWidget(self.cb_cls, 0, 0)
        opts_grid.addWidget(self.cb_qkv_bias, 0, 1)
        opts_grid.addWidget(self.cb_proj_bias, 1, 0)
        opts_grid.addWidget(self.cb_ffn_bias, 1, 1)
        opts_grid.addWidget(self.cb_patch_norm, 2, 0)
        opts_grid.addWidget(self.cb_layerscale, 2, 1)
        opts_grid.addWidget(self.cb_mask_k, 3, 0)
        opts_grid.addWidget(self.cb_untie_norm, 3, 1)

        layout.addWidget(grp_opts)

        # Extra kwargs for custom models
        extra_form = QFormLayout()
        self.edit_extra = QLineEdit()
        self.edit_extra.setPlaceholderText('Optional JSON kwargs for custom models, e.g. {"drop_path_rate": 0.1}')
        extra_form.addRow("Extra Kwargs (JSON):", self.edit_extra)
        layout.addLayout(extra_form)

        # Bottom Action Bar
        btn_bar = QHBoxLayout()

        btn_load_yaml = QPushButton("📂 Load YAML/JSON...")
        btn_load_yaml.clicked.connect(self._on_load_file)
        btn_bar.addWidget(btn_load_yaml)

        btn_save_yaml = QPushButton("💾 Save Config...")
        btn_save_yaml.clicked.connect(self._on_save_file)
        btn_bar.addWidget(btn_save_yaml)

        btn_guide = QPushButton("📘 Custom ViT Guide")
        btn_guide.clicked.connect(self._on_show_guide)
        btn_bar.addWidget(btn_guide)

        btn_bar.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(btn_cancel)

        btn_ok = QPushButton("✓ Load Model")
        btn_ok.setStyleSheet("background: #00bcd4; color: #000; font-weight: bold; padding: 6px 16px;")
        btn_ok.clicked.connect(self._on_accept)
        btn_bar.addWidget(btn_ok)

        layout.addLayout(btn_bar)

    def _populate_from_dict(self, cfg: Dict[str, Any]):
        if "model_class" in cfg and cfg["model_class"] in self._available_classes:
            self.combo_class.setCurrentText(cfg["model_class"])

        pos_type = str(cfg.get("pos_embed_type", "rope")).lower()
        if pos_type in ("rope", "learnable", "sinusoidal", "none"):
            self.combo_pos.setCurrentText(pos_type)

        img_sz = cfg.get("img_size", 224)
        if isinstance(img_sz, (list, tuple)):
            img_sz = img_sz[0]
        self.spin_img_size.setValue(int(img_sz))

        patch_sz = cfg.get("patch_size", 16)
        if isinstance(patch_sz, (list, tuple)):
            patch_sz = patch_sz[0]
        self.spin_patch_size.setValue(int(patch_sz))

        self.spin_embed_dim.setValue(int(cfg.get("embed_dim", 768)))
        self.spin_depth.setValue(int(cfg.get("depth", 12)))
        self.spin_heads.setValue(int(cfg.get("num_heads", 12)))
        self.spin_chans.setValue(int(cfg.get("in_chans", 3)))

        ffn_layer = str(cfg.get("ffn_layer", "mlp")).lower()
        if ffn_layer in ("mlp", "swiglu", "swiglu32", "swiglu64", "swiglu128"):
            self.combo_ffn.setCurrentText(ffn_layer)

        self.spin_ffn_ratio.setValue(float(cfg.get("ffn_ratio", 4.0)))

        norm_layer = str(cfg.get("norm_layer", "layernorm")).lower()
        if norm_layer in ("layernorm", "layernormbf16", "rmsnorm"):
            self.combo_norm.setCurrentText(norm_layer)

        self.spin_storage.setValue(int(cfg.get("n_storage_tokens", 0)))
        self.spin_rope_base.setValue(float(cfg.get("pos_embed_rope_base", 100.0) or 100.0))

        self.cb_cls.setChecked(bool(cfg.get("use_cls_token", True)))
        self.cb_qkv_bias.setChecked(bool(cfg.get("qkv_bias", True)))
        self.cb_proj_bias.setChecked(bool(cfg.get("proj_bias", True)))
        self.cb_ffn_bias.setChecked(bool(cfg.get("ffn_bias", True)))
        self.cb_patch_norm.setChecked(bool(cfg.get("patch_embed_norm", True)))
        self.cb_layerscale.setChecked(cfg.get("layerscale_init") is not None)
        self.cb_mask_k.setChecked(bool(cfg.get("mask_k_bias", False)))
        self.cb_untie_norm.setChecked(bool(cfg.get("untie_cls_and_patch_norms", False)))

    def get_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "model_class": self.combo_class.currentText(),
            "pos_embed_type": self.combo_pos.currentText(),
            "img_size": self.spin_img_size.value(),
            "patch_size": self.spin_patch_size.value(),
            "in_chans": self.spin_chans.value(),
            "embed_dim": self.spin_embed_dim.value(),
            "depth": self.spin_depth.value(),
            "num_heads": self.spin_heads.value(),
            "ffn_layer": self.combo_ffn.currentText(),
            "ffn_ratio": self.spin_ffn_ratio.value(),
            "norm_layer": self.combo_norm.currentText(),
            "n_storage_tokens": self.spin_storage.value(),
            "pos_embed_rope_base": self.spin_rope_base.value(),
            "use_cls_token": self.cb_cls.isChecked(),
            "qkv_bias": self.cb_qkv_bias.isChecked(),
            "proj_bias": self.cb_proj_bias.isChecked(),
            "ffn_bias": self.cb_ffn_bias.isChecked(),
            "patch_embed_norm": self.cb_patch_norm.isChecked(),
            "layerscale_init": 1e-5 if self.cb_layerscale.isChecked() else None,
            "mask_k_bias": self.cb_mask_k.isChecked(),
            "untie_cls_and_patch_norms": self.cb_untie_norm.isChecked(),
        }
        extra_txt = self.edit_extra.text().strip()
        if extra_txt:
            extra_dict = json.loads(extra_txt)
            if isinstance(extra_dict, dict):
                cfg.update(extra_dict)
        return cfg

    def _on_accept(self):
        extra_txt = self.edit_extra.text().strip()
        if extra_txt:
            try:
                parsed = json.loads(extra_txt)
                if not isinstance(parsed, dict):
                    raise ValueError("Extra kwargs must be a JSON object/dictionary.")
            except Exception as e:
                QMessageBox.warning(self, "Invalid Extra Kwargs", f"Could not parse Extra Kwargs JSON:\n{e}")
                return
        self.accept()

    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Configuration File", "", "Config Files (*.yaml *.yml *.json)"
        )
        if not path:
            return
        try:
            from app.model_loader import parse_config_file

            loaded = parse_config_file(path)
            self._populate_from_dict(loaded)
        except Exception as e:
            QMessageBox.critical(self, "Config Load Error", f"Failed to parse config file:\n{e}")

    def _on_save_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration File", "vit_config.yaml", "YAML Files (*.yaml *.yml);;JSON Files (*.json)"
        )
        if not path:
            return
        try:
            cfg = self.get_config()
            with open(path, "w") as f:
                if path.endswith(".json"):
                    json.dump(cfg, f, indent=2)
                else:
                    yaml.safe_dump(cfg, f, sort_keys=False)
            QMessageBox.information(self, "Saved", f"Configuration saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save config:\n{e}")

    def _on_show_guide(self):
        dlg = CustomViTGuideDialog(self)
        dlg.exec_()

