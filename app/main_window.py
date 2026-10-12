import os
from PIL import Image

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QComboBox, QLabel, QFileDialog, 
                             QStackedWidget, QTabWidget, QStatusBar, QMessageBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt

from app.model_loader import discover_models_in_folder, interactive_load_model
from app.inference import inference_standard, inference_native, inference_sliding_window
from app.widgets.drop_zone import DropZone
from app.widgets.thumbnail_strip import ThumbnailStrip
from app.widgets.config_dialog import CustomViTGuideDialog
from app.tabs.cosine_tab import CosineTab
from app.tabs.pca_tab import PCATab
from app.tabs.attention_tab import AttentionTab
from app.tabs.combined_pca_tab import CombinedPCATab
from app.utils.export import ensure_dir, save_image_array, save_qpixmap, save_pdf_from_array, save_pdf_from_bytes
from models import discover_model_classes

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ViT Feature Analyzer")
        self.setMinimumSize(1280, 800)
        
        self._model = None
        self._meta = None
        self._device = "cpu"
        self._model_name = "unknown_model"
        
        self._results = []
        self._active_idx = -1
        
        self._build_ui()
        
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 0)
        main_layout.setSpacing(10)
        
        # ── Toolbar 1: Model Loading ──
        tb1 = QHBoxLayout()
        
        self.btn_folder = QPushButton("📁 Select Model Folder")
        self.btn_folder.clicked.connect(self._on_select_folder)
        tb1.addWidget(self.btn_folder)
        
        self.combo_models = QComboBox()
        self.combo_models.setMinimumWidth(260)
        tb1.addWidget(self.combo_models)
        
        tb1.addWidget(QLabel("Arch:"))
        self.combo_arch = QComboBox()
        self._refresh_arch_list()
        tb1.addWidget(self.combo_arch)
        
        tb1.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        import torch
        num_gpus = torch.cuda.device_count()
        self.combo_device.addItems(["cpu"] + [f"cuda:{i}" for i in range(num_gpus)])
        if num_gpus > 0: self.combo_device.setCurrentIndex(1)
        tb1.addWidget(self.combo_device)
        
        self.btn_load_model = QPushButton("🔄 Load Selected Model")
        self.btn_load_model.clicked.connect(self._on_load_model)
        self.btn_load_model.setStyleSheet("background: #00bcd4; color: #000; font-weight: bold;")
        tb1.addWidget(self.btn_load_model)
        
        tb1.addWidget(QLabel("  or  "))
        
        self.btn_load_direct = QPushButton("📂 Direct Checkpoint Load")
        self.btn_load_direct.clicked.connect(self._on_load_direct)
        tb1.addWidget(self.btn_load_direct)
        
        tb1.addStretch()
        
        self.btn_custom_guide = QPushButton("📘 Custom ViT Guide")
        self.btn_custom_guide.setToolTip("Learn how to add a custom Vision Transformer architecture in models/")
        self.btn_custom_guide.clicked.connect(self._on_custom_guide)
        tb1.addWidget(self.btn_custom_guide)
        
        main_layout.addLayout(tb1)
        
        # ── Toolbar 2: Inference & Export ──
        tb2 = QHBoxLayout()
        
        self.btn_load_img = QPushButton("🖼️ Load Images")
        self.btn_load_img.clicked.connect(self._on_load_images)
        self.btn_load_img.setEnabled(False)
        tb2.addWidget(self.btn_load_img)
        
        tb2.addWidget(QLabel("  │  Mode:"))
        
        self.mode_group = QButtonGroup(self)
        self.rad_std = QRadioButton("Standard")
        self.rad_nat = QRadioButton("Native")
        self.rad_sld = QRadioButton("Sliding Window")
        self.rad_std.setChecked(True)
        self.mode_group.addButton(self.rad_std)
        self.mode_group.addButton(self.rad_nat)
        self.mode_group.addButton(self.rad_sld)
        
        tb2.addWidget(self.rad_std)
        tb2.addWidget(self.rad_nat)
        tb2.addWidget(self.rad_sld)
        
        tb2.addWidget(QLabel("  │  Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(["original", "224", "512", "768", "1024"])
        tb2.addWidget(self.combo_res)
        
        tb2.addStretch()
        
        self.btn_save_current = QPushButton("💾 Save Current Tab")
        self.btn_save_current.clicked.connect(self._on_save_current)
        self.btn_save_current.setEnabled(False)
        tb2.addWidget(self.btn_save_current)
        
        self.btn_save = QPushButton("💾 Save All Exports")
        self.btn_save.clicked.connect(self._on_save_all)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("font-weight: bold;")
        tb2.addWidget(self.btn_save)
        
        main_layout.addLayout(tb2)
        
        # ── Thumbnails ──
        self.thumb_strip = ThumbnailStrip()
        self.thumb_strip.image_selected.connect(self._on_thumbnail)
        self.thumb_strip.setVisible(False)
        main_layout.addWidget(self.thumb_strip)
        
        # ── Main Area (Stacked) ──
        self.stack = QStackedWidget()
        
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._process_images)
        self.stack.addWidget(self.drop_zone)
        
        self.tabs = QTabWidget()
        
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Select a model folder or load a checkpoint (.pth/.pt/.safetensors) directly.")
        
        self.tab_cos = CosineTab(self.status)
        self.tab_pca = PCATab(self.status)
        self.tab_attn = AttentionTab(self.status)
        self.tab_comb = CombinedPCATab(self.status)
        
        self.tabs.addTab(self.tab_cos, "Cosine Similarity")
        self.tabs.addTab(self.tab_pca, "PCA Feature Maps")
        self.tabs.addTab(self.tab_attn, "Attention Heads")
        self.tabs.addTab(self.tab_comb, "Combined PCA")
        
        self.stack.addWidget(self.tabs)
        main_layout.addWidget(self.stack, stretch=1)

    def _refresh_arch_list(self):
        current = self.combo_arch.currentText()
        self.combo_arch.clear()
        classes = list(discover_model_classes().keys())
        self.combo_arch.addItems(classes)
        if current in classes:
            self.combo_arch.setCurrentText(current)

    def _on_custom_guide(self):
        dlg = CustomViTGuideDialog(self)
        dlg.exec_()
        self._refresh_arch_list()

    def _on_select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Models Folder")
        if not folder: return
        
        models = discover_models_in_folder(folder)
        self.combo_models.clear()
        if not models:
            self.combo_models.addItem("No models found")
            return
            
        self._discovered_models = models
        for m in models:
            acc = f" ({m['best_acc']:.1f}%)" if m['best_acc'] > 0 else ""
            self.combo_models.addItem(f"{m['name']}{acc}")
            
    def _on_load_model(self):
        idx = self.combo_models.currentIndex()
        if idx < 0 or not hasattr(self, '_discovered_models') or idx >= len(self._discovered_models):
            return
        pth = self._discovered_models[idx]["path"]
        self._do_load(pth)
        
    def _on_load_direct(self):
        pth, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            "",
            "Model Checkpoints (*.pth *.pt *.bin *.safetensors);;All Files (*)",
        )
        if pth:
            self._do_load(pth)
            
    def _do_load(self, pth: str):
        self._refresh_arch_list()
        dev = self.combo_device.currentText()
        arch_name = self.combo_arch.currentText() or "VisionTransformer"
        self.status.showMessage(f"Loading model on {dev}...")
        self.repaint()
        
        try:
            model, meta = interactive_load_model(
                pth,
                device=dev,
                model_class_name=arch_name,
                parent_widget=self,
            )
            if model is None or meta is None:
                self.status.showMessage("Model loading cancelled.")
                return

            self._model = model
            self._meta = meta
            self._device = dev
            
            exp_name = os.path.splitext(os.path.basename(pth))[0]
            parent_dir = os.path.basename(os.path.dirname(pth))
            self._model_name = f"{parent_dir}_{exp_name}" if parent_dir else exp_name
            
            cls_used = meta.get("model_class", arch_name)
            pos_type = meta.get("raw_config", {}).get("pos_embed_type", "rope")
            self.status.showMessage(
                f"✓ Loaded {self._model_name} [{cls_used} | patch={meta['patch_size']} | "
                f"img={meta['img_size']} | pos={pos_type}] on {dev}"
            )
            self.btn_load_img.setEnabled(True)
        except Exception as e:
            self.status.showMessage(f"✗ Load failed: {e}")
            QMessageBox.critical(self, "Error", str(e))
            
    def _on_load_images(self):
        if not self._model: return
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if paths:
            self._process_images(paths)
            
    def _process_images(self, paths: list):
        if not self._model: return
        
        self._results.clear()
        
        res_txt = self.combo_res.currentText()
        target_res = int(res_txt) if res_txt != "original" else None
        
        mode = "standard"
        if self.rad_nat.isChecked(): mode = "native"
        elif self.rad_sld.isChecked(): mode = "sliding"
        
        display_imgs = []
        filenames = []
        
        for i, p in enumerate(paths):
            fname = os.path.basename(p)
            self.status.showMessage(f"Processing {i+1}/{len(paths)}: {fname} [{mode}]...")
            self.repaint()
            
            img = Image.open(p).convert("RGB")
            
            if mode == "native":
                res = inference_native(self._model, img, self._meta, self._device, fname, target_res)
            elif mode == "sliding":
                res = inference_sliding_window(self._model, img, self._meta, self._device, fname, target_res)
            else:
                res = inference_standard(self._model, img, self._meta, self._device, fname)
                
            self._results.append(res)
            display_imgs.append(res.img_display)
            filenames.append(fname)
            
        if len(self._results) > 1:
            self.thumb_strip.set_thumbnails(display_imgs, filenames)
            self.thumb_strip.setVisible(True)
        else:
            self.thumb_strip.setVisible(False)
            
        self.tab_comb.set_results(self._results)
            
        self._show_result(0)
        self.btn_save.setEnabled(True)
        self.btn_save_current.setEnabled(True)
        self.status.showMessage(f"✓ Loaded {len(paths)} image(s).")
        
    def _on_thumbnail(self, idx: int):
        self._show_result(idx)
        
    def _show_result(self, idx: int):
        if idx < 0 or idx >= len(self._results): return
        self.stack.setCurrentIndex(1) # Switch from drop zone to tabs
        self._active_idx = idx
        
        res = self._results[idx]
        self.tab_cos.set_result(res)
        self.tab_pca.set_result(res)
        self.tab_attn.set_result(res)
        
        if len(self._results) > 1:
            self.thumb_strip.set_active(idx)
            
    def _on_save_current(self):
        if self._active_idx < 0: return
        
        current_tab_idx = self.tabs.currentIndex()
        
        if current_tab_idx == 3:
            # For Combined PCA, we want to download all rendered maps
            if not self.tab_comb._combined_maps:
                self.status.showMessage("No combined PCA maps computed yet.")
                return
                
            path = QFileDialog.getExistingDirectory(self, "Select Directory to Save All Combined PCA Maps")
            if not path: return
            
            saved = 0
            for i, comb_pm in self.tab_comb._combined_maps.items():
                if not comb_pm.isNull():
                    fname = os.path.splitext(self._results[i].filename)[0]
                    save_path = os.path.join(path, f"{fname}_combined_pca.png")
                    save_qpixmap(comb_pm, save_path)
                    saved += 1
                    
            self.status.showMessage(f"✓ Saved {saved} combined PCA maps to {path}")
            return
            
        res = self._results[self._active_idx]
        img_name = os.path.splitext(res.filename)[0]
        
        default_name = f"{img_name}_export.png"
        
        if current_tab_idx == 0:
            default_name = f"{img_name}_cosine.png"
        elif current_tab_idx == 1:
            default_name = f"{img_name}_pca.png"
        elif current_tab_idx == 2:
            default_name = f"{img_name}_attention.png"
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Current Tab", default_name, "PNG Images (*.png)")
        if not path: return
        
        if current_tab_idx == 0:
            save_qpixmap(self.tab_cos.heatmap_panel.get_composite_image(), path)
        elif current_tab_idx == 1:
            if hasattr(self.tab_pca, '_pca_pixmap'):
                save_qpixmap(self.tab_pca._pca_pixmap, path)
        elif current_tab_idx == 2:
            attn_bytes = self.tab_attn.get_rendered_bytes()
            if attn_bytes:
                with open(path, "wb") as f:
                    f.write(attn_bytes)
                
        self.status.showMessage(f"✓ Saved to {path}")
            
    def _on_save_all(self):
        if self._active_idx < 0: return
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualized", self._model_name))
        
        # Save current active image
        res = self._results[self._active_idx]
        img_name = os.path.splitext(res.filename)[0]
        out_dir = os.path.join(base_dir, img_name)
        ensure_dir(out_dir)
        
        self.status.showMessage(f"Saving exports to {out_dir}...")
        self.repaint()
        
        # 1. Original
        save_image_array(res.img_display, os.path.join(out_dir, "original.png"))
        
        # 2. Cosine Sim
        save_qpixmap(self.tab_cos.heatmap_panel.get_composite_image(), os.path.join(out_dir, "cosine_sim_overlay.png"))
        save_qpixmap(self.tab_cos.heatmap_panel.get_heatmap_only(), os.path.join(out_dir, "cosine_sim_heatmap.png"))
        
        # 3. PCA
        if hasattr(self.tab_pca, '_pca_pixmap'):
            save_qpixmap(self.tab_pca._pca_pixmap, os.path.join(out_dir, "pca_map.png"))
            # PDF requires array. Re-build it since pixmap -> array is annoying
            from app.utils.pca import build_pca_map
            pca_arr = build_pca_map(res.patch_tokens, res.h_feat, res.w_feat, self.tab_pca.slider.value() / 100.0, self.tab_pca.invert_btn.isChecked())
            save_pdf_from_array(pca_arr, os.path.join(out_dir, "pca_map.pdf"), interpolation='nearest')
            
        # 4. Attention
        attn_bytes = self.tab_attn.get_rendered_bytes()
        if attn_bytes:
            with open(os.path.join(out_dir, "attention_heads.png"), "wb") as f:
                f.write(attn_bytes)
            save_pdf_from_bytes(attn_bytes, os.path.join(out_dir, "attention_heads.pdf"))
            
        # 5. Combined PCA (if it exists for this index)
        comb_pm = self.tab_comb.get_combined_map(self._active_idx)
        if not comb_pm.isNull():
            save_qpixmap(comb_pm, os.path.join(out_dir, "combined_pca.png"))
            
        self.status.showMessage(f"✓ Saved exports to {out_dir}")
