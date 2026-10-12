import time
from typing import Optional

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QCheckBox, QSpinBox, QSplitter
from PyQt5.QtCore import Qt, QTimer

from app.widgets.image_panel import ImagePanel
from app.inference import InferenceResult

class CosineTab(QWidget):
    def __init__(self, status_bar=None, parent=None):
        super().__init__(parent)
        self.status_bar = status_bar
        self._current_result: Optional[InferenceResult] = None
        
        self._build_ui()
        
        self._hover_timer = QTimer()
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(8)  # ~120fps cap
        self._hover_timer.timeout.connect(self._do_hover_update)
        self._pending_hover = None
        
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        
        # Controls Row 1
        ctrl = QHBoxLayout()
        
        ctrl.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Hover", "Click"])
        ctrl.addWidget(self._mode_combo)
        
        ctrl.addWidget(QLabel("  α:"))
        self._alpha_slider = QSlider(Qt.Horizontal)
        self._alpha_slider.setRange(0, 100)
        self._alpha_slider.setValue(70)
        self._alpha_slider.setFixedWidth(100)
        self._alpha_slider.valueChanged.connect(self._on_alpha)
        ctrl.addWidget(self._alpha_slider)
        
        ctrl.addWidget(QLabel("  Map:"))
        self._cmap_combo = QComboBox()
        from app.utils.colormaps import COLORMAPS
        self._cmap_combo.addItems(COLORMAPS)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap)
        ctrl.addWidget(self._cmap_combo)
        
        self._grid_cb = QCheckBox("Grid")
        self._grid_cb.toggled.connect(self._on_grid)
        ctrl.addWidget(self._grid_cb)
        
        self._smooth_cb = QCheckBox("Smooth")
        self._smooth_cb.toggled.connect(self._on_smooth)
        ctrl.addWidget(self._smooth_cb)
        
        ctrl.addWidget(QLabel("  ✛ Color:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(["#00bcd4", "red", "green", "blue", "white", "black"])
        self._color_combo.currentTextChanged.connect(self._on_crosshair)
        ctrl.addWidget(self._color_combo)
        
        ctrl.addWidget(QLabel(" Size:"))
        self._crosshair_size = QSpinBox()
        self._crosshair_size.setRange(1, 999)
        self._crosshair_size.setValue(6)
        self._crosshair_size.valueChanged.connect(self._on_crosshair)
        ctrl.addWidget(self._crosshair_size)
        
        ctrl.addWidget(QLabel(" Thick:"))
        self._crosshair_thick = QSpinBox()
        self._crosshair_thick.setRange(1, 999)
        self._crosshair_thick.setValue(1)
        self._crosshair_thick.valueChanged.connect(self._on_crosshair)
        ctrl.addWidget(self._crosshair_thick)
        
        ctrl.addStretch()
        layout.addLayout(ctrl)
        
        # Splitter for images
        splitter = QSplitter(Qt.Horizontal)
        self.orig_panel = ImagePanel()
        self.heatmap_panel = ImagePanel()
        
        self.orig_panel.patch_hovered.connect(self._on_hover)
        self.heatmap_panel.patch_hovered.connect(self._on_hover)
        self.orig_panel.patch_clicked.connect(self._on_click)
        self.heatmap_panel.patch_clicked.connect(self._on_click)
        
        splitter.addWidget(self.orig_panel)
        splitter.addWidget(self.heatmap_panel)
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter, stretch=1)
        
        # Apply initial settings
        self._on_alpha(self._alpha_slider.value())
        self._on_cmap(self._cmap_combo.currentText())

    def set_result(self, res: InferenceResult):
        self._current_result = res
        self.orig_panel.set_base_image(res.img_display, res.h_feat, res.w_feat)
        self.heatmap_panel.set_base_image(res.img_display, res.h_feat, res.w_feat)
        
        # Auto-center
        r, c = res.h_feat // 2, res.w_feat // 2
        self._update_heatmap(r, c)
        
    def _on_alpha(self, val):
        self.heatmap_panel.set_alpha(val / 100.0)
        
    def _on_cmap(self, name):
        self.heatmap_panel.set_colormap(name)
        if self._current_result and self._pending_hover:
            self._update_heatmap(self._pending_hover[0], self._pending_hover[1])
            
    def _on_grid(self, checked):
        self.orig_panel.set_show_grid(checked)
        self.heatmap_panel.set_show_grid(checked)
        
    def _on_smooth(self, checked):
        self.orig_panel.set_smooth_scaling(checked)
        self.heatmap_panel.set_smooth_scaling(checked)
        
    def _on_crosshair(self):
        s = self._crosshair_size.value()
        t = self._crosshair_thick.value()
        c = self._color_combo.currentText()
        for p in (self.orig_panel, self.heatmap_panel):
            p._crosshair_size = s
            p._crosshair_thickness = t
            p.set_crosshair_color(c)
            p._repaint()

    def _on_hover(self, row, col):
        if self._mode_combo.currentText() == "Hover":
            self._pending_hover = (row, col)
            if not self._hover_timer.isActive():
                self._hover_timer.start()
                
    def _on_click(self, row, col):
        if self._mode_combo.currentText() == "Click":
            self._pending_hover = (row, col)
            self._do_hover_update()

    def _do_hover_update(self):
        if self._pending_hover:
            self._update_heatmap(*self._pending_hover)

    def _update_heatmap(self, row, col):
        res = self._current_result
        if not res: return
        
        t0 = time.perf_counter()
        
        patch_idx = row * res.w_feat + col
        if patch_idx < 0 or patch_idx >= res.normed_tokens.shape[0]: return
        
        # O(N x D)
        sim_row = res.normed_tokens[patch_idx] @ res.normed_tokens.T
        self.heatmap_panel.set_heatmap_from_sim_row(sim_row)
        
        self.orig_panel.set_crosshair(row, col)
        self.heatmap_panel.set_crosshair(row, col)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        min_v, max_v = sim_row.min(), sim_row.max()
        
        if self.status_bar:
            self.status_bar.showMessage(
                f"Cosine Sim | Patch ({row}, {col}) | Range: [{min_v:.3f}, {max_v:.3f}] | "
                f"Render: {elapsed_ms:.1f}ms | Grid: {res.h_feat}×{res.w_feat}"
            )
