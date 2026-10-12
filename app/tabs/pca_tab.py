import time
import numpy as np
from typing import Optional

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton, QSplitter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from app.inference import InferenceResult
from app.utils.pca import build_pca_map

class PCATab(QWidget):
    def __init__(self, status_bar=None, parent=None):
        super().__init__(parent)
        self.status_bar = status_bar
        self._current_result: Optional[InferenceResult] = None
        
        self._build_ui()
        
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        
        ctrl = QHBoxLayout()
        
        ctrl.addWidget(QLabel("BG Threshold:"))
        self.thresh_val = QLabel("0.00")
        self.thresh_val.setFixedWidth(40)
        self.thresh_val.setStyleSheet("color: #00bcd4; font-weight: bold;")
        ctrl.addWidget(self.thresh_val)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-100, 100)
        self.slider.setValue(0)
        self.slider.setFixedWidth(200)
        self.slider.valueChanged.connect(self._on_slider_move)
        # ONLY compute on release to prevent lag
        self.slider.sliderReleased.connect(self._compute_pca)
        ctrl.addWidget(self.slider)
        
        self.invert_btn = QPushButton("⇄ Flip Mask")
        self.invert_btn.setCheckable(True)
        self.invert_btn.clicked.connect(self._compute_pca)
        ctrl.addWidget(self.invert_btn)
        
        self.smooth_cb = QCheckBox("Smooth")
        self.smooth_cb.toggled.connect(self._on_smooth)
        ctrl.addWidget(self.smooth_cb)
        
        ctrl.addStretch()
        layout.addLayout(ctrl)
        
        splitter = QSplitter(Qt.Horizontal)
        
        self.orig_lbl = QLabel()
        self.orig_lbl.setAlignment(Qt.AlignCenter)
        self.pca_lbl = QLabel()
        self.pca_lbl.setAlignment(Qt.AlignCenter)
        
        splitter.addWidget(self.orig_lbl)
        splitter.addWidget(self.pca_lbl)
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter, stretch=1)
        
    def _on_slider_move(self, val):
        self.thresh_val.setText(f"{val/100.0:.2f}")

    def set_result(self, res: InferenceResult):
        self._current_result = res
        
        # Original Image
        img_uint8 = (res.img_display * 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        img_uint8 = np.ascontiguousarray(img_uint8)
        qimg = QImage(img_uint8.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._orig_pixmap = QPixmap.fromImage(qimg)
        self._update_display()
        
        # Restore state if previously adjusted
        self.slider.blockSignals(True)
        self.slider.setValue(int(res.pca_threshold * 100))
        self.thresh_val.setText(f"{res.pca_threshold:.2f}")
        self.invert_btn.setChecked(res.pca_inverted)
        self.slider.blockSignals(False)
        
        self._compute_pca()
        
    def _compute_pca(self):
        res = self._current_result
        if not res: return
        
        t0 = time.perf_counter()
        thresh = self.slider.value() / 100.0
        inv = self.invert_btn.isChecked()
        
        # Save to result so Combined PCA tab can access it
        res.pca_threshold = thresh
        res.pca_inverted = inv
        
        pca_arr = build_pca_map(res.patch_tokens, res.h_feat, res.w_feat, thresh, inv)
        
        pca_uint8 = (pca_arr * 255).astype(np.uint8)
        ph, pw = pca_uint8.shape[:2]
        pca_uint8 = np.ascontiguousarray(pca_uint8)
        qimg = QImage(pca_uint8.tobytes(), pw, ph, 3 * pw, QImage.Format_RGB888)
        self._pca_pixmap = QPixmap.fromImage(qimg)
        self._update_display()
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self.status_bar:
            self.status_bar.showMessage(f"PCA Feature Map | Computed in {elapsed_ms:.1f}ms")

    def _on_smooth(self):
        self._update_display()

    def _update_display(self):
        if not hasattr(self, '_orig_pixmap'): return
        
        mode = Qt.SmoothTransformation if self.smooth_cb.isChecked() else Qt.FastTransformation
        
        sz = self.orig_lbl.size()
        if sz.width() > 0 and sz.height() > 0:
            self.orig_lbl.setPixmap(self._orig_pixmap.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if hasattr(self, '_pca_pixmap'):
                self.pca_lbl.setPixmap(self._pca_pixmap.scaled(self.pca_lbl.size(), Qt.KeepAspectRatio, mode))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()
