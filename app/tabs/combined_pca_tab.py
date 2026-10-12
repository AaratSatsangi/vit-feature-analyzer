import time
from typing import List
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QScrollArea, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from app.inference import InferenceResult
from app.utils.pca import build_combined_pca_maps

class CombinedPCATab(QWidget):
    def __init__(self, status_bar=None, parent=None):
        super().__init__(parent)
        self.status_bar = status_bar
        self._results: List[InferenceResult] = []
        self._combined_maps: dict = {} # map idx -> QPixmap
        self._build_ui()
        
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Class Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["All Same Class", "Split by Filename"])
        ctrl.addWidget(self.mode_combo)
        
        self.compute_btn = QPushButton("Compute Combined PCA")
        self.compute_btn.setStyleSheet("background-color: #00bcd4; color: #000; font-weight: bold;")
        self.compute_btn.clicked.connect(self._compute)
        ctrl.addWidget(self.compute_btn)
        
        ctrl.addStretch()
        layout.addLayout(ctrl)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QVBoxLayout(self.content)
        self.grid.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, stretch=1)
        
    def set_results(self, results: List[InferenceResult]):
        self._results = results
        self._clear_grid()
        self._combined_maps.clear()
        
    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Not doing full recursive delete for simplicity, as we mostly add widgets
                pass
                
    def _compute(self):
        if not self._results: return
        t0 = time.perf_counter()
        
        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("Computing...")
        self.repaint()
        
        try:
            # Group by class
            mode = self.mode_combo.currentText()
            groups = {}
            for i, res in enumerate(self._results):
                cls = "all"
                if mode == "Split by Filename":
                    cls = res.filename.split('-')[0] if '-' in res.filename else 'unclassified'
                if cls not in groups:
                    groups[cls] = []
                groups[cls].append(i)
                
            self._clear_grid()
            self._combined_maps.clear()
            
            for cls, indices in groups.items():
                # Add Header
                hdr = QLabel(f"Class: {cls} ({len(indices)} images)")
                hdr.setStyleSheet("font-size: 14px; font-weight: bold; color: #00bcd4; margin-top: 10px;")
                self.grid.addWidget(hdr)
                
                # Compute joint PCA for this class
                group_results = [self._results[i] for i in indices]
                thresholds = [{"t": res.pca_threshold, "inv": res.pca_inverted} for res in group_results]
                
                joint_maps = build_combined_pca_maps(group_results, thresholds)
                
                # Display horizontally wrapped
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setAlignment(Qt.AlignLeft)
                
                for i, img_map in zip(indices, joint_maps):
                    res = self._results[i]
                    
                    # Store map
                    pca_uint8 = (img_map * 255).astype(np.uint8)
                    ph, pw = pca_uint8.shape[:2]
                    pca_uint8 = np.ascontiguousarray(pca_uint8)
                    qimg = QImage(pca_uint8.tobytes(), pw, ph, 3 * pw, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    self._combined_maps[i] = pixmap
                    
                    # UI Box
                    box = QWidget()
                    box_layout = QVBoxLayout(box)
                    box_layout.setContentsMargins(0, 0, 0, 0)
                    
                    lbl = QLabel(res.filename)
                    lbl.setAlignment(Qt.AlignCenter)
                    lbl.setStyleSheet("font-size: 10px; color: #94a3b8;")
                    box_layout.addWidget(lbl)
                    
                    img_lbl = QLabel()
                    img_lbl.setFixedSize(200, 200)
                    img_lbl.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.FastTransformation))
                    img_lbl.setAlignment(Qt.AlignCenter)
                    img_lbl.setStyleSheet("border: 1px solid #334155; background: #0f172a;")
                    box_layout.addWidget(img_lbl)
                    
                    row_layout.addWidget(box)
                    
                self.grid.addWidget(row_widget)
                
        finally:
            self.compute_btn.setEnabled(True)
            self.compute_btn.setText("Compute Combined PCA")
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self.status_bar:
            self.status_bar.showMessage(f"Combined PCA | Computed in {elapsed_ms:.1f}ms")

    def get_combined_map(self, idx: int) -> QPixmap:
        return self._combined_maps.get(idx, QPixmap())
