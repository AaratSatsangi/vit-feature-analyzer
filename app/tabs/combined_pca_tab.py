import time
from typing import Dict, List, Tuple
import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QScrollArea,
    QSlider,
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap

from app.inference import InferenceResult
from app.utils.pca import build_combined_pca_maps


class CombinedPCATab(QWidget):
    def __init__(self, status_bar=None, parent=None):
        super().__init__(parent)
        self.status_bar = status_bar
        self._results: List[InferenceResult] = []
        self._combined_maps: Dict[int, QPixmap] = {}  # map idx -> full-res QPixmap
        self._image_labels: List[Tuple[QLabel, QPixmap]] = []
        self._zoom_pct: int = 100
        self._base_card_size: int = 240
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Class Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["All Same Class", "Split by Filename"])
        ctrl.addWidget(self.mode_combo)

        self.compute_btn = QPushButton("Compute Combined PCA")
        self.compute_btn.setStyleSheet("background-color: #00bcd4; color: #000; font-weight: bold;")
        self.compute_btn.clicked.connect(self._compute)
        ctrl.addWidget(self.compute_btn)

        ctrl.addWidget(QLabel("  │  Zoom:"))

        self.btn_zoom_out = QPushButton("➖")
        self.btn_zoom_out.setFixedWidth(36)
        self.btn_zoom_out.setToolTip("Zoom Out (or Mouse Wheel Down)")
        self.btn_zoom_out.clicked.connect(lambda: self._step_zoom(-15))
        ctrl.addWidget(self.btn_zoom_out)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(160)
        self.zoom_slider.valueChanged.connect(self._on_slider_zoom)
        ctrl.addWidget(self.zoom_slider)

        self.btn_zoom_in = QPushButton("➕")
        self.btn_zoom_in.setFixedWidth(36)
        self.btn_zoom_in.setToolTip("Zoom In (or Mouse Wheel Up)")
        self.btn_zoom_in.clicked.connect(lambda: self._step_zoom(15))
        ctrl.addWidget(self.btn_zoom_in)

        self.zoom_lbl = QLabel("100%")
        self.zoom_lbl.setFixedWidth(48)
        self.zoom_lbl.setStyleSheet("color: #00bcd4; font-weight: bold;")
        ctrl.addWidget(self.zoom_lbl)

        self.btn_reset_zoom = QPushButton("⛶ Reset (100%)")
        self.btn_reset_zoom.clicked.connect(lambda: self._set_zoom(100))
        ctrl.addWidget(self.btn_reset_zoom)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Scroll Area & Content Canvas with White Background
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(
            "QScrollArea { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; }"
        )
        self.scroll.viewport().setStyleSheet("background-color: #ffffff;")
        self.scroll.viewport().installEventFilter(self)

        self.content = QWidget()
        self.content.setStyleSheet("background-color: #ffffff; color: #0f172a;")
        self.grid = QVBoxLayout(self.content)
        self.grid.setAlignment(Qt.AlignTop)
        self.grid.setContentsMargins(16, 16, 16, 16)
        self.grid.setSpacing(16)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, stretch=1)

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() & Qt.ControlModifier or True:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._step_zoom(10)
                    return True
                elif delta < 0:
                    self._step_zoom(-10)
                    return True
        return super().eventFilter(obj, event)

    def _set_zoom(self, pct: int):
        pct = max(25, min(300, pct))
        self._zoom_pct = pct
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(pct)
        self.zoom_slider.blockSignals(False)
        self.zoom_lbl.setText(f"{pct}%")
        self._rescale_cards()

    def _step_zoom(self, delta: int):
        self._set_zoom(self._zoom_pct + delta)

    def _on_slider_zoom(self, val: int):
        self._zoom_pct = val
        self.zoom_lbl.setText(f"{val}%")
        self._rescale_cards()

    def _rescale_cards(self):
        side = max(60, int(self._base_card_size * (self._zoom_pct / 100.0)))
        for img_lbl, pixmap in self._image_labels:
            img_lbl.setFixedSize(side, side)
            img_lbl.setPixmap(pixmap.scaled(side, side, Qt.KeepAspectRatio, Qt.FastTransformation))

    def set_results(self, results: List[InferenceResult]):
        self._results = results
        self._clear_grid()
        self._combined_maps.clear()

    def _clear_grid(self):
        self._image_labels.clear()
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _compute(self):
        if not self._results:
            return
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
                    cls = res.filename.split("-")[0] if "-" in res.filename else "unclassified"
                if cls not in groups:
                    groups[cls] = []
                groups[cls].append(i)

            self._clear_grid()
            self._combined_maps.clear()

            side = max(60, int(self._base_card_size * (self._zoom_pct / 100.0)))

            for cls, indices in groups.items():
                # Add Header on white background
                hdr = QLabel(f"Class: {cls} ({len(indices)} images)")
                hdr.setStyleSheet(
                    "font-size: 14px; font-weight: bold; color: #0f172a; "
                    "background: #ffffff; margin-top: 6px;"
                )
                self.grid.addWidget(hdr)

                # Compute joint PCA for this class
                group_results = [self._results[i] for i in indices]
                thresholds = [{"t": res.pca_threshold, "inv": res.pca_inverted} for res in group_results]

                joint_maps = build_combined_pca_maps(group_results, thresholds)

                # Display horizontally
                row_widget = QWidget()
                row_widget.setStyleSheet("background-color: #ffffff;")
                row_layout = QHBoxLayout(row_widget)
                row_layout.setAlignment(Qt.AlignLeft)
                row_layout.setSpacing(14)

                for i, img_map in zip(indices, joint_maps):
                    res = self._results[i]

                    # Store map
                    pca_uint8 = (img_map * 255).astype(np.uint8)
                    ph, pw = pca_uint8.shape[:2]
                    pca_uint8 = np.ascontiguousarray(pca_uint8)
                    qimg = QImage(pca_uint8.tobytes(), pw, ph, 3 * pw, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    self._combined_maps[i] = pixmap

                    # UI Box on white background
                    box = QWidget()
                    box.setStyleSheet("background-color: #ffffff;")
                    box_layout = QVBoxLayout(box)
                    box_layout.setContentsMargins(4, 4, 4, 4)
                    box_layout.setSpacing(4)

                    lbl = QLabel(res.filename)
                    lbl.setAlignment(Qt.AlignCenter)
                    lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #0f172a; background: #ffffff;")
                    box_layout.addWidget(lbl)

                    img_lbl = QLabel()
                    img_lbl.setFixedSize(side, side)
                    img_lbl.setPixmap(pixmap.scaled(side, side, Qt.KeepAspectRatio, Qt.FastTransformation))
                    img_lbl.setAlignment(Qt.AlignCenter)
                    img_lbl.setStyleSheet("border: 1px solid #cbd5e1; background: #ffffff; padding: 2px;")
                    box_layout.addWidget(img_lbl)

                    self._image_labels.append((img_lbl, pixmap))
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
