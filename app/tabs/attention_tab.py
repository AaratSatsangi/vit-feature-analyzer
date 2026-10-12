import time
from typing import Optional
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QPushButton,
    QSlider,
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap

from app.inference import InferenceResult
from app.utils.attention import render_attention_grid


class AttentionTab(QWidget):
    def __init__(self, status_bar=None, parent=None):
        super().__init__(parent)
        self.status_bar = status_bar
        self._current_result: Optional[InferenceResult] = None
        self._raw_pixmap: Optional[QPixmap] = None
        self._rendered_bytes: Optional[bytes] = None
        self._fit_mode: bool = True
        self._zoom_pct: int = 100
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)

        # Controls Row: Zoom In / Out / Fit
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Zoom:"))

        self.btn_zoom_out = QPushButton("➖")
        self.btn_zoom_out.setFixedWidth(36)
        self.btn_zoom_out.setToolTip("Zoom Out (or Ctrl + Mouse Wheel Down)")
        self.btn_zoom_out.clicked.connect(lambda: self._step_zoom(-15))
        ctrl.addWidget(self.btn_zoom_out)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(15, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(180)
        self.zoom_slider.valueChanged.connect(self._on_slider_zoom)
        ctrl.addWidget(self.zoom_slider)

        self.btn_zoom_in = QPushButton("➕")
        self.btn_zoom_in.setFixedWidth(36)
        self.btn_zoom_in.setToolTip("Zoom In (or Ctrl + Mouse Wheel Up)")
        self.btn_zoom_in.clicked.connect(lambda: self._step_zoom(15))
        ctrl.addWidget(self.btn_zoom_in)

        self.zoom_lbl = QLabel("Fit")
        self.zoom_lbl.setFixedWidth(55)
        self.zoom_lbl.setStyleSheet("color: #00bcd4; font-weight: bold;")
        ctrl.addWidget(self.zoom_lbl)

        self.btn_fit = QPushButton("⛶ Fit to Window")
        self.btn_fit.setCheckable(True)
        self.btn_fit.setChecked(True)
        self.btn_fit.clicked.connect(self._on_fit_clicked)
        ctrl.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("1:1 (100%)")
        self.btn_100.clicked.connect(lambda: self._set_explicit_zoom(100))
        ctrl.addWidget(self.btn_100)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Scroll Area with White Background
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet(
            "QScrollArea { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; }"
        )
        self.scroll.viewport().setStyleSheet("background-color: #ffffff;")
        self.scroll.viewport().installEventFilter(self)

        self.img_lbl = QLabel("No attention data.")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setStyleSheet("background-color: #ffffff; color: #0f172a; padding: 12px;")
        self.scroll.setWidget(self.img_lbl)

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

    def set_result(self, res: InferenceResult):
        self._current_result = res

        if res.attn_cls is None:
            self._raw_pixmap = None
            self._rendered_bytes = None
            self.img_lbl.setPixmap(QPixmap())
            self.img_lbl.setText("Attention not computed.\n(Skipped in Sliding Window mode or disabled)")
            return

        t0 = time.perf_counter()

        # Render matplotlib grid as PNG bytes on white background
        self._rendered_bytes = render_attention_grid(res.img_display, res.attn_cls, res.h_feat, res.w_feat)

        pixmap = QPixmap()
        pixmap.loadFromData(self._rendered_bytes)
        self._raw_pixmap = pixmap

        # Always open fitted to the window so the entire plot is visible immediately
        self._fit_mode = True
        self.btn_fit.setChecked(True)
        self._update_pixmap_display()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self.status_bar:
            self.status_bar.showMessage(f"Attention Grid | Rendered in {elapsed_ms:.1f}ms")

    def _compute_fit_zoom_pct(self) -> int:
        if not self._raw_pixmap or self._raw_pixmap.isNull():
            return 100
        vw = max(100, self.scroll.viewport().width() - 28)
        vh = max(100, self.scroll.viewport().height() - 28)
        pw = self._raw_pixmap.width()
        ph = self._raw_pixmap.height()
        if pw <= 0 or ph <= 0:
            return 100
        scale = min(vw / pw, vh / ph)
        return max(15, min(300, int(scale * 100)))

    def _on_fit_clicked(self):
        self._fit_mode = True
        self.btn_fit.setChecked(True)
        self._update_pixmap_display()

    def _set_explicit_zoom(self, pct: int):
        self._fit_mode = False
        self.btn_fit.setChecked(False)
        pct = max(15, min(300, pct))
        self._zoom_pct = pct
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(pct)
        self.zoom_slider.blockSignals(False)
        self._update_pixmap_display()

    def _step_zoom(self, delta: int):
        base = self._compute_fit_zoom_pct() if self._fit_mode else self._zoom_pct
        self._set_explicit_zoom(base + delta)

    def _on_slider_zoom(self, val: int):
        self._fit_mode = False
        self.btn_fit.setChecked(False)
        self._zoom_pct = val
        self._update_pixmap_display()

    def _update_pixmap_display(self):
        if not self._raw_pixmap or self._raw_pixmap.isNull():
            return

        if self._fit_mode:
            pct = self._compute_fit_zoom_pct()
            self._zoom_pct = pct
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(pct)
            self.zoom_slider.blockSignals(False)
            self.zoom_lbl.setText(f"Fit ({pct}%)")
        else:
            pct = self._zoom_pct
            self.zoom_lbl.setText(f"{pct}%")

        target_w = max(1, int(self._raw_pixmap.width() * (pct / 100.0)))
        target_h = max(1, int(self._raw_pixmap.height() * (pct / 100.0)))
        scaled = self._raw_pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.img_lbl.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self._update_pixmap_display()

    def showEvent(self, event):
        super().showEvent(event)
        if self._fit_mode:
            self._update_pixmap_display()

    def get_rendered_bytes(self) -> Optional[bytes]:
        if self._rendered_bytes is not None:
            return self._rendered_bytes
        res = self._current_result
        if not res or res.attn_cls is None:
            return None
        return render_attention_grid(res.img_display, res.attn_cls, res.h_feat, res.w_feat)
