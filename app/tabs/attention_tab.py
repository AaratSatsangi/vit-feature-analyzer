import time
from typing import Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np

from app.inference import InferenceResult
from app.utils.attention import render_attention_grid

class AttentionTab(QWidget):
    def __init__(self, status_bar=None, parent=None):
        super().__init__(parent)
        self.status_bar = status_bar
        self._current_result: Optional[InferenceResult] = None
        self._build_ui()
        
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        
        self.img_lbl = QLabel("No attention data.")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.img_lbl)
        
        layout.addWidget(self.scroll)

    def set_result(self, res: InferenceResult):
        self._current_result = res
        
        if res.attn_cls is None:
            self.img_lbl.setText("Attention not computed.\n(Skipped in Sliding Window mode or disabled)")
            return
            
        t0 = time.perf_counter()
        
        # Render matplotlib grid as PNG bytes
        png_bytes = render_attention_grid(res.img_display, res.attn_cls, res.h_feat, res.w_feat)
        
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes)
        
        self.img_lbl.setPixmap(pixmap)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self.status_bar:
            self.status_bar.showMessage(f"Attention Grid | Rendered in {elapsed_ms:.1f}ms")
            
    def get_rendered_bytes(self) -> Optional[bytes]:
        res = self._current_result
        if not res or res.attn_cls is None: return None
        return render_attention_grid(res.img_display, res.attn_cls, res.h_feat, res.w_feat)
