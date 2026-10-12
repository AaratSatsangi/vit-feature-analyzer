from typing import List
import numpy as np

from PyQt5.QtWidgets import QScrollArea, QWidget, QHBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QSize

class ThumbnailStrip(QScrollArea):
    image_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedHeight(100)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._layout = QHBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(10)
        self.setWidget(self._container)
        
        self._buttons: list = []
        self._active_idx = -1

    def set_thumbnails(self, images: List[np.ndarray], filenames: List[str]):
        for b in self._buttons:
            self._layout.removeWidget(b)
            b.deleteLater()
        self._buttons.clear()

        for i, (img, fname) in enumerate(zip(images, filenames)):
            btn = QPushButton()
            btn.setFixedSize(72, 72)
            btn.setToolTip(fname)

            thumb = (img * 255).clip(0, 255).astype(np.uint8)
            h, w = thumb.shape[:2]
            thumb = np.ascontiguousarray(thumb)
            qimg = QImage(thumb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            pm = QPixmap.fromImage(qimg).scaled(64, 64, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            btn.setIcon(QIcon(pm))
            btn.setIconSize(QSize(64, 64))

            idx = i
            btn.clicked.connect(lambda checked, ii=idx: self._on_click(ii))
            self._layout.addWidget(btn)
            self._buttons.append(btn)

        self._layout.addStretch()
        self._update_styles()

    def _on_click(self, idx):
        self._active_idx = idx
        self._update_styles()
        self.image_selected.emit(idx)

    def set_active(self, idx):
        self._active_idx = idx
        self._update_styles()

    def _update_styles(self):
        for i, btn in enumerate(self._buttons):
            if i == self._active_idx:
                btn.setStyleSheet("""
                    QPushButton {
                        border: 3px solid #00bcd4; 
                        border-radius: 8px; 
                        background: #1e293b;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        border: 1px solid #334155; 
                        border-radius: 8px; 
                        background: #0f172a;
                    }
                    QPushButton:hover {
                        border-color: rgba(0, 188, 212, 0.5);
                    }
                """)
