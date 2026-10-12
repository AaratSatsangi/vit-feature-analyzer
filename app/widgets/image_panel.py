from typing import Optional, Tuple
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, pyqtSignal

from app.utils.colormaps import apply_colormap

class ImagePanel(QLabel):
    """Displays an image with an optional cosine similarity heatmap overlay and crosshair."""

    patch_hovered = pyqtSignal(int, int)  # (row, col)
    patch_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)

        self._base_img: Optional[np.ndarray] = None  # [H, W, 3] float 0-1
        self._heatmap_rgba: Optional[np.ndarray] = None  # [grid_h, grid_w, 4] uint8
        self._grid_h = 0
        self._grid_w = 0
        self._alpha = 0.5
        self._show_grid = False
        self._crosshair_patch: Optional[Tuple[int, int]] = None  # (row, col)
        self._cmap_name = "magma"
        self._smooth_scaling = False
        self._crosshair_size = 6
        self._crosshair_thickness = 1
        self._crosshair_color = QColor(0, 188, 212, 230) # Teal accent

    def set_crosshair_color(self, hex_or_name: str):
        self._crosshair_color = QColor(hex_or_name)
        self._crosshair_color.setAlpha(230)
        self._repaint()

    def set_base_image(self, img: np.ndarray, grid_h: int, grid_w: int):
        self._base_img = img
        self._grid_h = grid_h
        self._grid_w = grid_w
        self._heatmap_rgba = None
        self._crosshair_patch = None
        self._repaint()

    def set_heatmap_from_sim_row(self, sim_row: np.ndarray):
        min_v, max_v = sim_row.min(), sim_row.max()
        if max_v - min_v < 1e-8:
            normed = np.zeros_like(sim_row)
        else:
            normed = (sim_row - min_v) / (max_v - min_v)
        rgba = apply_colormap(normed, self._cmap_name)
        self._heatmap_rgba = rgba.reshape(self._grid_h, self._grid_w, 4)
        self._repaint()

    def set_alpha(self, alpha: float):
        self._alpha = alpha
        self._repaint()

    def set_show_grid(self, show: bool):
        self._show_grid = show
        self._repaint()

    def set_colormap(self, name: str):
        self._cmap_name = name

    def set_crosshair(self, row: int, col: int):
        self._crosshair_patch = (row, col)
        self._repaint()

    def clear_crosshair(self):
        self._crosshair_patch = None
        self._repaint()

    def get_composite_image(self) -> Optional[QPixmap]:
        if self._base_img is None: return None
        return self._render_pixmap()

    def get_heatmap_only(self) -> Optional[QPixmap]:
        if self._heatmap_rgba is None: return None
        h, w = self._base_img.shape[:2]
        hm_pil = Image.fromarray(self._heatmap_rgba, "RGBA").resize((w, h), Image.NEAREST)
        hm_np = np.ascontiguousarray(np.array(hm_pil))
        qimg = QImage(hm_np.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    def _render_pixmap(self) -> QPixmap:
        if self._base_img is None: return QPixmap()

        h, w = self._base_img.shape[:2]
        base_uint8 = np.ascontiguousarray((self._base_img * 255).astype(np.uint8))
        qimg_base = QImage(base_uint8.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg_base)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)

        if self._heatmap_rgba is not None:
            hm_pil = Image.fromarray(self._heatmap_rgba, "RGBA").resize((w, h), Image.NEAREST)
            hm_np = np.ascontiguousarray(np.array(hm_pil))
            qimg_hm = QImage(hm_np.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
            painter.setOpacity(self._alpha)
            painter.drawImage(0, 0, qimg_hm)
            painter.setOpacity(1.0)

        if self._show_grid and self._grid_h > 0 and self._grid_w > 0:
            pen = QPen(QColor(255, 255, 255, 80))
            pen.setWidth(1)
            painter.setPen(pen)
            patch_px_h = h / self._grid_h
            patch_px_w = w / self._grid_w
            for r in range(1, self._grid_h):
                y = int(r * patch_px_h)
                painter.drawLine(0, y, w, y)
            for c in range(1, self._grid_w):
                x = int(c * patch_px_w)
                painter.drawLine(x, 0, x, h)

        if self._crosshair_patch is not None:
            row, col = self._crosshair_patch
            patch_px_h = h / self._grid_h
            patch_px_w = w / self._grid_w
            cx = int((col + 0.5) * patch_px_w)
            cy = int((row + 0.5) * patch_px_h)
            arm = self._crosshair_size
            
            pen = QPen(self._crosshair_color)
            pen.setWidth(self._crosshair_thickness)
            painter.setPen(pen)
            painter.drawLine(cx - arm, cy, cx + arm, cy)
            painter.drawLine(cx, cy - arm, cx, cy + arm)

        painter.end()
        return pixmap

    def set_smooth_scaling(self, smooth: bool):
        self._smooth_scaling = smooth
        self._repaint()

    def _repaint(self):
        pixmap = self._render_pixmap()
        if pixmap.isNull(): return
        mode = Qt.SmoothTransformation if self._smooth_scaling else Qt.FastTransformation
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, mode)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._repaint()

    def _pixel_to_patch(self, pos) -> Optional[Tuple[int, int]]:
        pm = self.pixmap()
        if pm is None or self._base_img is None: return None
        pw, ph = pm.width(), pm.height()
        lw, lh = self.width(), self.height()
        x_off = (lw - pw) // 2
        y_off = (lh - ph) // 2

        rel_x = pos.x() - x_off
        rel_y = pos.y() - y_off
        if rel_x < 0 or rel_y < 0 or rel_x >= pw or rel_y >= ph:
            return None

        frac_x = rel_x / pw
        frac_y = rel_y / ph
        col = max(0, min(int(frac_x * self._grid_w), self._grid_w - 1))
        row = max(0, min(int(frac_y * self._grid_h), self._grid_h - 1))
        return (row, col)

    def mouseMoveEvent(self, event):
        patch = self._pixel_to_patch(event.pos())
        if patch: self.patch_hovered.emit(patch[0], patch[1])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            patch = self._pixel_to_patch(event.pos())
            if patch: self.patch_clicked.emit(patch[0], patch[1])
