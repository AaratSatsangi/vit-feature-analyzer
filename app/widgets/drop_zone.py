from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt5.QtCore import Qt, pyqtSignal

class DropZone(QLabel):
    """A visual drop zone that accepts dragged image files."""
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._set_idle_style()
        self.setText(
            "\n\n🖼️  Drag & Drop Images Here\n\n"
            "or use the 📂 Load Images button above\n\n"
            "Supported: PNG, JPG, JPEG, BMP, WebP, TIFF"
        )
        self.setFont(QFont("sans-serif", 14))

    def _set_idle_style(self):
        self.setStyleSheet("""
            DropZone {
                border: 3px dashed #334155;
                border-radius: 16px;
                background: #0f172a;
                color: #94a3b8;
                padding: 40px;
                transition: all 0.2s;
            }
        """)

    def _set_hover_style(self):
        self.setStyleSheet("""
            DropZone {
                border: 3px dashed #00bcd4;
                border-radius: 16px;
                background: rgba(0, 188, 212, 0.05);
                color: #00bcd4;
                padding: 40px;
                transition: all 0.2s;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif')):
                    event.acceptProposedAction()
                    self._set_hover_style()
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._set_idle_style()

    def dropEvent(self, event: QDropEvent):
        self._set_idle_style()
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif')):
                paths.append(path)
        if paths:
            event.acceptProposedAction()
            self.files_dropped.emit(paths)
