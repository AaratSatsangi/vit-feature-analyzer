from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication

def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(15, 23, 42))          # #0f172a
    palette.setColor(QPalette.WindowText, QColor(248, 250, 252))   # #f8fafc
    palette.setColor(QPalette.Base, QColor(30, 41, 59))            # #1e293b
    palette.setColor(QPalette.AlternateBase, QColor(15, 23, 42))   
    palette.setColor(QPalette.ToolTipBase, QColor(30, 41, 59))
    palette.setColor(QPalette.ToolTipText, QColor(248, 250, 252))
    palette.setColor(QPalette.Text, QColor(248, 250, 252))
    palette.setColor(QPalette.Button, QColor(30, 41, 59))
    palette.setColor(QPalette.ButtonText, QColor(248, 250, 252))
    palette.setColor(QPalette.Link, QColor(0, 188, 212))           # #00bcd4
    palette.setColor(QPalette.Highlight, QColor(0, 188, 212))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    stylesheet = """
    QMainWindow {
        background: #0f172a;
    }
    QWidget {
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    QLabel {
        color: #f8fafc;
    }
    QComboBox, QPushButton, QCheckBox, QRadioButton, QSpinBox, QDoubleSpinBox {
        background: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 13px;
        transition: all 0.2s;
    }
    QComboBox:hover, QPushButton:hover, QSpinBox:hover, QDoubleSpinBox:hover {
        border-color: #00bcd4;
        background: rgba(0, 188, 212, 0.1);
    }
    QPushButton:pressed {
        background: #00bcd4;
        color: #000;
    }
    QPushButton:disabled {
        background: #0f172a;
        color: #64748b;
        border-color: #1e293b;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #334155;
    }
    QComboBox QAbstractItemView {
        background: #1e293b;
        color: #f8fafc;
        selection-background-color: #00bcd4;
        selection-color: #000;
        border: 1px solid #334155;
        border-radius: 6px;
    }
    QSlider::groove:horizontal {
        height: 6px;
        background: #334155;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        width: 16px;
        height: 16px;
        margin: -5px 0;
        background: #00bcd4;
        border-radius: 8px;
    }
    QSlider::handle:horizontal:hover {
        background: #67e8f9;
        box-shadow: 0 0 5px #00bcd4;
    }
    QGroupBox {
        color: #94a3b8;
        border: 1px solid #334155;
        border-radius: 8px;
        margin-top: 10px;
        padding-top: 14px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #cbd5e1;
    }
    QStatusBar {
        background: #0f172a;
        color: #94a3b8;
        font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        border-top: 1px solid #1e293b;
    }
    QScrollArea {
        border: none;
        background: transparent;
    }
    QScrollBar:vertical {
        border: none;
        background: #0f172a;
        width: 10px;
        margin: 0px 0px 0px 0px;
    }
    QScrollBar::handle:vertical {
        background: #334155;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical:hover {
        background: #475569;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar:horizontal {
        border: none;
        background: #0f172a;
        height: 10px;
        margin: 0px 0px 0px 0px;
    }
    QScrollBar::handle:horizontal {
        background: #334155;
        min-width: 20px;
        border-radius: 5px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #475569;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    /* Tab Styling */
    QTabWidget::pane {
        border: 1px solid #334155;
        background: #0f172a;
        border-radius: 8px;
    }
    QTabBar::tab {
        background: #1e293b;
        color: #94a3b8;
        padding: 8px 16px;
        margin-right: 4px;
        border: 1px solid #334155;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        font-weight: 600;
    }
    QTabBar::tab:hover {
        background: rgba(0, 188, 212, 0.1);
        color: #f8fafc;
    }
    QTabBar::tab:selected {
        background: #0f172a;
        color: #00bcd4;
        border-bottom: 2px solid #00bcd4;
    }
    """
    app.setStyleSheet(stylesheet)
