import sys
from PyQt5.QtWidgets import QApplication
from app.main_window import MainWindow
from app.style import apply_theme

def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
