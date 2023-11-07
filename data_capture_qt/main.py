import sys

from PyQt5.Qt import QApplication

from data_capture_qt.ui.main_windows import MYWindows

if __name__ == "__main__":
    app = QApplication(sys.argv)
    windows = MYWindows()
    windows.show()
    sys.exit(app.exec_())
