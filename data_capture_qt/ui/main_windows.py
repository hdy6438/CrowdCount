from PyQt5.Qt import QMainWindow, QCloseEvent

from data_capture_qt.source_handler.cap_handler import cap_handler
from data_capture_qt.source_handler.file_handler import file_handler
from data_capture_qt.ui.ui import Ui_Form


class MYWindows(QMainWindow):
    def __init__(self):
        super(MYWindows, self).__init__()
        self.file = None
        self.capture = None
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.cap_btn = self.ui.from_cap
        self.cap_btn.clicked.connect(self.use_cap)

        self.file_btn = self.ui.from_file
        self.file_btn.clicked.connect(self.use_file)

        self.can_close = True

    def closeEvent(self, event: QCloseEvent):
        if not self.can_close:
            event.ignore()

    def set_btn_disabled(self):
        self.can_close = False
        self.file_btn.setEnabled(False)
        self.cap_btn.setEnabled(False)

    def set_btn_enabled(self):
        self.file_btn.setEnabled(True)
        self.cap_btn.setEnabled(True)
        self.can_close = True

    def use_cap(self):
        self.set_btn_disabled()
        self.capture = cap_handler(self)
        self.capture.begin()

    def use_file(self):
        self.set_btn_disabled()
        self.file = file_handler(self)
        self.file.select()
