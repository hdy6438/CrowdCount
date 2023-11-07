from PyQt5.Qt import QDialog, QDateTime, QDateTimeEdit, QPushButton, QVBoxLayout, QCloseEvent


class DateTimeEdit(QDialog):
    def __init__(self, file_handler, file_time=QDateTime.currentDateTime()):
        super(DateTimeEdit, self).__init__()

        self.file_handler = file_handler

        self.__cancel_btn = None
        self.__confirm_btn = None
        self.__file_time = file_time
        self.__dtEdit = None

        self.setWindowTitle('QDateTimeEdit')
        self.resize(400, 300)
        self.initUi()
        self.can_close = False

    def initUi(self):
        self.__dtEdit = QDateTimeEdit(self.__file_time, self)
        self.__dtEdit.setDisplayFormat('yyyy-MM-dd HH:mm:ss')

        self.__dtEdit.setCalendarPopup(True)

        self.__confirm_btn = QPushButton('确定')
        self.__confirm_btn.clicked.connect(self.confirm)

        self.__cancel_btn = QPushButton('取消')
        self.__cancel_btn.clicked.connect(self.cancel)

        vLayout = QVBoxLayout(self)
        vLayout.setSpacing(10)
        vLayout.addWidget(self.__dtEdit)
        vLayout.addWidget(self.__confirm_btn)
        vLayout.addWidget(self.__cancel_btn)

        self.setLayout(vLayout)

    def confirm(self):
        dateTime = self.__dtEdit.dateTime()

        self.file_handler.set_file_time(dateTime)

        self.can_close = True
        self.close()

        self.file_handler.begin()

    def cancel(self):
        self.file_handler.app_win.set_btn_enabled()
        self.can_close = True
        self.close()

    def closeEvent(self, event: QCloseEvent):
        if not self.can_close:
            event.ignore()
