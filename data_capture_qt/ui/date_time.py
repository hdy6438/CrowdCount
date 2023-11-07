from PyQt5.Qt import QDialog,QDateTime,QDateTimeEdit,QPushButton,QVBoxLayout,QCloseEvent




class DemoDateTimeEdit(QDialog):
    def __init__(self, file_handler, file_time=QDateTime.currentDateTime()):
        super(DemoDateTimeEdit, self).__init__()

        self.file_handler = file_handler
        self.cancel_btn = None
        self.confirm_btn = None
        self.file_time = file_time
        self.dtEdit = None
        self.setWindowTitle('QDateTimeEdit')

        self.resize(400, 300)

        self.initUi()

        self.can_close = False

    def initUi(self):
        self.dtEdit = QDateTimeEdit(self.file_time, self)
        self.dtEdit.setDisplayFormat('yyyy-MM-dd HH:mm:ss')

        self.dtEdit.setCalendarPopup(True)

        self.confirm_btn = QPushButton('确定')
        self.confirm_btn.clicked.connect(self.confirm)

        self.cancel_btn = QPushButton('取消')
        self.cancel_btn.clicked.connect(self.cancel)

        vLayout = QVBoxLayout(self)
        vLayout.setSpacing(10)
        vLayout.addWidget(self.dtEdit)
        vLayout.addWidget(self.confirm_btn)
        vLayout.addWidget(self.cancel_btn)


        self.setLayout(vLayout)

    def confirm(self):
        dateTime = self.dtEdit.dateTime()
        print(dateTime)

        self.file_handler.set_file_time(dateTime)

        self.can_close = True
        self.close()

        self.file_handler.begin()

    def cancel(self):
        self.file_handler.set_btn_enable()
        self.can_close = True
        self.close()

    def closeEvent(self, event: QCloseEvent):
        if not self.can_close:
            event.ignore()
