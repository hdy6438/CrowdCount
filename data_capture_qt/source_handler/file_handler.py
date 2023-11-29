import cv2
from PyQt5.Qt import *

from data_capture_qt.ui.date_time import DateTimeEdit
from data_capture_qt.until.setting import freq_sec


class file_handler:
    def __init__(self, app_win, frame_processor):
        self.__file_path = None
        self.__file_time = None
        self.app_win = app_win
        self.__frame_processor = frame_processor

    def set_file_time(self, file_time):
        self.__file_time = file_time

    def select(self):
        dialog = QFileDialog(self.app_win)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("MP4 Files (*.mp4);;AVI Files (*.avi)")
        dialog.setViewMode(QFileDialog.List)

        if dialog.exec():
            filename = dialog.selectedFiles()
            self.__file_path = filename[0]
            print(f"Selected file: {filename}")

            Date_dialog = DateTimeEdit(file_handler=self)
            Date_dialog.exec_()
        else:
            self.app_win.set_btn_enabled()

    def begin(self):
        capture = cv2.VideoCapture(self.__file_path)
        if not capture.isOpened():
            exit()

        freq = capture.get(cv2.CAP_PROP_FPS) * freq_sec

        frame_id = 0
        save_id = 0

        reshape = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2))

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            cv2.imshow("Camera", cv2.resize(frame, reshape))

            cv2.waitKey(1)

            if frame_id % freq == 0:
                self.__frame_processor.save_and_process(self.__file_time, save_id, frame)
                save_id += 1

            frame_id += 1

        capture.release()
        cv2.destroyAllWindows()

        self.app_win.set_btn_enabled()
