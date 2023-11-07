import threading
import time

import cv2
from PyQt5.Qt import *

from data_capture_qt.until.setting import freq_sec
from data_capture_qt.ui.date_time import  DemoDateTimeEdit
from data_capture_qt.until.frame_process import save_and_process




class file_handler:
    def __init__(self, app_win):
        self.__file_path = None
        self.__file_time = None
        self.__app_win = app_win

    def set_file_time(self, file_time):
        self.__file_time = file_time

    def select(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("MP4 Files (*.mp4);;AVI Files (*.avi)")
        dialog.setViewMode(QFileDialog.List)

        if dialog.exec():
            filename = dialog.selectedFiles()
            self.__file_path = filename[0]
            print(f"Selected file: {filename}")

            Date_dialog = DemoDateTimeEdit(file_handler=self)
            Date_dialog.exec_()


    def get_frame_time(self,idx):
        print(idx,self.__file_time)
        return int(time.time())

    def begin(self):
        capture = cv2.VideoCapture(self.__file_path)
        if not capture.isOpened():
            exit()
        freq = capture.get(cv2.CAP_PROP_FPS) * freq_sec

        fid = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            cv2.imshow("Camera", cv2.resize(frame, (640, 360)))

            cv2.waitKey(1)

            if fid % freq == 0:
                threading.Thread(target=save_and_process, args=(self.get_frame_time(idx=fid), frame)).start()

            fid += 1

        capture.release()
        cv2.destroyAllWindows()
        self.set_btn_enable()

    def set_btn_enable(self):
        self.__app_win.set_btn_enabled()
        self.__app_win.can_close = True