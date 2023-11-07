import threading

import cv2
from PyQt5.QtCore import QDateTime

from data_capture_qt.until.frame_process import save_and_process
from data_capture_qt.until.setting import freq_sec


class cap_handler:
    def __init__(self, app_win):
        self.__capture = cv2.VideoCapture(0)
        if not self.__capture.isOpened():
            exit()

        self.__freq = self.__capture.get(cv2.CAP_PROP_FPS) * freq_sec
        self.__app_win = app_win
        self.__file_time = None

    def begin(self):
        self.__file_time = QDateTime.currentDateTime()

        fid = 0
        while True:
            ret, frame = self.__capture.read()
            if not ret:
                break
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) == ord('q'):
                break

            if fid % self.__freq == 0:
                threading.Thread(target=save_and_process, args=(self.__file_time, fid, frame)).start()
                fid += 1

        self.__capture.release()
        cv2.destroyAllWindows()

        self.__app_win.set_btn_enabled()
