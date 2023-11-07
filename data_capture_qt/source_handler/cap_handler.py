import threading
import time

import cv2

from data_capture_qt.until.setting import freq_sec
from data_capture_qt.until.frame_process import save_and_process


class cap_handler:
    def __init__(self):
        self.__capture = cv2.VideoCapture(0)
        if not self.__capture.isOpened():
            exit()
        self.freq = self.__capture.get(cv2.CAP_PROP_FPS) * freq_sec

    def begin(self, app_win):

        fid = 0
        while True:
            ret, frame = self.__capture.read()
            if not ret:
                break
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) == ord('q'):
                break

            if fid % self.freq == 0:
                threading.Thread(target=save_and_process, args=(int(time.time()), frame)).start()

            fid += 1

        self.__capture.release()
        cv2.destroyAllWindows()

        app_win.set_btn_enabled()
        app_win.can_close = True