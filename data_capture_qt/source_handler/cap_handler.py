import cv2
from PyQt5.QtCore import QDateTime

from data_capture_qt.until.setting import freq_sec


class cap_handler:
    def __init__(self, app_win, frame_processor):
        self.__capture = cv2.VideoCapture(0)
        if not self.__capture.isOpened():
            exit()

        self.__app_win = app_win
        self.__file_time = None
        self.__frame_processor = frame_processor

    def begin(self):
        file_time = QDateTime.currentDateTime()

        freq = self.__capture.get(cv2.CAP_PROP_FPS) * freq_sec

        frame_id = 0
        save_id = 0

        reshape = (int(self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2), int(self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2))

        while True:
            ret, frame = self.__capture.read()
            if not ret:
                break

            cv2.imshow("Camera", cv2.resize(frame, reshape))

            if cv2.waitKey(1) == ord('q'):
                break

            if frame_id % freq == 0:
                self.__frame_processor.save_and_process(file_time, save_id, frame)
                save_id += 1

            frame_id += 1

        self.__capture.release()
        cv2.destroyAllWindows()

        self.__app_win.set_btn_enabled()
