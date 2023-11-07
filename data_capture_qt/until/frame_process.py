from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QDateTime

from data_capture_qt.until.setting import freq_sec
from data_capture_qt.until.setting import num_worker, base_path
from predict import predictor


class frame_process:
    def __init__(self):
        self.__pool = ThreadPoolExecutor(max_workers=num_worker)
        self.__model = predictor()

    def save_and_process(self, file_time: QDateTime, save_id: int, frame):
        def do():
            frame_time = file_time.addSecs(save_id * freq_sec)
            print(frame_time, "processing")
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            num, pmap = self.__model.predict_img(img_pil)
            np.savez(f"{base_path}\\{frame_time.toSecsSinceEpoch()}", density=num, density_map=pmap, image=frame)
            print(frame_time, "done")

        self.__pool.submit(do)
