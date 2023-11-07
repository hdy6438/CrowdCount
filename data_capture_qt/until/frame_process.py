import time

import cv2
from PyQt5.QtCore import QDateTime


def save_and_process(file_time: QDateTime, fid: int, frame):
    print(file_time)
    print("save")
    img_name = f"H:\\CrowdCount\\datasets\\crowd_count_time_seq_dataset\\imgs\\{time.time()}.jpg"
    cv2.imwrite(img_name, frame)
