import cv2


def save_and_process(frame_time, frame):
    img_name = f"..\\datasets\\crowd_count_time_seq_dataset\\imgs\\{frame_time}.jpg"
    cv2.imwrite(img_name, frame)
