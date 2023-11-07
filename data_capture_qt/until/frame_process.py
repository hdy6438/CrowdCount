import cv2


def save_and_process(frame_time, frame):
    print("save")
    img_name = f"H:\\CrowdCount\\datasets\\crowd_count_time_seq_dataset\\imgs\\{frame_time}.jpg"
    cv2.imwrite(img_name, frame)
