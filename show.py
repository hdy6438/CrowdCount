import time

import numpy as np
from PIL import Image

import setting
from predict import predictor
import cv2

predictor = predictor()

device = 'gpu' if setting.gpu else 'cpu'

cap = cv2.VideoCapture(0)

s = time.time()

idx = 0

while cap.isOpened():
    success,frame = cap.read()
    if not success:
        break

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    num, pmap = predictor.predict_img(img_pil)

    heatmapshow = None
    heatmapshow = cv2.normalize(pmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    predict_output = np.concatenate([frame, heatmapshow], axis=1)

    org = (25, 25)  # 文字开始的坐标，例如从(50,50)位置开始
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    fontScale = 0.5  # 字体的尺度，可以根据需要调整
    color = (255, 0, 0)  # 字体颜色，BGR格式，例如蓝色为(255,0,0)
    thickness = 1  # 字体的粗细

    # 在图像上添加文字
    fps = idx/(time.time() - s)

    predict_output = cv2.putText(predict_output, f"device: {device} num: {num} fps: {fps}", org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("predict_output", predict_output)

    cv2.waitKey(1)

    idx += 1

cap.release()
cv2.destroyAllWindows()
