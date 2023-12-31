import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import setting
from predict import predictor

if __name__ == "__main__":

    video_path = r"d:\Users\23120\Desktop\WeChat_20231201233813.mp4"
    freq_sec = 0.5

    predictor = predictor()

    device = 'gpu' if setting.gpu else 'cpu'

    capture = cv2.VideoCapture(video_path)

    freq = int(capture.get(cv2.CAP_PROP_FPS) * freq_sec)

    frame_id = 0

    reshape = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)//2), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)//2))
    s = time.time()

    # 参数：文件名、编解码器、帧速率（可选）、分辨率（可选）
    video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), capture.get(cv2.CAP_PROP_FPS), (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)))

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        cv2.waitKey(1)

        print(frame_id, "processing")
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        num, pmap = predictor.predict_img(img_pil)

        heatmapshow = None
        heatmapshow = cv2.normalize(pmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

        predict_output = np.concatenate([frame, heatmapshow], axis=1)

        org = (50, 60)  # 文字开始的坐标，例如从(50,50)位置开始
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        fontScale = 2  # 字体的尺度，可以根据需要调整
        color = (255, 0, 0)  # 字体颜色，BGR格式，例如蓝色为(255,0,0)
        thickness = 3  # 字体的粗细

        # 在图像上添加文字
        fps = frame_id / (time.time() - s)

        predict_output = cv2.putText(predict_output, f"device: {device} num: {round(num,3)} fps: {round(fps,3)}", org, font, fontScale, color, thickness, cv2.LINE_AA)

        predict_output = cv2.resize(predict_output, (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)))

        cv2.imshow("predict_output", predict_output)
        video_writer.write(predict_output)
        cv2.imwrite(f"predict_output/predict_{frame_id}.png", predict_output)

        print(frame_id, "done")



        frame_id += 1

    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

