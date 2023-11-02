import numpy as np
import torch

from model.CC import CrowdCounter
from setting import predict
from tools.utils import predict_img_loader


class predictor:
    def __init__(self):
        self.__net = CrowdCounter(mode="predict")
        self.__net.load_state_dict(torch.load(predict.model_path)["net"])
        self.__net.cuda()
        self.__net.eval()
        self.__img_loader = predict_img_loader()

    def predict_img(self, img):
        """""
        img 为图片路径或pil图片
        """""
        img = self.__img_loader.load(img)
        with torch.no_grad():
            pred_map = self.__net.predict(img)
            pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]
            pred = np.sum(pred_map) / 100.0
            return pred, pred_map
