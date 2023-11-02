import torch.nn as nn

from .Res101_SFCN import Res101_SFCN


class CrowdCounter(nn.Module):
    def __init__(self, mode="train"):
        super(CrowdCounter, self).__init__()
        self.loss_mse = None
        self.CCN = Res101_SFCN(mode).cuda()
        self.loss_mse_fn = nn.MSELoss()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        return loss_mse

    def predict(self, img):
        density_map = self.CCN(img)
        return density_map
