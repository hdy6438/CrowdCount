import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from tools.layer import convDU, convLR
from tools.utils import initialize_weights


class Res101_SFCN(nn.Module):
    def __init__(self, mode):
        super(Res101_SFCN, self).__init__()
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.backend = make_layers(self.backend_feat, in_channels=1024)
        self.convDU = convDU(in_out_channels=64, kernel_size=(1, 9))
        self.convLR = convLR(in_out_channels=64, kernel_size=(9, 1))

        self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU())

        initialize_weights(self.modules())

        res = models.resnet101()
        if mode == "train":
            res.load_state_dict(torch.load("/root/Desktop/aaaa/model/pretrain_model/resnet101-cd907fc2.pth"))

        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 23, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        x = self.backend(x)
        x = self.convDU(x)
        x = self.convLR(x)
        x = self.output_layer(x)
        x = F.interpolate(x, scale_factor=8)
        return x


def make_layers(cfg, in_channels=3):
    layers = nn.Sequential()
    for v in cfg:
        layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2))
        layers.append(nn.ReLU(inplace=True))
        in_channels = v
    return nn.Sequential(*layers)


def make_res_layer(block, planes, blocks, stride=1):
    downsample = nn.Sequential()
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample.append(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))

        downsample.append(nn.BatchNorm2d(planes * block.expansion))

    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


if __name__ == "__main__":
    model = Res101_SFCN("predict")
    print(model)
