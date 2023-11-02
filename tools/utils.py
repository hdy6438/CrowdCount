import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
from PIL import Image
from torch import nn
from torch.autograd import Variable

from setting import dataset


def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print(m)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class predict_img_loader:
    def __init__(self):
        self.__img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*dataset.mean_std)
        ])

    def load(self, path=None, pil_img=None):
        if path is not None:
            img = Image.open(path)  # 打开图片
        else:
            img = pil_img

        if img.mode == 'L':
            img = img.convert('RGB')
        img = self.__img_transform(img)
        return Variable(img[None, :, :, :]).cuda()


def draw_map(matrix):
    plt.imshow(matrix, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
