import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

from datasets.SHHB.SHHB import SHHB
from setting import dataset
from tools import transforms as own_transforms


def loading_data():
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomHorizontallyFlip()
    ])

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*dataset.mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(dataset.scale)
    ])

    train_set = SHHB(dataset.path + '/train_data', main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=dataset.train_batch_size, num_workers=dataset.num_workers, shuffle=True, drop_last=True)

    val_set = SHHB(dataset.path + '/test_data', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=dataset.val_batch_size, num_workers=dataset.num_workers, shuffle=True, drop_last=False)

    return train_loader, val_loader
