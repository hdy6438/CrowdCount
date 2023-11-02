import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model.CC import CrowdCounter
from setting import training, dataset
from tools.utils import AverageMeter


class Trainer:
    def __init__(self, dataloader):
        self.net = CrowdCounter().cuda()
        self.optimizer = Adam(self.net.CCN.parameters(), lr=training.optimizer.lr, weight_decay=training.optimizer.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=training.scheduler.lr_decay_frequency, gamma=training.scheduler.lr_decay_rate)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}

        self.score_list = [{"epoch": i, "train_score": None, "val_score": []} for i in range(training.epoch)]

        self.train_loader, self.val_loader = dataloader()

        self.epoch = 0

    def train(self):
        for epoch in range(training.epoch):
            self.epoch = epoch

            self.train_epoch()

            self.validate_epoch()

            self.check_point()

            if epoch > training.scheduler.lr_decay_start:
                self.scheduler.step()

        self.save_score()

    def train_epoch(self):  # training for all datasets
        self.net.train()

        losses = AverageMeter()

        tq = tqdm(self.train_loader)
        tq.set_description(f"epoch {self.epoch} training")
        for img, gt_map in tq:
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            self.net(img, gt_map)

            loss = self.net.loss
            losses.update(loss.item())

            loss.backward()
            self.optimizer.step()

        loss = losses.avg
        self.score_list[self.epoch]["train_score"] = loss

        print(f"train loss : {loss}")

    def validate_epoch(self):
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        tq = tqdm(self.val_loader)
        tq.set_description(f"epoch {self.epoch} validating")
        for data in tq:
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                pred_map = self.net.forward(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / dataset.scale
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) ** 2)

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.score_list[self.epoch]["val_score"] = [mae, mse, loss]

        print('mae %.2f mse %.2f val loss %.4f' % (mae, mse, loss))

    def check_point(self):

        mae, mse, loss = self.score_list[self.epoch]["val_score"]

        model_state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'score': self.score_list[self.epoch]
        }

        if mae < self.train_record['best_mae'] or mse < self.train_record['best_mse']:
            snapshot_name = 'all_ep_%d_mae_%.1f_mse_%.1f' % (self.epoch + 1, mae, mse)
            print(f"model improve : mae from {self.train_record['best_mae']} to {mae} and mse from {self.train_record['best_mse']} to {mse} ,saving to {snapshot_name}")

            self.train_record['best_model_name'] = snapshot_name
            self.train_record['best_mae'] = min(mae, self.train_record['best_mae'])
            self.train_record['best_mse'] = min(mse, self.train_record['best_mse'])

            torch.save(model_state, f'/root/Desktop/aaaa/model/res/{snapshot_name}.pth')
        else:
            print('model not improve : best_model: %s , mae %.2f, mse %.2f' % (self.train_record['best_model_name'], self.train_record['best_mae'], self.train_record['best_mse']))
            torch.save(model_state, '/root/Desktop/aaaa/model/res/latest_state.pth')

    def save_score(self):
        np.save("/root/Desktop/aaaa/model/res/score", self.score_list)
        with open("/root/Desktop/aaaa/model/res/best_score.txt", mode="w") as f:
            for key, value in self.train_record.values():
                f.write(f"{key}:{value}\n")
            f.close()
