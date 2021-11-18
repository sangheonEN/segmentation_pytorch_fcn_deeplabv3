import datetime
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import math
import pytz
import utils
import os
from importlib import import_module
import shutil

class solver(object):
    def __init__(self, train_data_loader, valid_data_loader, opts):
        self.data_loader_train = train_data_loader
        self.data_loader_valid = valid_data_loader

        if opts.model == "deeplabv3":
            model_module = import_module('models.{}.deeplabv3_{}'.format(
                opts.backbone, opts.backbone_layer))
            self.model = model_module.Deeplabv3(n_class=21)
        else:
            model_module = import_module('models.{}.fcn_{}'.format(
                opts.backbone, opts.backbone_layer))
            self.model = model_module.FCN(n_class=21)

        self.model.resume(opts.resume, test=opts.mode in ['inference'])

        if opts.mode == 'trainval':
            optim_module = import_module('models.{}.helpers'.format(
                opts.backbone))
            self.optim = optim_module.prepare_optim(opts, self.model)

        self.model.to(opts.cuda)

    def cross_entropy2d(self, input, target, weight=None):

        """Softmax + Negative Log Likelihood
           input: (n, c, h, w), target: (n, h, w)
           log_p: (n, c, h, w)
           log_p: (n*h*w, c)
           target: (n*h*w,)
        """
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
        return loss


class Trainer(solver):
    def __init__(self, train_data_loader, valid_data_loader, opts):
        super(Trainer, self).__init__(train_data_loader, valid_data_loader, opts)
        self.cuda = opts.cuda
        self.opts = opts
        self.train_loader = train_data_loader
        self.val_loader = valid_data_loader

        if opts.mode in ['inference']:
            return

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('America/Bogota'))

        self.interval_validate = opts.cfg.get('interval_validate',
                                              len(self.train_loader))
        if self.interval_validate is None:
            self.interval_validate = len(self.train_loader)

        self.out = opts.out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = opts.cfg['max_iteration']
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        label_trues, label_preds = [], []
        with torch.no_grad():

            # val data load
            for batch_idx, (data, target) in tqdm.tqdm(
                    enumerate(self.val_loader),
                    total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration,
                    ncols=80,
                    leave=False):
                data, target = data.to(self.cuda), target.to(self.cuda)
                score = self.model(data)

                # val loss function
                loss = self.cross_entropy2d(score, target)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while validating')
                val_loss += float(loss.item()) / len(data)

                # val metrics calculation
                imgs = data.data.cpu()
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu()
                for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                    img, lt = self.val_loader.dataset.untransform(img, lt)
                    lp = np.expand_dims(lp, -1)
                    label_trues.append(lt)
                    label_preds.append(lp)

        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class)

        val_loss /= len(self.val_loader)

        # val metric save
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('America/Bogota')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # val best model save
        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save(
            {
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'best_mean_iu': self.best_mean_iu,
            }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        # data load
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch,
                ncols=80,
                leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            # validation start
            if self.iteration != 0 and self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            # train data augmentation
            data, target = utils.augmentation_train(data, target)
            data, target = data.to(self.cuda), target.to(self.cuda)

            # optimization
            self.optim.zero_grad()
            score = self.model(data)

            # loss function
            loss = self.cross_entropy2d(score, target)
            loss /= len(data)
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            # Segmentation metrics calculation
            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            # Segmentation metrics save
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('America/Bogota')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.item()] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def Train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train',
                                 ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
