#!/usr/bin/python
# -*- encoding: utf-8 -*-

import time
import logging
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from backbone import Embeddor
from loss import BottleneckLoss
from market1501 import Market1501
from balanced_sampler import BalancedSampler


## logging
if not os.path.exists('./res/'): os.makedirs('./res/')
logfile = 'sft_reid-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
logfile = os.path.join('res', logfile)
FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
logging.root.addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)



def lr_scheduler(epoch, optimizer):
    ## TODO: warmup epoch or iter ?
    warmup_epoch = 20
    warmup_lr = 1e-3
    lr_steps = [80, 100]
    start_lr = 1e-1
    lr_factor = 0.1

    if epoch <= warmup_epoch:  # lr warmup
        warmup_scale = (start_lr / warmup_lr) ** (1.0 / warmup_epoch)
        lr = warmup_lr * (warmup_scale ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.defaults['lr'] = lr
    else:  # lr jump
        for i, el in enumerate(lr_steps):
            if epoch == el:
                lr = start_lr * (lr_factor ** (i + 1))
                logger.info('====> LR is set to: {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.defaults['lr'] = lr
    lrs = [round(el['lr'], 6) for el in optimizer.param_groups]
    return optimizer, lrs


def train():
    ## data
    P, K = 16, 8
    batchsize = P * K
    logger.info('creating dataloader')
    dataset = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train',
            is_train = True)
    num_classes = dataset.get_num_classes()
    sampler = BalancedSampler(dataset, P, K)
    dl = DataLoader(dataset,
            batch_sampler = sampler,
            num_workers = 8)

    ## network and loss
    logger.info('setup model and loss')
    bottleneck_loss = BottleneckLoss(2048, num_classes)
    bottleneck_loss.cuda()
    bottleneck_loss.train()
    net = Embeddor()
    net.cuda()
    net.train()
    #  net = nn.DataParallel(net)

    ## optimizer
    logger.info('creating optimizer')
    lr = 0.1
    momentum = 0.9
    params = list(net.parameters())
    params += list(bottleneck_loss.parameters())
    optim = torch.optim.SGD(params, lr=lr, momentum=momentum)

    ## training
    logger.info('start training')
    n_epochs = 140
    t_start = time.time()
    loss_it = []
    for ep in range(n_epochs):
        optim, lrs = lr_scheduler(ep, optim)
        for it, (imgs, lbs, _) in enumerate(dl):
            imgs = imgs.cuda()
            lbs = lbs.cuda()

            optim.zero_grad()
            embs_org, embs_sft = net(imgs)
            loss_org = bottleneck_loss(embs_org, lbs)
            loss_sft = bottleneck_loss(embs_sft, lbs)
            loss = loss_org + loss_sft
            loss.backward()
            optim.step()

            loss = loss.cpu().item()
            loss_it.append(loss)
            if it % 10 == 0 and not it == 0:
                t_end = time.time()
                t_interval = t_end - t_start
                log_loss = sum(loss_it) / len(loss_it)
                msg = ', '.join([
                    'epoch: {}',
                    'iter: {}',
                    'loss: {:.4f}',
                    'lr: {}',
                    'time: {:.4f}'
                    ]).format(ep,it, log_loss, lrs, t_interval)
                logger.info(msg)
                loss_it = []
                t_start = t_end

    ## save model
    if hasattr(net, 'module'):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    torch.save(state_dict, './res/model_final.pth')
    logger.info('\nTraining done, model saved to {}\n\n'.format('./res/model_final.pth'))


if __name__ == '__main__':
    train()
