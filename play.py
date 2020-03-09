import sys
import os

import warnings
from profile import *
from model import CSRNet
from utils import save_checkpoint
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import pdb
import numpy as np
import argparse
import json
import cv2
import dataset
import time
from termcolor import cprint
from tensorboardX import SummaryWriter
from CAN import CANNet

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('--train_json', default='part_A_train.json',
                    help='path to train json')
parser.add_argument('--test_json', default='part_A_test.json',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('--gpu', type=str, default='0',
                    help='GPU id to use.')

parser.add_argument('--task', type=str, default='0',
                    help='task id to use.')


class Tripleloss(torch.nn.Module):
    def __init__(self, args):
        super(Tripleloss, self).__init__()
        self.weight1 = args[0]
        self.weight2 = args[1]
        self.weight3 = args[2]

        self.loss_1_fn = nn.MSELoss()
        self.loss_2_fn = nn.SmoothL1Loss()
        self.loss_3_fn = nn.SmoothL1Loss()

        # self.loss_2_fn = nn.MSELoss()

    def forward(self, es_den, gt_den, gt64, out64, gt256, out256):
        n, c, h, w = es_den.shape
        n2, c2, h2, w2 = gt64.shape
        n3, c3, h3, w3 = gt256.shape

        maaloss = self.loss_1_fn(es_den, gt_den) * (w * h / (w + h)) * self.weight1
        perception1 = self.loss_2_fn(out64, gt64) * (w2 * h2 / (w2 + h2)) * self.weight2
        perception2 = self.loss_3_fn(out256, gt256) * (w3 * h3 / (w3 + h3)) * self.weight3
        # perception2 = torch.zeros(1)
        return maaloss, perception1, perception2


class Logger():
    def __init__(self, log_file):
        self.logger = logging.getLogger(name='train')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s]:%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log(self, info, color=None):
        self.logger.info(info)
        if color:
            cprint(info, color=color)


def main():
    global args, best_prec1
    best_prec1 = 1e6

    args = parser.parse_args()
    args.method = 'test'
    args.time = time.strftime("%m-%d_%H-%M", time.localtime())
    args.original_lr = 1e-6
    args.lr = 1e-6
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 500
    args.steps = [-1, 1, 150, 300]
    args.scales = [1, 1, 0.5, 0.5]
    args.workers = 4
    args.seed = 126745
    args.print_freq = 40
    best_epoch = -1
    args.weight = (1, 1, 1)
    args.pre = None  # 'model/original_setting_mse+0.3*smoothL1+smoothL1_lr@e-6_ep_41_mae_68.353.pth'
    LOG = Logger(args.method + 'output.log')
    LOG.log(args)
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.manual_seed(args.seed)

    model = CANNet(per=True)

    model = model.cuda()
    writer = SummaryWriter(comment=args.time)
    # criterion = nn.MSELoss(size_average=False)
    criterion = Tripleloss(args.weight).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            pdb.set_trace()
            LOG.log("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            LOG.log("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.pre, checkpoint['epoch']))
        else:
            LOG.log("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        current_lr = adjust_learning_rate(optimizer, epoch)

        LOG.log('$$$ EPOCH : %d  @ lr:%.5f $$$' % (epoch, current_lr * 100000))

        model = train(train_list, model, criterion, optimizer, epoch, writer=writer, LOG=LOG)
        prec1 = validate(val_list, model, epoch, criterion, writer=writer, LOG=LOG)

        LOG.log('val - epoch:%d - mae:%.3f' % (epoch, prec1), color='yellow')
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        if is_best:
            best_epoch = epoch
        a = time.time()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, best_prec1, args.method)
        dura = time.time() - start
        LOG.log(args.method + ' # ' + args.time + ' # ' + 'current best mae:%.3f @ epoch:%d cost_time:%.3f\n' % (
            best_prec1, best_epoch, dura))
        b = time.time()
        cprint ('outa-outb:%.3f'%(a-b),color='green')

def train(train_list, model, criterion, optimizer, epoch, writer, LOG):
    losses = AverageMeter()
    percep1 = AverageMeter()
    percep2 = AverageMeter()
    maaloss = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            phase='train',
                            seen=model.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers, epoch=epoch),
        batch_size=args.batch_size)

    model.train()
    end = time.time()
    mae = 0.0
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        a = time.time()
        img = img.cuda()
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        output, x64, gt64, x256, gt256 = model(img, target)
        b = time.time()
        maa, p1, p2 = criterion(output, target, x64, gt64, x256, gt256)
        loss = maa + p1 + p2
        c= time.time()
        mae += abs(output.sum().item() - target.sum().item())

        losses.update(loss.item(), img.shape[0])
        maaloss.update(maa.item(), img.shape[0])
        percep1.update(p1.item(), img.shape[0])
        percep2.update(p2.item(), img.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        d = time.time()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % (args.print_freq * 7) == 0:
            LOG.log('TRAIN - Epoch: [{0}][{1}/{2}]  '
                    'Time {batch_time.val:.3f}   '
                    'Data {data_time.val:.3f}  '
                    'Loss {loss.avg:.4f}  '
                    'Maa:{maaloss.avg:.4f}    Percep1:{ploss1.avg:.4f}    Percep2:{ploss2.avg:.4f}\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, maaloss=maaloss, ploss1=percep1, ploss2=percep2))

        cprint ('b-a:%.3f @ c-b:%.3f @d-c:%.3f'%(b-a,c-b,d-c),color='green')
    mae = mae / len(train_loader)
    writer.add_scalar('trainmae', mae, epoch)
    writer.add_scalar('trainloss', losses.avg, epoch)
    LOG.log('train - epoch:%d - mae:%.3f\n' % (epoch, mae), color='yellow')
    return model

def validate(val_list, model, epoch, criterion, writer, LOG):
    LOG.log('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), phase='val'),
        batch_size=args.batch_size)

    model.eval()
    losses = AverageMeter()
    percep1 = AverageMeter()
    percep2 = AverageMeter()
    maaloss = AverageMeter()

    mae = 0

    for i, (img, target) in enumerate(test_loader):

        img = img.cuda()
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        output, x64, gt64, x256, gt256 = model(img, target)
        maa, p1, p2 = criterion(output, target, x64, gt64, x256, gt256)
        loss = maa + p1 + p2

        losses.update(loss.item(), img.shape[0])
        maaloss.update(maa.item(), img.shape[0])
        percep1.update(p1.item(), img.shape[0])
        percep2.update(p2.item(), img.shape[0])

        mae += abs(output.sum().item() - target.sum().item())

        if i % args.print_freq == 0:
            LOG.log('TEST - Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.avg:.4f}  '
                    'Maa:{maaloss.avg:.4f}    Percep1:{ploss1.avg:.4f}    Percep2:{ploss2.avg:.4f}'
                .format(
                epoch, i, len(test_loader), loss=losses, maaloss=maaloss, ploss1=percep1, ploss2=percep2))
    mae = mae / len(test_loader)
    writer.add_scalar('valmae', mae, epoch)
    writer.add_scalar('valloss', losses.avg, epoch)

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    return args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
