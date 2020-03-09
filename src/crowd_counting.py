from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.nn import DataParallel
from torch.autograd import Variable
from src.network import Conv2d, FC, Conv2d_dilated, np_to_variable
from torchvision import models
import src.network as network
import numpy as np
import pdb
import importlib


class Perception(nn.Module):
    def __init__(self, p64=True, p256=True, weight=[1, 1]):
        super(Perception, self).__init__()
        self.p64 = p64
        self.p256 = p256
        mod = models.vgg16(pretrained=True)
        self.weight = weight
        for p in mod.parameters():
            p.requires_grad = False
        features = list(mod.features.children())

        if self.p256:
            self.perception256 = nn.Sequential(*features[0:15])
            self.loss256 = nn.SmoothL1Loss()
        if self.p64:
            self.perception64 = nn.Sequential(*features[0:3])
            self.loss64 = nn.SmoothL1Loss()

    def forward(self, x, gt):
        if x.shape[1] == 1:
            x3 = torch.cat([x, x, x], 1)
            gt3 = torch.cat([gt, gt, gt], 1)
        else:
            x3 = x
            gt3 = gt
        x64 = self.perception64(x3) if self.perception64 else None
        x256 = self.perception256(x3) if self.perception256 else None

        gt64 = self.perception64(gt3) if self.perception64 else None
        gt256 = self.perception256(gt3) if self.perception256 else None
        self.LOSS64, self.LOSS256 = self.buildloss(x64, gt64, x256, gt256)
        return x64, gt64, x256, gt256

    def buildloss(self, x64, gt64, x256, gt256):
        LOSS64 = None
        LOSS256 = None
        if self.p64:
            n, c, h, w = x64.shape
            LOSS64 = self.loss64(x64, gt64)
            LOSS64 = LOSS64* h * w / (h + w) * self.weight[0]
        if self.p256:
            n, c, h, w = x256.shape
            LOSS256 = self.loss256(x256, gt256)
            LOSS256 = LOSS256* h * w / (h + w) * self.weight[1]
        return LOSS64, LOSS256

    @property
    def loss(self):
        return self.LOSS64, self.LOSS256


class CrowdCounter(nn.Module):

    def __init__(self, optimizer, opt):
        super(CrowdCounter, self).__init__()
        self.opt = opt
        self.device = 'cuda'
        self.model = self.find_model_using_name(opt.model_name)()
        self.loss_fn_ = self.find_loss_using_name(opt.loss or 'MSE')()
        self.init_model(opt.pretrain)
        self.per = opt.per

        if self.per:
            self.perception = Perception(weight=[1, 1]).cuda()
            self.perloss = None

        if optimizer is not None:
            self.optimizer = optimizer(self)
            self.optimizer.zero_grad()

    @property
    def loss(self):
        if not self.per:
            return self.loss_
        else:
            return self.loss_, self.perloss[0], self.perloss[1]

    def init_model(self, model_path=None):
        if model_path is not None:
            network.load_net(model_path, self.model)
        else:
            network.weights_normal_init(self.model, dev=1e-6)
            # network.load_net('../../pruned_VGG.h5', self.model.front_end, skip=True)
            # network.load_net("../../vgg16.h5", self.model.front_end, skip=True)

        def calpara(model):
            print('---------- Networks initialized -------------')
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
            print('-----------------------------------------------')

        calpara(self.model)

        network.weights_normal_init(self.loss_fn_, dev=0.01)

        if len(self.opt.gpus) > 0:
            assert (torch.cuda.is_available())
            self.model.to(self.device)
            #self.model = torch.nn.DataParallel(self.model, self.opt.gpus)  # multi-GPUs
            if self.opt.loss is not None and 'SSIM' in self.opt.loss:
                self.loss_fn_.to(self.device)
                #self.loss_fn = torch.nn.DataParallel(self.loss_fn_, self.opt.gpus)  # multi-GPUs
            else:
                self.loss_fn = self.loss_fn_

    def forward(self, img_data, gt_data=None, hooker=None, **kargs):

        if self.training:
            img_data = img_data.to(self.device)
            with torch.no_grad():
                noise = img_data.data.new(img_data.shape).uniform_(-0.03, 0.03)
                img_data = img_data + noise
            gt_data = gt_data.to(self.device)
        else:
            img_data = img_data.to(self.device)

        if self.training:
            pdb.set_trace()
            density_map = self.model(img_data, **kargs)
            if self.per:
                self.perout = self.perception(density_map, gt_data)
                self.perloss = self.perception.loss
        else:
            with torch.no_grad():
                density_map = self.model(img_data, **kargs)

        if self.training:
            self.loss_ = self.loss_fn_(density_map, gt_data)
            
            if len(self.opt.gpus) > 1:
                self.loss_ = self.loss_.mean()

        if hooker is not None:
            return density_map, hooker(self.model.visual)
        else:
            return density_map

    def backward(self, scale=1.0):
        if self.per:
            self.LOSS = self.loss_ + self.perloss[0] + self.perloss[1]
            self.LOSS = self.LOSS*scale
        else:
            self.LOSS = self.loss_ * scale
        self.LOSS.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def find_loss_using_name(self, loss_name):
        # Given the option --model [modelname],
        # the file "models/modelname_model.py"
        # will be imported.
        ssimlib = importlib.import_module('src.ssim')

        if loss_name == 'MSE':
            return nn.MSELoss
        loss_fn = None
        for name, cls in ssimlib.__dict__.items():
            if name.lower() == loss_name.lower():
                print('using loss_fn {}'.format(name))
                loss_fn = cls

        if loss_fn is None:
            print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
                model_filename, target_model_name))
            exit(0)

        return loss_fn

    def find_model_using_name(self, model_name):
        # Given the option --model [modelname],
        # the file "models/modelname_model.py"
        # will be imported.
        model_filename = "models." + model_name
        modellib = importlib.import_module(model_filename)

        # In the file, the class called ModelNameModel() will
        # be instantiated. It has to be a subclass of BaseModel,
        # and it is case-insensitive.
        model = None
        for name, cls in modellib.__dict__.items():
            if name.lower() == model_name.lower():
                model = cls

        if model is None:
            print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
                model_filename, target_model_name))
            exit(0)

        return model
