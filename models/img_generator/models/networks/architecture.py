"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.img_generator.models.networks.normalization import SPADE
from models.img_generator.models.networks.condconv import DepthConv
from tools.utils import get_seg_size


class DepthsepCCBlock(nn.Module):
    def __init__(self, fin, fout, opt, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # layers to generate conditional convolution weights
        nhidden = 128
        self.weight_channels = fmiddle * 9
        self.gen_weights1 = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fin * 9, kernel_size=3, padding=1))
        self.gen_weights2 = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fout * 9, kernel_size=3, padding=1))

        self.gen_se_weights1 = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fmiddle, kernel_size=3, padding=1),
            nn.Sigmoid())
        self.gen_se_weights2 = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fout, kernel_size=3, padding=1),
            nn.Sigmoid())

        # create conv layers
        if int(os.environ['WORLD_SIZE']) > 1: # distributed training
            BNFunc = nn.SyncBatchNorm
        else:
            BNFunc = nn.BatchNorm2d
        self.conv_0 = DepthConv(fin, opt)
        self.norm_0 = BNFunc(fmiddle, affine=True)
        self.conv_1 = nn.Conv2d(fin, fmiddle, kernel_size=1)
        self.norm_1 = BNFunc(fin, affine=True)
        self.conv_2 = DepthConv(fmiddle, opt)
        self.norm_2 = BNFunc(fmiddle, affine=True)
        self.conv_3 = nn.Conv2d(fmiddle, fout, kernel_size=1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))
            spade_config_str = opt.norm_G.replace('spectral', '')
            spade_config_str = "spade" + spade_config_str + "3x3"
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    def forward(self, x, seg):

        # predict weight for conditional convolution
        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        conv_weights1 = self.gen_weights1(segmap)
        conv_weights2 = self.gen_weights2(segmap)
        se_weights1 = self.gen_se_weights1(segmap)
        se_weights2 = self.gen_se_weights2(segmap)

        x_s = self.shortcut(x, segmap)

        dx = self.norm_1(x)
        dx = self.conv_0(dx, conv_weights1)
        dx = self.conv_1(dx)
        dx = torch.mul(dx, se_weights1)
        dx = self.actvn(dx)
        dx = self.norm_2(dx)
        dx = self.conv_2(dx, conv_weights2)
        dx = self.conv_3(dx)
        dx = torch.mul(dx, se_weights2)
        dx = self.actvn(dx)

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        input_nc = get_seg_size(opt.num_semantics, opt.num_things, opt.panoptic, opt.instance_type_for_img)
        self.norm_0 = SPADE(spade_config_str, fin, input_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, input_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, input_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
