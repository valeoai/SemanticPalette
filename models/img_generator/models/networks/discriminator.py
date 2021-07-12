"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.img_generator.models.networks.base_network import BaseNetwork
from models.img_generator.models.networks.normalization import get_nonspade_norm_layer
from tools.utils import get_seg_size


# Feature-Pyramid Semantics Embedding Discriminator
class FPSEDiscriminator(BaseNetwork):
    def __init__(self, opt, conditional=True):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = get_seg_size(opt.num_semantics, opt.num_things, opt.panoptic, opt.instance_type_for_img)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        self.conditional = conditional

        # bottom-up pathway
        self.enc1 = nn.Sequential(
            norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))

        # top-down pathway
        self.lat2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.final3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.final4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))

        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf * 2, 1, kernel_size=1)
        if self.conditional:
            self.seg = nn.Conv2d(nf * 2, nf * 2, kernel_size=1)
            self.embedding = nn.Conv2d(label_nc, nf * 2, kernel_size=1)

    def forward(self, fake_and_real_img, segmap=None, sem_alignment_only=False):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)
        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)
        # Patch-based True/False prediction
        if not sem_alignment_only:
            pred2 = self.tf(feat32)
            pred3 = self.tf(feat33)
            pred4 = self.tf(feat34)
        else:
            pred2, pred3, pred4 = None, None, None
        # intermediate features for discriminator feature matching loss
        feats = [feat12, feat13, feat14, feat15]

        if segmap is not None and self.conditional:
            # Patch-based True/False prediction
            seg2 = self.seg(feat32)
            seg3 = self.seg(feat33)
            seg4 = self.seg(feat34)
            # segmentation map embedding
            segemb = self.embedding(segmap)
            segemb = F.avg_pool2d(segemb, kernel_size=2, stride=2)
            segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)
            segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)
            segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)
            # semantics embedding discriminator score
            sem_pred2 = torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
            sem_pred3 = torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
            sem_pred4 = torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)
            pred2 = pred2 + sem_pred2 if pred2 is not None else sem_pred2
            pred3 = pred3 + sem_pred3 if pred3 is not None else sem_pred3
            pred4 = pred4 + sem_pred4 if pred4 is not None else sem_pred4

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]

        return [feats, results]


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt, conditional=True):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, conditional)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, conditional=True):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, conditional)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt, conditional=True):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt, conditional)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=2, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt, conditional=True):
        if conditional:
            input_nc = get_seg_size(opt.num_semantics, opt.num_things, opt.panoptic, opt.instance_type_for_img)
            input_nc += 3
        else:
            input_nc = 3
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
