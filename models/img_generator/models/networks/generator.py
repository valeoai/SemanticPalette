"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.img_generator.models.networks.base_network import BaseNetwork
from models.img_generator.models.networks.normalization import get_nonspade_norm_layer
from models.img_generator.models.networks.architecture import ResnetBlock as ResnetBlock
from models.img_generator.models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.img_generator.models.networks.architecture import DepthsepCCBlock as DepthsepCCBlock
from tools.utils import get_seg_size


class CondConvGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_upsampling_layers = opt.num_upsampling_layers

        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)
        input_nc = get_seg_size(opt.num_semantics, opt.num_things, opt.panoptic, opt.instance_type_for_img)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(input_nc, 16 * nf, 3, padding=1)

        # global-context-aware weight prediction network
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        self.labelenc1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf, 3, padding=1)), nn.LeakyReLU(0.2, True)) # 256
        self.labelenc2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2)), nn.LeakyReLU(0.2, True)) # 128
        self.labelenc3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2)), nn.LeakyReLU(0.2, True)) # 64
        self.labelenc4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2)), nn.LeakyReLU(0.2, True)) # 32
        self.labelenc5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2)), nn.LeakyReLU(0.2, True)) # 16
        self.labelenc6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2)), nn.LeakyReLU(0.2, True)) # 8
        if self.num_upsampling_layers == 'more':
            self.labelenc7 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2)), nn.LeakyReLU(0.2, True)) # 4

        # lateral for fpn
        self.labellat1 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1)), nn.LeakyReLU(0.2, True))#16
        self.labellat2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1)), nn.LeakyReLU(0.2, True))#32
        self.labellat3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1)), nn.LeakyReLU(0.2, True))#64
        self.labellat4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1)), nn.LeakyReLU(0.2, True))#128
        self.labellat5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1)), nn.LeakyReLU(0.2, True))#256
        if self.num_upsampling_layers == 'more':
            self.labellat6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1)), nn.LeakyReLU(0.2, True))

        self.labeldec1 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1)), nn.LeakyReLU(0.2, True))
        self.labeldec2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1)), nn.LeakyReLU(0.2, True))
        self.labeldec3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1)), nn.LeakyReLU(0.2, True))
        self.labeldec4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1)), nn.LeakyReLU(0.2, True))
        self.labeldec5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1)), nn.LeakyReLU(0.2, True))
        if self.num_upsampling_layers == 'more':
            self.labeldec6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1)), nn.LeakyReLU(0.2, True))

        # image generator
        self.head_0 = DepthsepCCBlock(16 * nf, 16 * nf, opt, input_nc + nf)
        self.G_middle_0 = DepthsepCCBlock(16 * nf, 16 * nf, opt, input_nc + nf)
        self.G_middle_1 = DepthsepCCBlock(16 * nf, 16 * nf, opt, input_nc + nf)

        self.up_0 = DepthsepCCBlock(16 * nf, 8 * nf, opt, input_nc + nf)
        self.up_1 = DepthsepCCBlock(8 * nf, 4 * nf, opt, input_nc + nf)
        self.up_2 = DepthsepCCBlock(4 * nf, 2 * nf, opt, input_nc + nf)
        self.up_3 = DepthsepCCBlock(2 * nf, 1 * nf, opt, input_nc + nf)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.width_size // (2 ** num_up_layers)
        sh = round(sw * opt.height_size / opt.width_size)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        # encode segmentation labels
        seg1 = self.labelenc1(seg)  # 256
        seg2 = self.labelenc2(seg1)  # 128
        seg3 = self.labelenc3(seg2)  # 64
        seg4 = self.labelenc4(seg3)  # 32
        seg5 = self.labelenc5(seg4)  # 16
        seg6 = self.labelenc6(seg5)  # 8
        if self.num_upsampling_layers == 'more':
            seg7 = self.labelenc7(seg6)
            segout1 = seg7
            segout2 = self.up(segout1) + self.labellat1(seg6)
            segout2 = self.labeldec1(segout2)
            segout3 = self.up(segout2) + self.labellat2(seg5)
            segout3 = self.labeldec2(segout3)
            segout4 = self.up(segout3) + self.labellat3(seg4)
            segout4 = self.labeldec3(segout4)
            segout5 = self.up(segout4) + self.labellat4(seg3)
            segout5 = self.labeldec4(segout5)
            segout6 = self.up(segout5) + self.labellat5(seg2)
            segout6 = self.labeldec5(segout6)
            segout7 = self.up(segout6) + self.labellat6(seg1)
            segout7 = self.labeldec6(segout7)
        else:
            segout1 = seg6
            segout2 = self.up(segout1) + self.labellat1(seg5)
            segout2 = self.labeldec1(segout2)
            segout3 = self.up(segout2) + self.labellat2(seg4)
            segout3 = self.labeldec2(segout3)
            segout4 = self.up(segout3) + self.labellat3(seg3)
            segout4 = self.labeldec3(segout4)
            segout5 = self.up(segout4) + self.labellat4(seg2)
            segout5 = self.labeldec4(segout5)
            segout6 = self.up(segout5) + self.labellat5(seg1)
            segout6 = self.labeldec5(segout6)

        x = self.head_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout1), dim=1))  # 8

        x = self.up(x)
        x = self.G_middle_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout2), dim=1))  # 16
        if self.num_upsampling_layers == 'more':
            x = self.up(x)
            x = self.G_middle_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout3), dim=1))
        else:
            x = self.G_middle_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout2),
                                             dim=1))  # 16

        x = self.up(x)
        if self.num_upsampling_layers == 'more':
            x = self.up_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout4), dim=1))  # 32
        else:
            x = self.up_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout3), dim=1))  # 32

        x = self.up(x)
        if self.num_upsampling_layers == 'more':
            x = self.up_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout5), dim=1))  # 64
        else:
            x = self.up_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout4), dim=1))  # 64

        x = self.up(x)
        if self.num_upsampling_layers == 'more':
            x = self.up_2(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout6), dim=1))  # 128
        else:
            x = self.up_2(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout5), dim=1))  # 128

        x = self.up(x)
        if self.num_upsampling_layers == 'more':
            x = self.up_3(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout7), dim=1))  # 256
        else:
            x = self.up_3(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout6), dim=1))  # 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class SPADEGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            input_nc = get_seg_size(opt.num_semantics, opt.num_things, opt.panoptic, opt.instance_type_for_img)
            self.fc = nn.Conv2d(input_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.width_size // (2 ** num_up_layers)
        sh = round(sw * opt.height_size / opt.width_size)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim, dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg) #.detach())

        x = self.up(x)
        x = self.G_middle_0(x, seg) #.detach())

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg) #.detach())

        x = self.up(x)
        x = self.up_0(x, seg) #.detach())
        x = self.up(x)
        x = self.up_1(x, seg) #.detach())
        x = self.up(x)
        x = self.up_2(x, seg) #.detach())
        x = self.up(x)
        x = self.up_3(x, seg) #.detach())

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg) #.detach())

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        input_nc = get_seg_size(opt.num_semantics, opt.num_things, opt.panoptic, opt.instance_type_for_img)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, 3, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
