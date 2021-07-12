import numpy as np
import torch.nn as nn
import torch

import models.seg_generator.modules as modules
from tools.utils import get_seg_size


class ProGANGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.cond_dim = opt.cond_dim
        self.latent_dim = opt.latent_dim
        self.max_scale = int(np.log2(opt.max_dim)) - 1
        self.scale = 1
        self.cond_seg = opt.cond_seg

        blocks = []
        for i in range(self.max_scale):
            in_dim, out_dim = self._feature_dims_for_block(i, max_hidden_dim=opt.max_hidden_dim)
            blocks.append(ProGANGeneratorBlock(in_dim, out_dim, initial=i == 0, final=i == self.max_scale - 1, opt=opt))
        self.blocks = nn.ModuleList(blocks)

    @property
    def max_dim(self):
        return 2 ** (self.max_scale + 1)

    @property
    def res(self):
        return 2 ** (self.scale + 1)

    @res.setter
    def res(self, val):
        if (val & (val - 1)) != 0 or val == 0:
            raise ValueError("res must be power of two")
        self.scale = int(np.log2(val)) - 1

    def _feature_dims_for_block(self, i, max_hidden_dim=512):
        if i == 0:
            in_dim = self.cond_dim + self.latent_dim
        else:
            in_dim = min(max_hidden_dim, 2 ** (13 - i))
        out_dim = min(max_hidden_dim, 2 ** (13 - i - 1))
        return in_dim, out_dim

    def get_input_vector(self, z, cond):
        if self.cond_seg == "semantic" or self.cond_seg == "panoptic":
            z = torch.cat((z, cond["sem_cond"]), dim=1)
        if self.cond_seg == "instance" or self.cond_seg == "panoptic":
            z = torch.cat((z, cond["ins_cond"]), dim=1)
        return z

    def forward(self, z, cond=None):
        segs = []
        x = self.get_input_vector(z, cond)
        for i in range(self.scale - 1):
            x, inter_segs = self.blocks[i](x, cond, to_joint=True)
            segs += inter_segs
        x = self.blocks[self.scale - 1](x, cond, to_joint=False)
        final_seg = self.blocks[self.scale - 1].mask(x, cond)
        segs.append(final_seg)
        return segs

    def interpolate(self, z, alpha, cond=None):
        segs = []
        x = self.get_input_vector(z, cond)

        for i in range(self.scale - 2):
            x, inter_segs = self.blocks[i](x, cond, to_joint=True)
            segs += inter_segs
        x = self.blocks[self.scale - 2](x, cond, to_joint=False)
        out_coarse = self.blocks[self.scale - 2].mask(x, cond, double=True)

        x, inter_segs = self.blocks[self.scale - 2].cond_joint(x, cond)
        segs += inter_segs
        x = self.blocks[self.scale - 1](x, cond, to_joint=False)
        out_fine = self.blocks[self.scale - 1].mask(x, cond)

        final_seg = {key: (1 - alpha) * out_coarse[key] + alpha * out_fine[key] for key in out_coarse}
        segs.append(final_seg)
        return segs


class ProGANGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initial=False, final=False, opt=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial = initial
        self.final = final

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = modules.PixelNorm()
        self.upsample = modules.NearestInterpolate(scale_factor=2)
        if initial:
            self.h0 = 4
            self.w0 = int(opt.aspect_ratio * self.h0)
            self.fc1 = modules.EqualizedLinear(in_channels, out_channels * self.h0 * self.w0, gain=np.sqrt(2) / 4)
        else:
            self.conv1 = modules.EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = modules.EqualizedConv2d(out_channels, out_channels, 3, padding=1)
        joints_mul = 0 if final else opt.joints_mul # no assisting joint for last layer
        self.cond_joint = modules.CondJoint(out_channels, opt, joints_mul=joints_mul)

    def forward(self, x, cond=None, to_joint=False):
        if self.initial:
            x = self.norm(x)
            x = self.fc1(x)
            x = x.view(x.size(0), -1, self.h0, self.w0)
        else:
            x = self.upsample(x)
            x = self.conv1(x)
        x = self.norm(self.lrelu(x))
        x = self.norm(self.lrelu(self.conv2(x)))
        if to_joint:
            return self.cond_joint(x, cond)
        else:
            return x

    def mask(self, x, cond=None, double=False):
        if double:
            x = self.upsample(x)
        return self.cond_joint.mask(x, cond)


class ProGANDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.max_scale = int(np.log2(opt.max_dim)) - 1
        self.scale = 1
        self.panoptic = opt.panoptic
        self.original = "original_cgan" in opt.cond_mode

        if self.original:
            self.fc0 = modules.EqualizedLinear(opt.num_semantics, 8)

        blocks = []
        for i in range(self.max_scale):
            block_i = self.max_scale - 1 - i
            in_dim, out_dim = self._feature_dims_for_block(block_i, max_hidden_dim=opt.max_hidden_dim)
            blocks.append(ProGANDiscriminatorBlock(in_dim, out_dim, final=i == self.max_scale - 1, opt=opt))
        self.blocks = nn.ModuleList(blocks)

    @property
    def max_dim(self):
        return 2 ** (self.max_scale + 1)

    @property
    def res(self):
        return 2 ** (self.scale + 1)

    @res.setter
    def res(self, val):
        if (val & (val - 1)) != 0 or val == 0:
            raise ValueError("dim must be power of two")
        self.scale = int(np.log2(val)) - 1

    def _feature_dims_for_block(self, i, max_hidden_dim=512):
        in_dim = min(max_hidden_dim, 2 ** (13 - i - 1))
        out_dim = min(max_hidden_dim, 2 ** (13 - i))
        return in_dim, out_dim

    def forward(self, x, sem_cond=None):
        if type(x) is dict:
            if self.panoptic:
                x = torch.cat([x["sem_seg"], x["ins_center"], x["ins_offset"], x["ins_edge"], x["ins_density"]], dim=1)
            else:
                x = x["sem_seg"]

        if self.original:
            b, c, h, w = x.shape
            x_cond = self.fc0(sem_cond).view(b, 8, 1, 1)
            x_cond = x_cond.repeat(1, 1, h, w)
            x = torch.cat([x, x_cond], dim=1)

        x = self.blocks[self.max_scale - self.scale].mask(x)

        for i in range(self.max_scale - self.scale, self.max_scale):
            x = self.blocks[i](x)
        return x

    def interpolate(self, x, alpha, sem_cond=None):
        if type(x) is dict:
            if self.panoptic:
                x = torch.cat([x["sem_seg"], x["ins_center"], x["ins_offset"], x["ins_edge"], x["ins_density"]], dim=1)
            else:
                x = x["sem_seg"]

        if self.original:
            b, c, h, w = x.shape
            x_cond = self.fc0(sem_cond).view(b, 8, 1, 1)
            x_cond = x_cond.repeat(1, 1, h, w)
            x = torch.cat([x, x_cond], dim=1)

        x_fine = self.blocks[self.max_scale - self.scale].mask(x)
        x_coarse = self.blocks[self.max_scale - self.scale + 1].mask(x, half=True)
        x_fine = self.blocks[self.max_scale - self.scale](x_fine)
        x = (1 - alpha) * x_coarse + alpha * x_fine
        for i in range(self.max_scale - self.scale + 1, self.max_scale):
            x = self.blocks[i](x)
        return x


class ProGANDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final=False, opt=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final = final
        self.original = "original_cgan" in opt.cond_mode

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.downsample = nn.AvgPool2d(2)

        mask_channels = get_seg_size(opt.num_semantics, opt.num_things, opt.panoptic, opt.instance_type)
        if self.original:
            mask_channels += 8

        self.from_mask = modules.EqualizedConv2d(mask_channels, in_channels, 1, gain=1)

        ar = int(opt.aspect_ratio) if final else 1

        self.conv1 = modules.EqualizedConv2d(in_channels, in_channels, 3, stride=(1, ar), padding=1)
        if final:
            self.conv2 = modules.EqualizedConv2d(in_channels, out_channels, 4)
            self.fc3 = modules.EqualizedLinear(out_channels, 1)
        else:
            self.conv2 = modules.EqualizedConv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        if self.final:
            x = x.view(x.size(0), -1)
            x = self.fc3(x)
        else:
            x = self.downsample(x)
        return x

    def mask(self, x, half=False):
        if half:
            x = self.downsample(x)
        return self.lrelu(self.from_mask(x))


