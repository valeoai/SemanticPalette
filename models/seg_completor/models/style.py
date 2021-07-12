###############################################################################
# Code inspired from
# https://github.com/rosinality/style-based-gan-pytorch
# Modified the original code to turn it into a conditional gan
###############################################################################

import numpy as np
import random

import torch.nn as nn
from torch.nn import init
import torch

import models.seg_generator.modules as modules


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class StyleGANGenerator(nn.Module):
    def __init__(self, conditional_dim=0, latent_dim=512, n_mlp=8, max_dim=128, rgb=True, num_semantics=3, num_things=0,
                 T=1, aspect_ratio=1, cond_mode="", cond_seg=None, assisted_mul=1, panoptic=False, center_volume=None):
        super().__init__()
        if (max_dim & (max_dim - 1)) != 0 or max_dim == 0:
            raise ValueError("max_dim must be power of two")
        self.max_scale = int(np.log2(max_dim)) - 1
        self.scale = 1
        self.aspect_ratio = aspect_ratio

        blocks = []
        for i in range(self.max_scale):
            in_dim, out_dim = self._feature_dims_for_block(i)
            vol = center_volume[i] if center_volume is not None else None
            blocks.append(StyleGANGeneratorBlock(in_dim, out_dim, initial=i == 0, num_semantics=num_semantics, rgb=rgb,
                                               T=T, aspect_ratio=aspect_ratio, cond_mode=cond_mode, cond_seg=cond_seg,
                                               assisted_mul=assisted_mul, panoptic=panoptic, center_volume=vol,
                                               num_things=num_things, fused=i >= 5, style_dim=latent_dim))
        self.blocks = nn.ModuleList(blocks)

        style_layers = [modules.PixelNorm()]
        style_layers.append(modules.EqualizedLinear(latent_dim + conditional_dim, latent_dim))
        for i in range(n_mlp - 1):
            style_layers.append(modules.EqualizedLinear(latent_dim, latent_dim))
            style_layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*style_layers)

        self.rgb = rgb

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

    def _feature_dims_for_block(self, i):
        in_dim = min(512, 2 ** (13 - i))
        out_dim = min(512, 2 ** (13 - i - 1))
        return in_dim, out_dim

    def prepare_style(self, x, c, mean_style, style_weight):
        styles = []
        if c is not None:
            for i, z in enumerate(x):
                x[i] = torch.cat((z, *c), dim=1)
        for i in x:
            styles.append(self.style(i))
        if mean_style is not None:
            styles_norm = []
            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))
            styles = styles_norm
        return styles

    def prepare_noise(self, noise, batch_size, device):
        if noise is None:
            noise = []
            for i in range(self.scale):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch_size, 1, size, size * self.aspect_ratio, device=device))
        return noise

    def forward(self, x, c=None, return_a=False, noise=None, mean_style=None, style_weight=0, mixing_range=(-1, -1),
                interpolate=False):
        if type(x) not in (list, tuple):
            x = [x]
        style = self.prepare_style(x, c, mean_style, style_weight)
        noise = self.prepare_noise(noise, x[0].shape[0], x[0].device)

        x = noise[0]

        if len(style) < 2:
            inject_index = [len(self.blocks) + 1]
        else:
            inject_index = random.sample(list(range(self.scale - 1)), len(style) - 1)

        crossover = 0
        x_prev = None

        for i in range(self.scale):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))
                style_step = style[crossover]
            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]
                else:
                    style_step = style[0]

            if i == self.scale - 2:
                x_prev = x

            x = self.blocks[i](x, style_step, noise[i])

        if interpolate:
            return x, x_prev
        if self.rgb:
            return self.blocks[self.scale - 1].rgb(x)
        else:
            return self.blocks[self.scale - 1].mask(x, c, return_a=return_a)

    def interpolate(self, x, alpha, c=None, return_a=False, noise=None, mean_style=None, style_weight=0,
                    mixing_range=(-1, -1)):
        x, x_prev = self.forward(x, c=c, return_a=False, noise=noise, mean_style=mean_style,
                                 style_weight=style_weight, mixing_range=mixing_range, interpolate=True)

        if self.rgb:
            out_coarse = self.blocks[self.scale - 2].rgb(x_prev, double=True)
            out_fine = self.blocks[self.scale - 1].rgb(x)
        else:
            out_coarse, a_coarse = self.blocks[self.scale - 2].mask(x_prev, c, double=True, return_a=True)
            out_fine, a_fine = self.blocks[self.scale - 1].mask(x, c, return_a=True)

        if return_a:
            return (1 - alpha) * out_coarse + alpha * out_fine, (1 - alpha) * a_coarse + alpha * a_fine
        else:
            return (1 - alpha) * out_coarse + alpha * out_fine


class StyleGANGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initial=False, num_semantics=3, rgb=True, T=1, aspect_ratio=1,
                 cond_mode="", cond_seg=None, assisted_mul=1, panoptic=False, center_volume=1, num_things=0,
                 style_dim=512, fused=False):
        super().__init__()
        self.upsample = modules.NearestInterpolate(scale_factor=2)

        if initial:
            self.conv1 = modules.ConstantInput(in_channels, aspect_ratio=aspect_ratio)

        else:
            if fused:
                self.conv1 = nn.Sequential(modules.FusedUpsample(in_channels, out_channels, kernel_size=3, padding=1),
                                           modules.Blur(out_channels))

            else:
                self.conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                           modules.EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                           modules.Blur(out_channels))

        self.noise1 = modules.equalized_lr(modules.NoiseInjection(out_channels))
        self.adain1 = modules.AdaptiveInstanceNorm(out_channels, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = modules.EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.noise2 = modules.equalized_lr(modules.NoiseInjection(out_channels))
        self.adain2 = modules.AdaptiveInstanceNorm(out_channels, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

        if rgb:
            self.to_rgb = modules.EqualizedConv2d(out_channels, 3, 1, gain=1)
        else:
            self.to_mask = modules.CondMask(out_channels, num_semantics, num_things, cond_mode=cond_mode,
                                            cond_seg=cond_seg, assisted_mul=assisted_mul, panoptic=panoptic,
                                            center_volume=center_volume)

    def forward(self, x, style, noise):
        x = self.conv1(x)
        x = self.noise1(x, noise)
        x = self.lrelu1(x)
        x = self.adain1(x, style)
        x = self.conv2(x)
        x = self.noise2(x, noise)
        x = self.lrelu2(x)
        x = self.adain2(x, style)
        return x

    def rgb(self, x, double=False):
        if double:
            x = self.upsample(x)
        return self.to_rgb(x)

    def mask(self, x, c=None, double=False, return_a=False):
        if double:
            x = self.upsample(x)
        return self.to_mask(x, c, return_a=return_a)