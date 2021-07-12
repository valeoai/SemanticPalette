import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        gain=math.sqrt(2),
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
        std = gain / math.sqrt(fan_in)
        self.scale = std

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        w = self.scale * self.weight
        return F.conv2d(
            input, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class EqualizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, gain=math.sqrt(2), bias=True):
        super().__init__(in_features, out_features, bias=bias)
        fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
        std = gain / math.sqrt(fan_in)
        self.scale = std

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        w = self.scale * self.weight
        return F.linear(input, w, self.bias)


class EqualizedLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualizedLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equalized_lr(module, name='weight'):
    EqualizedLR.apply(module, name)

    return module