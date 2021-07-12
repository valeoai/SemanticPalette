import os
import torch
import torch.nn as nn

## depthwise seperable conv + spectral norm + batch norm
class DepthConv(nn.Module):
    def __init__(self, fmiddle, opt, kw=3, padding=1, stride=1):
        super().__init__()
        self.kw = kw
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=(self.kw,self.kw), dilation=1, padding=1, stride=stride)
        if int(os.environ['WORLD_SIZE']) > 1:
            BNFunc = nn.SyncBatchNorm
        else:
            BNFunc = nn.BatchNorm2d
        self.norm_layer = BNFunc(fmiddle, affine=True)
        
    def forward(self, x, conv_weights):
        N, C, H, W = x.size()
        conv_weights = conv_weights.view(N * C, self.kw * self.kw, H//self.stride, W//self.stride)
        x = self.unfold(x).view(N * C, self.kw * self.kw, H//self.stride, W//self.stride)
        x = torch.mul(conv_weights, x).sum(dim=1, keepdim=False).view(N, C, H//self.stride, W//self.stride)
        return x


