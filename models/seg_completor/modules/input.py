import torch
import torch.nn as nn

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4, aspect_ratio=1):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size * aspect_ratio))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out