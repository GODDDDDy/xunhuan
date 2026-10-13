"""Exact author primitives, extracted without VGG download/GPU import side effects."""
import functools
import torch
from torch import nn

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class GlobalAvgPool(nn.Module):
    """(N,C,H,W) -> (N,C)"""

    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        return x.view(N, C, -1).mean(-1)

class SEBlock(nn.Module):
    """(N,C,H,W) -> (N,C,H,W)"""

    def __init__(self, in_channel, r):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_channel, in_channel // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // r, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        return x * se_weight

class ResBlock(nn.Module):
    """
        ResBlock using bottleneck structure
        dim -> dim
    """

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()

        sequence = [
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.model(x)
        return out
