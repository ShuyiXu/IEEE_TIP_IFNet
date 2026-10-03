"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy as np
import math



class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv2d(in_channels+1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1)) 
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
  

def t():
    import matplotlib.pyplot as plt

    canny_map_filters_in = 8
    canny_map = np.random.normal(size=(1, canny_map_filters_in, 10, 10))  # NxCxHxW
    resnet_map = np.random.normal(size=(1, 1, 10, 10))  # NxCxHxW
    plt.imshow(canny_map[0, 0])
    plt.show()

    canny_map = torch.from_numpy(canny_map).float()
    resnet_map = torch.from_numpy(resnet_map).float()

    gconv = GatedSpatialConv2d(canny_map_filters_in, canny_map_filters_in,
                               kernel_size=3, stride=1, padding=1)
    output_map = gconv(canny_map, resnet_map)
    print('done')


if __name__ == "__main__":
    t()

