# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 7: Network with 1x1 Convolutions
Tests 1x1 kernels common in ResNet bottleneck blocks and MobileNet

Architecture:
  INPUT(28x28x1)
  -> QuantIdentity
  -> Conv2D(3x3, 32) -> ReLU
  -> MaxPool(2x2)
  -> Conv2D(1x1, 64) -> ReLU  (1x1 expand)
  -> Conv2D(1x1, 32) -> ReLU  (1x1 reduce)
  -> MaxPool(2x2)
  -> GlobalAvgPool (7x7 -> 1x1)
  -> QuantIdentity
  -> Linear(32->10)

Purpose: Test 1x1 convolutions for dimension reduction/expansion
Expected: 0.0% error on GAP9
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity


class BottleneckNet(nn.Module):
    """Network with 1x1 convolutions for dimension reduction/expansion.

    Uses MaxPool layers to reduce spatial dimensions, minimizing accumulated
    quantization error in GlobalAvgPool.
    """

    def __init__(self):
        super(BottleneckNet, self).__init__()

        # Quantization config
        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Initial 3x3 convolution (1->32 channels)
        self.conv1 = QuantConv2d(1, 32, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # 1x1 expand (32->64 channels)
        self.conv2 = QuantConv2d(32, 64, kernel_size=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # 1x1 reduce (64->32 channels)
        self.conv3 = QuantConv2d(64, 32, kernel_size=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu3 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Classifier
        self.classifier = QuantLinear(32, 10, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        # Input quantization
        x = self.input_quant(x)

        # 3x3 conv + maxpool
        x = self.conv1(x)    # 3x3: 1->32
        x = self.relu1(x)
        x = self.pool1(x)    # 28x28 -> 14x14

        # 1x1 expand -> 1x1 reduce (bottleneck pattern)
        x = self.conv2(x)    # 1x1: 32->64 (expand)
        x = self.relu2(x)

        x = self.conv3(x)    # 1x1: 64->32 (reduce)
        x = self.relu3(x)
        x = self.pool2(x)    # 14x14 -> 7x7

        # Global average pooling
        x = self.global_pool(x)
        x = self.pool_quant(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
