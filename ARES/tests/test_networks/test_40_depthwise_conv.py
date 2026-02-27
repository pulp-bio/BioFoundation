# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 38: Depthwise Separable Convolutions (MobileNet-style)
Tests depthwise 2D convolutions for NE16 depthwise support development.

Architecture:
  INPUT(28x28x1)
  -> Conv2D(3x3, 16, padding=1) -> ReLU -> QuantIdentity  # Standard conv
  -> DepthwiseSeparable(16->32): DWConv(3x3, groups=16) -> ReLU -> PWConv(1x1, 32) -> ReLU
  -> MaxPool(2x2) -> QuantIdentity  # 14x14x32
  -> DepthwiseSeparable(32->64): DWConv(3x3, groups=32) -> ReLU -> PWConv(1x1, 64) -> ReLU
  -> MaxPool(2x2) -> QuantIdentity  # 7x7x64
  -> GlobalAvgPool -> Linear(64->10)

Purpose: Enable NE16 depthwise convolution support (NE16_FLAG_MODE_3x3_DW)
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable convolution: depthwise 3x3 + pointwise 1x1."""

    def __init__(self, in_channels, out_channels, bit_width=8):
        super(DepthwiseSeparableBlock, self).__init__()

        # Depthwise convolution (groups=in_channels)
        self.depthwise = QuantConv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels,  # Key: makes it depthwise
            bias=True,
            weight_bit_width=bit_width
        )
        self.depthwise_relu = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Pointwise convolution (1x1)
        self.pointwise = QuantConv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=True,
            weight_bit_width=bit_width
        )
        self.pointwise_relu = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x):
        # Depthwise
        x = self.depthwise(x)
        x = self.depthwise_relu(x)

        # Pointwise
        x = self.pointwise(x)
        x = self.pointwise_relu(x)

        return x


class DepthwiseConvNet(nn.Module):
    """MobileNet-style network with depthwise separable convolutions."""

    def __init__(self):
        super(DepthwiseConvNet, self).__init__()

        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Initial standard convolution
        self.conv1 = QuantConv2d(1, 16, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Depthwise separable block 1: 16 -> 32 channels
        self.dw_block1 = DepthwiseSeparableBlock(16, 32, bit_width)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Depthwise separable block 2: 32 -> 64 channels
        self.dw_block2 = DepthwiseSeparableBlock(32, 64, bit_width)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pre_linear_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.classifier = QuantLinear(64, 10, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        # Input quantization
        x = self.input_quant(x)

        # Initial conv
        x = self.conv1(x)
        x = self.relu1(x)

        # Depthwise separable block 1
        x = self.dw_block1(x)
        x = self.pool1(x)
        x = self.pool1_quant(x)

        # Depthwise separable block 2
        x = self.dw_block2(x)
        x = self.pool2(x)
        x = self.pool2_quant(x)

        # Global average pool
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.pre_linear_quant(x)

        # Classifier
        x = self.classifier(x)

        return x
