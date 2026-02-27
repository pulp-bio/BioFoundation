# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 41: Large Depthwise Separable Convolutions (NE16 Beneficial)
Tests depthwise convolutions with large channel counts where NE16 provides speedup.

Architecture:
  INPUT(64x64x3) - synthetic RGB input
  -> Conv2D(3x3, 64, stride=1) -> ReLU       # 64x64x64
  -> DepthwiseSeparable(64->64, stride=1)    # 64x64x64  (DW: 64*64*64*9 = 2.36M MACs) [OK] NE16
  -> MaxPool(2x2)                            # 32x32x64
  -> DepthwiseSeparable(64->128, stride=1)   # 32x32x128 (DW: 64*32*32*9 = 589K MACs) [OK] NE16
  -> MaxPool(2x2)                            # 16x16x128
  -> DepthwiseSeparable(128->256, stride=1)  # 16x16x256 (DW: 128*16*16*9 = 294K MACs) [OK] NE16
  -> MaxPool(2x2)                            # 8x8x256
  -> GlobalAvgPool -> Linear(256->10)

MACs analysis for NE16 depthwise threshold (200K):
  - Block 1 DW: 64 ch * 64*64 * 9 = 2,359,296 MACs  [OK] NE16
  - Block 2 DW: 64 ch * 32*32 * 9 = 589,824 MACs    [OK] NE16
  - Block 3 DW: 128 ch * 16*16 * 9 = 294,912 MACs   [OK] NE16

Purpose: Benchmark NE16 depthwise vs software depthwise on larger layers
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable convolution with optional stride."""

    def __init__(self, in_channels, out_channels, stride=1, bit_width=8):
        super(DepthwiseSeparableBlock, self).__init__()

        # Depthwise convolution (groups=in_channels)
        self.depthwise = QuantConv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1, stride=stride,
            groups=in_channels,
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
        x = self.depthwise(x)
        x = self.depthwise_relu(x)
        x = self.pointwise(x)
        x = self.pointwise_relu(x)
        return x


class LargeDepthwiseNet(nn.Module):
    """MobileNet-style network with large depthwise layers for NE16 benchmarking."""

    def __init__(self):
        super(LargeDepthwiseNet, self).__init__()

        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Initial convolution: 64x64x3 -> 64x64x64 (stride=1 to keep spatial large)
        self.conv1 = QuantConv2d(3, 64, kernel_size=3, padding=1, stride=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Block 1: 64x64x64 -> 64x64x64 (DW: 2.36M MACs - NE16 eligible)
        self.block1 = DepthwiseSeparableBlock(64, 64, stride=1, bit_width=bit_width)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 32x32x64
        self.quant1 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Block 2: 32x32x64 -> 32x32x128 (DW: 589K MACs - NE16 eligible)
        self.block2 = DepthwiseSeparableBlock(64, 128, stride=1, bit_width=bit_width)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 16x16x128
        self.quant2 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Block 3: 16x16x128 -> 16x16x256 (DW: 294K MACs - NE16 eligible)
        self.block3 = DepthwiseSeparableBlock(128, 256, stride=1, bit_width=bit_width)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 8x8x256
        self.quant3 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pre_linear_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.classifier = QuantLinear(256, 10, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        x = self.input_quant(x)

        # Initial conv
        x = self.conv1(x)
        x = self.relu1(x)

        # Depthwise separable blocks with pooling
        x = self.block1(x)
        x = self.pool1(x)
        x = self.quant1(x)

        x = self.block2(x)
        x = self.pool2(x)
        x = self.quant2(x)

        x = self.block3(x)
        x = self.pool3(x)
        x = self.quant3(x)

        # Global average pool
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.pre_linear_quant(x)

        # Classifier
        x = self.classifier(x)

        return x
