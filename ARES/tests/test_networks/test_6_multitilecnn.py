# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Test 9: MultiTileCNN – forces Conv2D tiling with manageable L1 usage."""

import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity


class MultiTileCNN(nn.Module):
    """Large-spatial CNN (96x96 input) with light channel counts for L1 tiling."""

    def __init__(self):
        super().__init__()
        bit_width = 8

        self.resize = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.conv1 = QuantConv2d(1, 8, kernel_size=3, padding=1, bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.conv2 = QuantConv2d(8, 16, kernel_size=3, padding=1, bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.conv3 = QuantConv2d(16, 24, kernel_size=3, padding=1, bias=True, weight_bit_width=bit_width)
        self.relu3 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.pre_linear_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.classifier = QuantLinear(24 * 6 * 6, 10, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        x = self.resize(x)
        x = self.input_quant(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.pool1_quant(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.pool2_quant(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.pool3_quant(x)

        x = self.pool4(x)
        x = self.pool4_quant(x)

        x = x.view(x.size(0), -1)
        x = self.pre_linear_quant(x)
        x = self.classifier(x)
        return x
