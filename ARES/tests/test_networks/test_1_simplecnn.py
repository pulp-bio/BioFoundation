# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 1: SimpleCNN - Baseline Reference
Current working architecture with 3x3 kernels

Architecture:
  INPUT(28x28x1)
  → Conv2D(3x3, 16, padding=1) → ReLU → MaxPool(2x2) → QuantIdentity  # 14x14x16
  → Conv2D(3x3, 32, padding=1) → ReLU → MaxPool(2x2) → QuantIdentity  # 7x7x32
  → Flatten(1568) → QuantIdentity
  → Linear(1568→10)

Purpose: Baseline reference (already verified at 0.0% error)
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity


class SimpleCNN(nn.Module):
    """Baseline SimpleCNN with 3x3 kernels."""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Quantization config
        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Block 1: Conv(3x3) → ReLU → MaxPool
        self.conv1 = QuantConv2d(1, 16, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Block 2: Conv(3x3) → ReLU → MaxPool
        self.conv2 = QuantConv2d(16, 32, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Flatten + QuantIdentity
        self.pre_linear_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Classifier
        self.classifier = QuantLinear(7 * 7 * 32, 10, bias=True,
                                     weight_bit_width=bit_width)

    def forward(self, x):
        # Input quantization
        x = self.input_quant(x)

        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.pool1_quant(x)

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.pool2_quant(x)

        # Flatten
        x = x.view(x.size(0), -1)
        x = self.pre_linear_quant(x)

        # Classifier
        x = self.classifier(x)

        return x
