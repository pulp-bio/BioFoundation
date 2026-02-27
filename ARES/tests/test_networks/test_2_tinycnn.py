# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 3: TinyCNN - Minimal Network with 5x5 Kernels
Tests larger kernel size and minimal architecture

Architecture:
  INPUT(28x28x1)
  → Conv2D(5x5, 8, padding=2) → ReLU → MaxPool(2x2) → QuantIdentity  # 14x14x8
  → Flatten(1568) → QuantIdentity
  → Linear(1568→10)

Purpose: Edge case testing with larger kernels, fast iteration
Kernel difference: Uses 5x5 instead of 3x3
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity


class TinyCNN(nn.Module):
    """Minimal CNN with 5x5 kernel."""

    def __init__(self):
        super(TinyCNN, self).__init__()

        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Single conv block with 5x5 kernel
        self.conv1 = QuantConv2d(1, 8, kernel_size=5, padding=2,
                                 bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Flatten + QuantIdentity
        self.pre_linear_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Classifier
        self.classifier = QuantLinear(14 * 14 * 8, 10, bias=True,
                                     weight_bit_width=bit_width)

    def forward(self, x):
        x = self.input_quant(x)

        # Conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.pool1_quant(x)

        # Flatten
        x = x.view(x.size(0), -1)
        x = self.pre_linear_quant(x)

        # Classifier
        x = self.classifier(x)

        return x
