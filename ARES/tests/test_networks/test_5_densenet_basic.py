# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 7: DenseNet-Style Network
Tests Concatenate and AvgPool operations

Architecture:
  INPUT(28x28x1)
  → QuantIdentity
  → Conv2D(3x3, 8, padding=1) → ReLU → QuantIdentity  ──┐
  → Conv2D(3x3, 8, padding=1) → ReLU → QuantIdentity ───┤
  → Concatenate ← ────────────────────────────────────────┘
  → QuantIdentity (16 channels)
  → AvgPool(2x2, stride=2) → QuantIdentity  # 14x14x16
  → Conv2D(3x3, 16, padding=1) → ReLU → QuantIdentity
  → MaxPool(2x2, stride=2) → QuantIdentity  # 7x7x16
  → Flatten(784) → QuantIdentity
  → Linear(784→10)

Purpose: Test Concatenate and AvgPool operations
Expected: 0.0% error on GAP9
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from .brevitas_custom_layers import QuantConcatenate


class DenseNetBasic(nn.Module):
    """DenseNet-style network with dense connections and average pooling."""

    def __init__(self):
        super(DenseNetBasic, self).__init__()

        # Quantization config
        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Dense block - First layer
        self.conv1 = QuantConv2d(1, 8, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.quant1 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Dense block - Second layer
        self.conv2 = QuantConv2d(8, 8, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.quant2 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Dense concatenation
        self.concat = QuantConcatenate(bit_width=bit_width, return_quant_tensor=True, dim=1)
        self.concat_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Transition layer with average pooling
        self.avgpool1 = nn.AvgPool2d(2, stride=2)
        self.avgpool_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Additional conv block
        self.conv3 = QuantConv2d(16, 16, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu3 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.quant3 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Max pooling
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.maxpool_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Flatten quantization
        self.pre_linear_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Classifier
        self.classifier = QuantLinear(7 * 7 * 16, 10, bias=True,
                                     weight_bit_width=bit_width)

    def forward(self, x):
        # Input quantization
        x = self.input_quant(x)

        # Dense block - First layer
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1 = self.quant1(x1)

        # Dense block - Second layer (input is x1, not x)
        x2 = self.conv2(x1)
        x2 = self.relu2(x2)
        x2 = self.quant2(x2)

        # Concatenate along channel dimension
        x_cat = self.concat([x1, x2])
        x_cat = self.concat_quant(x_cat)

        # Transition layer with average pooling
        x_cat = self.avgpool1(x_cat)
        x_cat = self.avgpool_quant(x_cat)

        # Additional conv block
        x3 = self.conv3(x_cat)
        x3 = self.relu3(x3)
        x3 = self.quant3(x3)

        # Max pooling
        x3 = self.maxpool(x3)
        x3 = self.maxpool_quant(x3)

        # Flatten
        x3 = x3.view(x3.size(0), -1)
        x3 = self.pre_linear_quant(x3)

        # Classifier
        out = self.classifier(x3)

        return out
