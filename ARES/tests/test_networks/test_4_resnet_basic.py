# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 6: ResNet-Style Network
Tests Add (skip connection) and GlobalAvgPool operations

Architecture:
  INPUT(28x28x1)
  → QuantIdentity
  → Conv2D(3x3, 16, padding=1) → ReLU → QuantIdentity  ──┐
  → Conv2D(3x3, 16, padding=1) → ReLU → QuantIdentity ───┤
  → Add (skip connection) ← ──────────────────────────────┘
  → QuantIdentity
  → GlobalAvgPool (28x28 → 1x1)
  → QuantIdentity
  → Linear(16→10)

Purpose: Test Add and GlobalAvgPool operations
Expected: 0.0% error on GAP9
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from .brevitas_custom_layers import QuantAdd


class ResNetBasic(nn.Module):
    """ResNet-style network with skip connection and global average pooling."""

    def __init__(self):
        super(ResNetBasic, self).__init__()

        # Quantization config
        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Residual block - Identity path
        self.conv1 = QuantConv2d(1, 16, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.quant1 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Residual block - Residual path
        self.conv2 = QuantConv2d(16, 16, kernel_size=3, padding=1,
                                 bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.quant2 = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Skip connection addition
        self.add = QuantAdd(bit_width=bit_width, return_quant_tensor=True)
        self.add_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Classifier
        self.classifier = QuantLinear(16, 10, bias=True,
                                     weight_bit_width=bit_width)

    def forward(self, x):
        # Input quantization
        x = self.input_quant(x)

        # Identity path (save for skip connection)
        identity = self.conv1(x)
        identity = self.relu1(identity)
        identity = self.quant1(identity)

        # Residual path
        out = self.conv2(identity)
        out = self.relu2(out)
        out = self.quant2(out)

        # Skip connection
        out = self.add(identity, out)
        out = self.add_quant(out)

        # Global average pooling
        out = self.global_pool(out)
        out = self.pool_quant(out)

        # Flatten and classify
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out
