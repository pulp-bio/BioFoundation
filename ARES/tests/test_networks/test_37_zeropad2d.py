# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 37: ZeroPad2d Operation Test

Tests asymmetric padding (ZeroPad2d) which is commonly needed
for "same" padding equivalent in ONNX/Brevitas models.

Architecture:
  Input: (1, 1, 8, 64)
    |
  ZeroPad2d (1, 2, 0, 0)  - asymmetric width padding
    |
  Conv2d (1, 4) kernel
    |
  ReLU + MaxPool
    |
  ZeroPad2d (3, 4, 0, 0)  - different asymmetric padding
    |
  Conv2d (1, 8) kernel
    |
  ReLU + AvgPool
    |
  Flatten + Linear
    |
  Output: (1, 10)

Purpose: Validate ZeroPad2d support for asymmetric padding operations.
"""

import torch
import torch.nn as nn
from brevitas import nn as qnn


class ZeroPad2dTestNet(nn.Module):
    """Test network with ZeroPad2d layers for asymmetric padding."""

    def __init__(self, num_classes=10, bit_width=8):
        super().__init__()

        self.input_quant = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Block 1: ZeroPad + Conv + ReLU + Pool
        # Input: (1, 1, 8, 64)
        # After pad1: (1, 1, 8, 67) = 64 + 1 + 2
        # After conv1: (1, 4, 8, 64) = 67 - 4 + 1
        # After pool1: (1, 4, 8, 8) = 64 / 8
        self.pad1 = nn.ZeroPad2d((1, 2, 0, 0))  # (left, right, top, bottom)
        self.conv1 = qnn.QuantConv2d(1, 4, (1, 4), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d((1, 8))

        # Block 2: ZeroPad + Conv + ReLU + AvgPool
        # Input: (1, 4, 8, 8)
        # After pad2: (1, 4, 8, 15) = 8 + 3 + 4
        # After conv2: (1, 8, 8, 8) = 15 - 8 + 1
        # After pool2: (1, 8, 8, 2) = 8 / 4
        self.pad2 = nn.ZeroPad2d((3, 4, 0, 0))  # Different asymmetric padding
        self.conv2 = qnn.QuantConv2d(4, 8, (1, 8), padding=0, weight_bit_width=bit_width, return_quant_tensor=True)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.pool2 = nn.AvgPool2d((1, 4))

        # Classifier: Flatten + FC
        # Input: (1, 8, 8, 2) = 128 elements
        self.pre_fc_quant = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)
        self.flatten = nn.Flatten()
        self.fc = qnn.QuantLinear(128, num_classes, bias=True, weight_bit_width=bit_width, return_quant_tensor=False)

    def forward(self, x):
        x = self.input_quant(x)

        # Block 1
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Classifier
        x = self.pre_fc_quant(x)
        x = self.flatten(x)
        return self.fc(x)


def get_model():
    """Factory function for test generation."""
    return ZeroPad2dTestNet(num_classes=10, bit_width=8)


def get_input_shape():
    """Return expected input shape."""
    return (1, 1, 8, 64)


if __name__ == "__main__":
    model = get_model()
    model.eval()

    input_shape = get_input_shape()
    x = torch.randn(input_shape)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        out = model(x)

    print(f"Output shape: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("\nTest PASSED!")
