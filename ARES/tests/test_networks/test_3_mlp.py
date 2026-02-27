# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 5: MLP - Multi-Layer Perceptron (No Convolutions)
Tests pure dense networks without any convolutions

Architecture:
  INPUT(28x28x1)
  → Flatten(784) → QuantIdentity
  → Linear(784→256) → ReLU → QuantIdentity
  → Linear(256→128) → ReLU → QuantIdentity
  → Linear(128→10)

Purpose: Test pure dense networks, different memory access patterns
Kernel difference: No convolutions, only linear layers
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity


class MLP(nn.Module):
    """Multi-layer perceptron with no convolutions."""

    def __init__(self):
        super(MLP, self).__init__()

        bit_width = 8

        # Input quantization (after flatten)
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 1: Linear(784→256) → ReLU
        self.fc1 = QuantLinear(784, 256, bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc1_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 2: Linear(256→128) → ReLU
        self.fc2 = QuantLinear(256, 128, bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc2_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 3: Linear(128→10) (output)
        self.fc3 = QuantLinear(128, 10, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.input_quant(x)

        # Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc1_quant(x)

        # Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc2_quant(x)

        # Layer 3 (output)
        x = self.fc3(x)

        return x
