# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test NE16: NE16 Accelerator Validation - Linear Layers

Tests the NE16 hardware accelerator path for linear layers.
Includes layers of various sizes to test:
- Large layers (NE16-eligible): 64->256, 256->128
- Small layer (SW fallback): 128->10

Architecture:
  INPUT(64)
  → QuantIdentity
  → Linear(64→256) → ReLU → QuantIdentity   [NE16: 64*256 = 16,384 MACs]
  → Linear(256→128) → ReLU → QuantIdentity  [NE16: 256*128 = 32,768 MACs]
  → Linear(128→10)                          [NE16: 128*10 = 1,280 MACs]

Purpose: Validate NE16 accelerator integration
- Verify packed weight loading
- Verify S8→U8 input conversion
- Verify bias correction (bias - 128*sum(weights))
- Verify S32→S8 output requantization
- Compare SW vs NE16 outputs (should be identical)

Build and run with NE16:
  make clean all run platform=gvsoc USE_NE16=1
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity


class NE16LinearTest(nn.Module):
    """Test network for NE16 linear layer validation."""

    def __init__(self):
        super(NE16LinearTest, self).__init__()

        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 1: Linear(192→64) → ReLU
        # Matches tinymyo selftest dimensions exactly (in=192, out=64)
        self.fc1 = QuantLinear(192, 64, bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc1_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 2: Linear(64→64) → ReLU
        # Small layer to test NE16 with smaller dimensions
        self.fc2 = QuantLinear(64, 64, bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc2_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 3: Linear(64→10) (output)
        # Small output layer
        self.fc3 = QuantLinear(64, 10, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        # x is expected to be shape (batch, 64)
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
