# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test NE16 Large: NE16 Accelerator Scaling Test - Larger Linear Layers

Tests NE16 efficiency with larger linear layers to measure scaling.
Larger layers should show better MACs/cycle as setup overhead becomes negligible.

Architecture (designed to fit in L2 to use NE16):
  INPUT(256)
  → QuantIdentity
  → Linear(256→256) → ReLU → QuantIdentity  [NE16: 49*256*256 = 3.2M MACs, weights=64KB]
  → Linear(256→256) → ReLU → QuantIdentity  [NE16: 49*256*256 = 3.2M MACs, weights=64KB]
  → Linear(256→256) → ReLU → QuantIdentity  [NE16: 49*256*256 = 3.2M MACs, weights=64KB]
  → Linear(256→256) → ReLU → QuantIdentity  [NE16: 49*256*256 = 3.2M MACs, weights=64KB]
  → Linear(256→10)                          [NE16: 49*256*10 = 125K MACs]

Total: ~13M MACs (vs 834K in test_38) - 15x larger workload
All layers fit in L2 (64KB each), so all use NE16

Purpose: Measure NE16 efficiency scaling with layer size
- Small layers (~200K MACs): ~1.5-2 MACs/cycle (setup overhead dominates)
- Large layers (~3M MACs): Target >2.5-3 MACs/cycle

Build and run with NE16:
  make clean all run platform=gvsoc USE_NE16=1 MINIMAL_OUTPUT=1
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity


class NE16LargeTest(nn.Module):
    """Large test network for NE16 efficiency scaling measurement."""

    def __init__(self):
        super(NE16LargeTest, self).__init__()

        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 1: Linear(256→256) → ReLU - Fits in L2 (64KB weights)
        self.fc1 = QuantLinear(256, 256, bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc1_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 2: Linear(256→256) → ReLU
        self.fc2 = QuantLinear(256, 256, bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc2_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 3: Linear(256→256) → ReLU
        self.fc3 = QuantLinear(256, 256, bias=True, weight_bit_width=bit_width)
        self.relu3 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc3_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 4: Linear(256→256) → ReLU
        self.fc4 = QuantLinear(256, 256, bias=True, weight_bit_width=bit_width)
        self.relu4 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.fc4_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 5: Linear(256→10) (output)
        self.fc5 = QuantLinear(256, 10, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        # x is expected to be shape (batch, 49, 256)
        x = self.input_quant(x)

        # Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc1_quant(x)

        # Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc2_quant(x)

        # Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc3_quant(x)

        # Layer 4
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc4_quant(x)

        # Layer 5 (output)
        x = self.fc5(x)

        return x
