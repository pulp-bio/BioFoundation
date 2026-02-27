# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 42: Auto-Tuner Stress Test
Network with unusual dimensions to stress-test the auto-tuner.

Architecture:
  INPUT(1, 64)
  -> Linear(64 -> 768)   -> ReLU  # Wide expansion (12x)
  -> Linear(768 -> 768)  -> ReLU  # Square layer
  -> Linear(768 -> 237)  -> ReLU  # Odd contraction (prime-ish)
  -> Linear(237 -> 513)  -> ReLU  # Odd expansion
  -> Linear(513 -> 128)  -> ReLU  # Contraction
  -> Linear(128 -> 5)             # Classifier

Purpose: Test auto-tuner with non-standard dimensions that heuristics
may not handle optimally. These shapes are chosen to:
1. Have large expansion/contraction ratios
2. Use non-power-of-2 dimensions (237, 513)
3. Create scenarios where tile size choices significantly impact perf
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity


class AutotuneStressNet(nn.Module):
    """MLP with unusual dimensions to stress-test auto-tuner."""

    def __init__(self):
        super(AutotuneStressNet, self).__init__()

        bit_width = 8

        # Input quantization
        self.input_quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        # Layer 1: Wide expansion (64 -> 768)
        self.fc1 = QuantLinear(64, 768, bias=True, weight_bit_width=bit_width)
        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Layer 2: Square layer (768 -> 768)
        self.fc2 = QuantLinear(768, 768, bias=True, weight_bit_width=bit_width)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Layer 3: Odd contraction (768 -> 237)
        self.fc3 = QuantLinear(768, 237, bias=True, weight_bit_width=bit_width)
        self.relu3 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Layer 4: Odd expansion (237 -> 513)
        self.fc4 = QuantLinear(237, 513, bias=True, weight_bit_width=bit_width)
        self.relu4 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Layer 5: Contraction (513 -> 128)
        self.fc5 = QuantLinear(513, 128, bias=True, weight_bit_width=bit_width)
        self.relu5 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

        # Classifier (128 -> 5)
        self.classifier = QuantLinear(128, 5, bias=True, weight_bit_width=bit_width)

    def forward(self, x):
        x = self.input_quant(x)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.relu4(x)

        x = self.fc5(x)
        x = self.relu5(x)

        x = self.classifier(x)
        return x


# Test configuration
def get_model():
    return AutotuneStressNet()


def get_input_shape():
    return (1, 64)  # Single sample, 64 features


def get_num_classes():
    return 5


if __name__ == "__main__":
    model = get_model()
    model.eval()
    x = torch.randn(1, 64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
