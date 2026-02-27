# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

class QuantGELUBasic(nn.Module):
    """
    Simple test network to validate GELU implementation.
    Architecture: MNIST Input -> Flatten -> Linear -> GELU -> Linear -> Output
    """
    def __init__(self):
        super(QuantGELUBasic, self).__init__()

        # MNIST input: (1, 1, 28, 28) = 784 features after flatten
        self.flatten = nn.Flatten()

        # Quantize flattened input
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # First linear layer processes flattened MNIST tensor
        self.fc1 = qnn.QuantLinear(
            784, 64,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=True
        )

        # GELU activation
        # Note: Brevitas doesn't have native QuantGELU,
        # we'll use GELU with QuantIdentity after
        self.gelu = nn.GELU()

        self.post_gelu_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Classifier
        self.fc2 = qnn.QuantLinear(
            64, 10,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_quant(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.post_gelu_quant(x)
        x = self.fc2(x)
        return x

def create_model():
    return QuantGELUBasic()

def get_sample_input():
    return torch.randn(1, 1, 28, 28)

if __name__ == "__main__":
    model = create_model()
    model.eval()

    x = get_sample_input()
    with torch.no_grad():
        output = model(x)

    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
