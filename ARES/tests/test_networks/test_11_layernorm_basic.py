# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

class QuantLayerNormBasic(nn.Module):
    """
    Simple test network to validate LayerNorm implementation.
    Architecture: Input -> Flatten -> LayerNorm -> Linear -> Output
    """
    def __init__(self):
        super(QuantLayerNormBasic, self).__init__()

        # Input quantization
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Input: (1, 1, 28, 28) = 784 features after flatten
        self.flatten = nn.Flatten()

        # LayerNorm on 784 features
        # Note: Brevitas doesn't have native QuantLayerNorm,
        # we'll use regular LayerNorm with QuantIdentity before/after
        self.pre_norm_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        self.layernorm = nn.LayerNorm(784)

        self.post_norm_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Classifier
        self.fc = qnn.QuantLinear(
            784, 10,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.flatten(x)
        x = self.pre_norm_quant(x)
        x = self.layernorm(x)
        x = self.post_norm_quant(x)
        x = self.fc(x)
        return x

def create_model():
    return QuantLayerNormBasic()

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
