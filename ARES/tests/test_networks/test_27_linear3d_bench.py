# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 37: 3D Linear (Multi-Token) Benchmark

Purpose: Fast benchmark to tune GAP9 tiling/requantization for transformer-style
MLP linear layers without running full MHSA/TinyMyo graphs.

Architecture:
  - Input: [B, seq_len, embed_dim]
  - MLP: fc1 (embed_dim -> hidden_dim), GELU, fc2 (hidden_dim -> embed_dim)
  - Pool: AdaptiveAvgPool1d over seq_len -> 1 token
  - Classifier: embed_dim -> num_classes (FP32 logits)
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantIdentity, QuantLinear
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat


class Permute(nn.Module):
    """Permute layer to rearrange tensor dimensions."""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Squeeze(nn.Module):
    """Squeeze a single dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Linear3DBenchQuant(nn.Module):
    """Multi-token MLP benchmark."""

    def __init__(
        self,
        seq_len: int = 400,
        embed_dim: int = 192,
        hidden_dim: int = 768,
        num_classes: int = 7,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.input_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Transformer-style MLP
        self.fc1 = QuantLinear(
            embed_dim,
            hidden_dim,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.gelu = nn.GELU()
        self.gelu_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.fc2 = QuantLinear(
            hidden_dim,
            embed_dim,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=False,
        )
        self.out_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # Pool tokens -> single embedding (use supported ops for codegen)
        self.permute = Permute(0, 2, 1)  # [B, seq, dim] -> [B, dim, seq]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)  # [B, dim, 1] -> [B, dim]
        self.pool_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        self.classifier = QuantLinear(
            embed_dim,
            num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            output_quant=None,  # FP32 output
            return_quant_tensor=False,
        )

    def forward(self, x):
        x = self.input_quant(x)
        if hasattr(x, "value"):
            x = x.value

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.gelu_quant(x)
        if hasattr(x, "value"):
            x = x.value

        x = self.fc2(x)
        x = self.out_quant(x)
        if hasattr(x, "value"):
            x = x.value

        x = self.permute(x)
        x = self.pool(x)
        x = self.squeeze(x)
        x = self.pool_quant(x)

        return self.classifier(x)


def create_model():
    return Linear3DBenchQuant()


def get_sample_input():
    # [B, seq_len, embed_dim]
    return torch.randn(1, 400, 192)


if __name__ == "__main__":
    model = create_model()
    model.eval()

    x = get_sample_input()
    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
