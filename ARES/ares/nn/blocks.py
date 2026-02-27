# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Composite building blocks for ARES-compatible model definitions."""

from __future__ import annotations

import torch.nn as nn

from .activation import GELU, ReLU
from .attention import QuantMultiHeadAttention
from .conv import Conv2d
from .elementwise import QuantAdd
from .linear import Linear
from .normalization import LayerNorm
from .quant_config import default_quant_identity


class ResidualBlock(nn.Module):
    """Conv-ReLU-Conv residual block using explicit QuantAdd."""

    def __init__(self, channels: int, kernel_size: int = 3, bit_width: int = 8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
            bit_width=bit_width,
            return_quant_tensor=True,
        )
        self.relu = ReLU(bit_width=bit_width, return_quant_tensor=True)
        self.conv2 = Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
            bit_width=bit_width,
            return_quant_tensor=True,
        )
        self.add = QuantAdd(bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.add(identity, out)


class TransformerBlock(nn.Module):
    """LayerNorm + MHSA + residual, then LayerNorm + MLP + residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        seq_len: int = 128,
        bit_width: int = 8,
    ):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.norm1_quant = default_quant_identity(bit_width=bit_width, return_quant_tensor=True)
        self.attn = QuantMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            pool_sequence="none",
        )
        self.add1 = QuantAdd(bit_width=bit_width, return_quant_tensor=True)

        self.norm2 = LayerNorm(embed_dim)
        self.norm2_quant = default_quant_identity(bit_width=bit_width, return_quant_tensor=True)
        self.ff1 = Linear(embed_dim, ff_dim, bit_width=bit_width, return_quant_tensor=True)
        self.ff_act = GELU()
        self.ff_act_quant = default_quant_identity(bit_width=bit_width, return_quant_tensor=True)
        self.ff2 = Linear(ff_dim, embed_dim, bit_width=bit_width, return_quant_tensor=True)
        self.add2 = QuantAdd(bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        # QuantIdentity marks a scale boundary so extractor/runtime can track
        # the activation scale feeding MHSA.
        x = self.norm1_quant(x)
        x = self.attn(x)
        x = self.add1(x, residual)

        residual = x
        x = self.norm2(x)
        # Same boundary before the MLP branch; this matches tested model
        # patterns used by the extraction flow.
        x = self.norm2_quant(x)
        x = self.ff1(x)
        x = self.ff_act(x)
        x = self.ff_act_quant(x)
        x = self.ff2(x)
        x = self.add2(x, residual)
        return x
