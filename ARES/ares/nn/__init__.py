# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""User-facing ARES model-building blocks."""

from __future__ import annotations

import brevitas.nn as qnn
import torch.nn as nn

from .activation import GELU, ReLU, SiLU
from .attention import (
    QuantAlternatingAttention,
    QuantCrossAttention,
    QuantMultiHeadAttention,
    QuantRoPESelfAttention,
    QuantSelfAttention,
)
from .blocks import ResidualBlock, TransformerBlock
from .compat import check_compatibility
from .conv import Conv2d
from .elementwise import QuantAdd, QuantConcatenate, QuantMean
from .embedding import Embedding, PatchEmbed, PositionalEmbedding
from .linear import Linear
from .mamba import (
    QuantConv1dDepthwise,
    QuantMambaBlock,
    QuantMambaWrapper,
    QuantSSM,
    QuantSiLU,
)
from .normalization import GroupNorm, LayerNorm, RMSNorm
from .pooling import AdaptiveAvgPool1d, AvgPool2d, GlobalAvgPool, MaxPool2d
from .quant_config import DEFAULT_BIT_WIDTH, default_quant_identity
from .reshape import Flatten, Permute, Reshape, Squeeze

# Alias to keep ares.nn API ergonomic for common input quant wrappers.
QuantIdentity = qnn.QuantIdentity
ZeroPad2d = nn.ZeroPad2d
QuantLinear = qnn.QuantLinear
QuantConv2d = qnn.QuantConv2d
QuantReLU = qnn.QuantReLU

__all__ = [
    "DEFAULT_BIT_WIDTH",
    "default_quant_identity",
    "check_compatibility",
    "QuantIdentity",
    "QuantLinear",
    "QuantConv2d",
    "QuantReLU",
    "ZeroPad2d",
    "Conv2d",
    "Linear",
    "ReLU",
    "GELU",
    "SiLU",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "MaxPool2d",
    "AvgPool2d",
    "GlobalAvgPool",
    "AdaptiveAvgPool1d",
    "Flatten",
    "Squeeze",
    "Reshape",
    "Permute",
    "QuantAdd",
    "QuantConcatenate",
    "QuantMean",
    "Embedding",
    "PatchEmbed",
    "PositionalEmbedding",
    "QuantSelfAttention",
    "QuantMultiHeadAttention",
    "QuantRoPESelfAttention",
    "QuantCrossAttention",
    "QuantAlternatingAttention",
    "QuantConv1dDepthwise",
    "QuantSiLU",
    "QuantSSM",
    "QuantMambaWrapper",
    "QuantMambaBlock",
    "TransformerBlock",
    "ResidualBlock",
]
