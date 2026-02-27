# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Mamba/SSM blocks for ARES-compatible model definitions."""

from __future__ import annotations

from ._custom_layers import (
    QuantConv1dDepthwise,
    QuantMambaBlock,
    QuantMambaWrapper,
    QuantSSM,
    QuantSiLU,
)

__all__ = [
    "QuantConv1dDepthwise",
    "QuantSiLU",
    "QuantSSM",
    "QuantMambaWrapper",
    "QuantMambaBlock",
]
