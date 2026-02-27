# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Attention blocks for ARES-compatible model definitions."""

from __future__ import annotations

from ._custom_layers import (
    QuantAlternatingAttention,
    QuantCrossAttention,
    QuantMultiHeadAttention,
    QuantRoPESelfAttention,
    QuantSelfAttention,
)

__all__ = [
    "QuantSelfAttention",
    "QuantMultiHeadAttention",
    "QuantRoPESelfAttention",
    "QuantCrossAttention",
    "QuantAlternatingAttention",
]
