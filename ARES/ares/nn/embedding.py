# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Embedding-related layers for ARES-compatible model definitions."""

from __future__ import annotations

import torch
import torch.nn as nn

from ._custom_layers import QuantPatchEmbed as PatchEmbed


class Embedding(nn.Embedding):
    """Embedding alias for consistent ares.nn imports."""


class PositionalEmbedding(nn.Module):
    """
    Minimal learnable positional embedding block.

    Exposes `pos_embed` parameter and `pos_quant` hook to follow extraction
    patterns used by FEMBA models.

    Notes:
    - Default `pos_quant` is `nn.Identity()` for minimal behavior.
    - For explicit quantization boundaries/scale extraction, replace
      `self.pos_quant` with `brevitas.nn.QuantIdentity(...)` in your model.
    """

    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.pos_quant = nn.Identity()

    def forward(self, x):
        return x + self.pos_quant(self.pos_embed)
