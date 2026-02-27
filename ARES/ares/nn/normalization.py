# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Normalization layers for ARES-compatible model definitions."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    """LayerNorm alias for consistent ares.nn imports."""


class GroupNorm(nn.GroupNorm):
    """GroupNorm alias for consistent ares.nn imports."""


if hasattr(nn, "RMSNorm"):

    class RMSNorm(nn.RMSNorm):  # type: ignore[misc]
        """RMSNorm alias when available in the local PyTorch build."""

else:

    class RMSNorm(nn.Module):
        """
        Lightweight RMSNorm fallback with class name `RMSNorm`.

        The extractor supports RMSNorm via class-name fallback when nn.RMSNorm
        is unavailable.
        """

        def __init__(self, normalized_shape, eps: float = 1e-5):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
            return (x / rms) * self.weight
