# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Shape/layout helper layers with extractor-recognized class names."""

from __future__ import annotations

import torch.nn as nn


class Flatten(nn.Flatten):
    """Flatten alias for consistent ares.nn imports."""


class Squeeze(nn.Module):
    """Extractor-recognized squeeze helper (class name: Squeeze)."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Reshape(nn.Module):
    """Extractor-recognized reshape helper (class name: Reshape)."""

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class Permute(nn.Module):
    """Extractor-recognized permute helper (class name: Permute)."""

    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
