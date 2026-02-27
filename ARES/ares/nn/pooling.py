# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Pooling layers for ARES-compatible model definitions."""

from __future__ import annotations

import torch.nn as nn


class MaxPool2d(nn.MaxPool2d):
    """MaxPool2d alias for consistent ares.nn imports."""


class AvgPool2d(nn.AvgPool2d):
    """AvgPool2d alias for consistent ares.nn imports."""


class GlobalAvgPool(nn.AdaptiveAvgPool2d):
    """Global average pooling helper (`output_size=1`)."""

    def __init__(self):
        super().__init__(1)


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    """AdaptiveAvgPool1d alias for consistent ares.nn imports."""
