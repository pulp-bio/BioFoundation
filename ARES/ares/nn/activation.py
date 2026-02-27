# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Activation wrappers for ARES-compatible model definitions."""

from __future__ import annotations

import torch.nn as nn
import brevitas.nn as qnn

from ._custom_layers import QuantSiLU


class ReLU(qnn.QuantReLU):
    """Quantized ReLU wrapper with ARES-friendly defaults."""

    def __init__(self, *args, bit_width: int = 8, return_quant_tensor: bool = True, **kwargs):
        kwargs.setdefault("bit_width", bit_width)
        kwargs.setdefault("return_quant_tensor", return_quant_tensor)
        super().__init__(*args, **kwargs)


class GELU(nn.GELU):
    """GELU alias for consistent ares.nn imports."""


class SiLU(QuantSiLU):
    """
    Quantized SiLU custom layer recognized by the extractor.

    Use this class (not `torch.nn.SiLU`) when you want the operation mapped to
    the ARES `SiLU` extraction/runtime path.
    """
