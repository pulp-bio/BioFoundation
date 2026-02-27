# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Convolution wrappers for ARES-compatible model definitions."""

from __future__ import annotations

import brevitas.nn as qnn


class Conv2d(qnn.QuantConv2d):
    """Quantized Conv2d wrapper with ARES-friendly defaults."""

    def __init__(self, *args, bit_width: int = 8, return_quant_tensor: bool = True, **kwargs):
        kwargs.setdefault("weight_bit_width", bit_width)
        kwargs.setdefault("return_quant_tensor", return_quant_tensor)
        super().__init__(*args, **kwargs)
