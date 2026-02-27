# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Default quantization helpers for ARES model-building wrappers."""

from __future__ import annotations

import brevitas.nn as qnn


DEFAULT_BIT_WIDTH = 8


def default_quant_identity(bit_width: int = DEFAULT_BIT_WIDTH, return_quant_tensor: bool = True, **kwargs):
    """Create a QuantIdentity with ARES-tested defaults."""
    return qnn.QuantIdentity(
        bit_width=bit_width,
        return_quant_tensor=return_quant_tensor,
        **kwargs,
    )
