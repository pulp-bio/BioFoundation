# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Element-wise quantized helper layers for ARES-compatible models."""

from __future__ import annotations

import torch
import torch.nn as nn
from brevitas.nn import QuantIdentity


def _val(tensor):
    """Return raw tensor value for both Tensor and QuantTensor inputs."""
    return tensor.value if hasattr(tensor, "value") else tensor


class QuantAdd(nn.Module):
    """Quantized residual add wrapper."""

    def __init__(self, bit_width: int = 8, return_quant_tensor: bool = True, **kwargs):
        super().__init__()
        self.quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
            **kwargs,
        )

    def forward(self, x1, x2):
        return self.quant(_val(x1) + _val(x2))


class QuantConcatenate(nn.Module):
    """Quantized concatenation wrapper."""

    def __init__(self, bit_width: int = 8, return_quant_tensor: bool = True, dim: int = 1, **kwargs):
        super().__init__()
        self.dim = dim
        self.quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
            **kwargs,
        )

    def forward(self, tensors):
        values = [_val(t) for t in tensors]
        out = torch.cat(values, dim=self.dim)
        return self.quant(out)


class QuantMean(nn.Module):
    """Quantized mean wrapper."""

    def __init__(
        self,
        dim: int = 1,
        keepdim: bool = False,
        bit_width: int = 8,
        return_quant_tensor: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.quant = QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=return_quant_tensor,
            **kwargs,
        )

    def forward(self, x):
        out = _val(x).mean(dim=self.dim, keepdim=self.keepdim)
        return self.quant(out)
