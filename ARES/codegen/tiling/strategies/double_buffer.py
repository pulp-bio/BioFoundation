# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Double-buffer strategy shim."""

from .strategy_base import TilingStrategyBase


class DoubleBufferStrategy(TilingStrategyBase):
    name = "double_buffer"
