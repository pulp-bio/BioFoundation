# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Single-buffer strategy shim."""

from .strategy_base import TilingStrategyBase


class SingleBufferStrategy(TilingStrategyBase):
    name = "single_buffer"
