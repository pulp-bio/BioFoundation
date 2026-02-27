# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Triple-buffer-weights strategy shim."""

from .strategy_base import TilingStrategyBase


class TripleBufferWeightsStrategy(TilingStrategyBase):
    name = "triple_buffer_weights"
