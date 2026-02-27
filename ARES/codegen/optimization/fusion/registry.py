# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Registry scaffolding for fusion pattern matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

LayerSpec = Dict[str, Any]
FusionSpec = Dict[str, Any]
FusionMatchFn = Callable[[Sequence[LayerSpec], int], Optional[FusionSpec]]

DEFAULT_PATTERN_ORDER = (
    "conv_relu_quant",
    "linear_relu_quant",
    "conv_relu_maxpool_or_fallback",
    "conv_relu",
    "linear_relu",
    "pool_quant",
)


@dataclass(frozen=True)
class FusionPattern:
    """Single fusion pattern entry."""

    name: str
    min_layers: int
    match_fn: FusionMatchFn


class FusionRegistry:
    """Ordered fusion registry to preserve monolith matching precedence."""

    def __init__(self, patterns: Optional[Sequence[FusionPattern]] = None) -> None:
        self._patterns: List[FusionPattern] = list(patterns) if patterns else []

    def register(self, pattern: FusionPattern) -> None:
        self._patterns.append(pattern)

    def patterns(self) -> List[FusionPattern]:
        return list(self._patterns)


def build_default_registry() -> FusionRegistry:
    """Build default pattern order equivalent to current generator behavior."""
    from .matcher import (
        match_conv_relu,
        match_conv_relu_maxpool_or_fallback,
        match_conv_relu_quant,
        match_linear_relu,
        match_linear_relu_quant,
        match_pool_quant,
    )

    registry = FusionRegistry()
    registry.register(FusionPattern("conv_relu_quant", 3, match_conv_relu_quant))
    registry.register(FusionPattern("linear_relu_quant", 3, match_linear_relu_quant))
    registry.register(
        FusionPattern(
            "conv_relu_maxpool_or_fallback",
            3,
            match_conv_relu_maxpool_or_fallback,
        )
    )
    registry.register(FusionPattern("conv_relu", 2, match_conv_relu))
    registry.register(FusionPattern("linear_relu", 2, match_linear_relu))
    registry.register(FusionPattern("pool_quant", 2, match_pool_quant))
    return registry

