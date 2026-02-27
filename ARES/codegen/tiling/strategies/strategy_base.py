# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Tiling strategy base primitives.

This module defines a small strategy contract used by
`gap9_model.compute_tile_plan`. The default behavior keeps output aligned with
the existing tiling calculators while exposing strategy selection/reporting.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


PARITY_STRATEGY_NAME = "parity_default"


@dataclass
class TilePlanDecision:
    config: Any
    selected_strategy: str
    requested_strategy: str
    rejected_strategies: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "selected_strategy": self.selected_strategy,
            "requested_strategy": self.requested_strategy,
            "rejected_strategies": self.rejected_strategies,
        }


class TilingStrategyBase:
    """
    Base strategy class.

    Concrete strategies may override `compute`, but the default implementation
    uses this shared mapping behavior: call the default calculator and report
    that non-default
    strategy requests are mapped back to `parity_default`.
    """

    name = PARITY_STRATEGY_NAME

    def compute(
        self,
        op_spec: Dict[str, Any],
        memory_constraints: Optional[Dict[str, Any]],
        parity_compute_fn: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Any],
        requested_strategy: str,
    ) -> TilePlanDecision:
        config = parity_compute_fn(op_spec, memory_constraints)
        rejected_strategies: List[Dict[str, str]] = []
        if requested_strategy != PARITY_STRATEGY_NAME:
            rejected_strategies.append({
                "strategy": requested_strategy,
                "reason": "mapped_to_parity_default",
            })

        return TilePlanDecision(
            config=config,
            selected_strategy=PARITY_STRATEGY_NAME,
            requested_strategy=requested_strategy,
            rejected_strategies=rejected_strategies,
        )


class ParityDefaultStrategy(TilingStrategyBase):
    name = PARITY_STRATEGY_NAME
