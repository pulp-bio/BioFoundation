# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Tiling strategy registry and dispatcher."""

from typing import Any, Callable, Dict, Optional, Type

from .strategy_base import (
    PARITY_STRATEGY_NAME,
    ParityDefaultStrategy,
    TilingStrategyBase,
)
from .single_buffer import SingleBufferStrategy
from .double_buffer import DoubleBufferStrategy
from .triple_buffer_weights import TripleBufferWeightsStrategy


_STRATEGY_REGISTRY: Dict[str, Type[TilingStrategyBase]] = {
    PARITY_STRATEGY_NAME: ParityDefaultStrategy,
    "single_buffer": SingleBufferStrategy,
    "double_buffer": DoubleBufferStrategy,
    "triple_buffer_weights": TripleBufferWeightsStrategy,
}


def list_supported_tiling_strategies():
    return list(_STRATEGY_REGISTRY.keys())


def compute_tile_plan_with_strategy(
    op_spec: Dict[str, Any],
    memory_constraints: Optional[Dict[str, Any]],
    strategy: Optional[str],
    parity_compute_fn: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Any],
) -> Dict[str, Any]:
    if not isinstance(op_spec, dict):
        raise ValueError("op_spec must be a dict")
    if "op_type" not in op_spec:
        raise ValueError("op_spec must include 'op_type'")
    if not callable(parity_compute_fn):
        raise ValueError("parity_compute_fn must be callable")

    requested_strategy = strategy or op_spec.get("strategy") or PARITY_STRATEGY_NAME
    unsupported_rejections = []

    strategy_cls = _STRATEGY_REGISTRY.get(requested_strategy)
    strategy_request_for_compute = requested_strategy

    if strategy_cls is None:
        unsupported_rejections.append({
            "strategy": requested_strategy,
            "reason": "unsupported_strategy",
        })
        strategy_cls = _STRATEGY_REGISTRY[PARITY_STRATEGY_NAME]
        strategy_request_for_compute = PARITY_STRATEGY_NAME

    strategy_impl = strategy_cls()
    decision = strategy_impl.compute(
        op_spec=op_spec,
        memory_constraints=memory_constraints,
        parity_compute_fn=parity_compute_fn,
        requested_strategy=strategy_request_for_compute,
    )

    if decision.requested_strategy != requested_strategy:
        decision.requested_strategy = requested_strategy
    if unsupported_rejections:
        decision.rejected_strategies = unsupported_rejections + decision.rejected_strategies

    return decision.to_dict()
