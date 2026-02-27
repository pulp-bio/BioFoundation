# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Planner policies and factory for memory planning."""

from typing import Dict, Type

from .arena_first_fit import ArenaFirstFitPlanner, SizePriorityPlanner
from .planner_base import PlannerBase, PlannerPolicyError
from .planner_result import PlannerResult

_PLANNER_REGISTRY: Dict[str, Type[PlannerBase]] = {
    ArenaFirstFitPlanner.policy_name: ArenaFirstFitPlanner,
    SizePriorityPlanner.policy_name: SizePriorityPlanner,
}


def available_planner_policies() -> Dict[str, bool]:
    """Return policy -> experimental flag map."""
    return {
        name: planner_cls.is_experimental
        for name, planner_cls in sorted(_PLANNER_REGISTRY.items())
    }


def create_planner(policy: str, allow_experimental: bool = False) -> PlannerBase:
    """Instantiate a planner policy by name."""
    normalized = (policy or ArenaFirstFitPlanner.policy_name).strip().lower()
    if normalized in ("arena", "arena_first_fit", "arena-first-fit"):
        normalized = ArenaFirstFitPlanner.policy_name

    planner_cls = _PLANNER_REGISTRY.get(normalized)
    if planner_cls is None:
        choices = ", ".join(sorted(_PLANNER_REGISTRY.keys()))
        raise PlannerPolicyError(f"Unknown planner policy '{policy}'. Available policies: {choices}")

    if planner_cls.is_experimental and not allow_experimental:
        raise PlannerPolicyError(
            f"Planner policy '{normalized}' is experimental and disabled. "
            "Set ARES_ENABLE_EXPERIMENTAL_PLANNERS=1 to enable it."
        )

    return planner_cls()


__all__ = [
    "PlannerBase",
    "PlannerPolicyError",
    "PlannerResult",
    "ArenaFirstFitPlanner",
    "SizePriorityPlanner",
    "available_planner_policies",
    "create_planner",
]
