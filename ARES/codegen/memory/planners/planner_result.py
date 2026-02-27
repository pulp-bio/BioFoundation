# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Planner result container for policy-based memory planners."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PlannerResult:
    """Structured output from a memory planner policy."""

    policy: str
    lifetimes: Dict[str, Dict[str, int]]
    offsets: Dict[str, int]
    total_size: int
    unresolved_conflicts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "policy": self.policy,
            "peak_arena_bytes": self.total_size,
            "lifetimes": self.lifetimes,
            "offsets": self.offsets,
            "unresolved_conflicts": self.unresolved_conflicts,
        }
