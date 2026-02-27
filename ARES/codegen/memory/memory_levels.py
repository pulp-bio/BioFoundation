# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Memory hierarchy primitives for explicit L1/L2/L3 modeling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class MemoryLevel(str, Enum):
    """Logical memory tiers used by the code generator."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


def as_memory_level(value: Optional[object]) -> Optional[MemoryLevel]:
    """Convert arbitrary input into a MemoryLevel enum when possible."""
    if value is None:
        return None
    if isinstance(value, MemoryLevel):
        return value
    return MemoryLevel(str(value).upper())


@dataclass(frozen=True)
class MemoryHierarchy:
    """Typed memory hierarchy with connectivity and optional capacities."""

    levels: Tuple[MemoryLevel, ...]
    adjacency: Dict[MemoryLevel, Tuple[MemoryLevel, ...]]
    capacities_bytes: Dict[MemoryLevel, Optional[int]]

    @classmethod
    def standard_3_level(
        cls,
        l1_bytes: Optional[int] = None,
        l2_bytes: Optional[int] = None,
        l3_bytes: Optional[int] = None,
    ) -> "MemoryHierarchy":
        """Construct a standard 3-level (L1/L2/L3) hierarchy graph."""
        levels = (MemoryLevel.L1, MemoryLevel.L2, MemoryLevel.L3)
        return cls(
            levels=levels,
            adjacency={
                MemoryLevel.L1: (MemoryLevel.L2,),
                MemoryLevel.L2: (MemoryLevel.L1, MemoryLevel.L3),
                MemoryLevel.L3: (MemoryLevel.L2,),
            },
            capacities_bytes={
                MemoryLevel.L1: l1_bytes,
                MemoryLevel.L2: l2_bytes,
                MemoryLevel.L3: l3_bytes,
            },
        )

    def neighbors(self, level: MemoryLevel) -> Tuple[MemoryLevel, ...]:
        """Return adjacent memory levels."""
        return self.adjacency.get(level, ())

    def path_levels(self, source: MemoryLevel, target: MemoryLevel) -> Tuple[MemoryLevel, ...]:
        """Find the shortest path between two levels (inclusive)."""
        if source == target:
            return (source,)
        queue = deque([(source, (source,))])
        visited = {source}
        while queue:
            node, path = queue.popleft()
            for neighbor in self.neighbors(node):
                if neighbor in visited:
                    continue
                next_path = path + (neighbor,)
                if neighbor == target:
                    return next_path
                visited.add(neighbor)
                queue.append((neighbor, next_path))
        return ()

    def path_names(self, source: MemoryLevel, target: MemoryLevel) -> List[str]:
        """Find a path and return level names."""
        return [level.value for level in self.path_levels(source, target)]

    def path_to_inner(self, source: MemoryLevel) -> List[str]:
        """Path helper for movement toward L1."""
        return self.path_names(source, MemoryLevel.L1)

    def path_to_outer(self, source: MemoryLevel) -> List[str]:
        """Path helper for movement toward L3."""
        return self.path_names(source, MemoryLevel.L3)

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "levels": [level.value for level in self.levels],
            "adjacency": {
                level.value: [neighbor.value for neighbor in neighbors]
                for level, neighbors in self.adjacency.items()
            },
            "capacities_bytes": {
                level.value: self.capacities_bytes.get(level)
                for level in self.levels
            },
        }
