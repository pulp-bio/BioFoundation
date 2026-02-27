# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Target abstraction package."""

from __future__ import annotations

from typing import Dict, List, Type

from .gap9_target import GAP9Target
from .siracusa_target import SiracusaTarget
from .target_base import MemoryCapabilities, TargetBase

TARGET_REGISTRY: Dict[str, Type[TargetBase]] = {
    GAP9Target.name: GAP9Target,
    SiracusaTarget.name: SiracusaTarget,
}


def create_target(name: str) -> TargetBase:
    """Create a target by canonical name."""
    canonical = (name or "").strip().lower()
    target_cls = TARGET_REGISTRY.get(canonical)
    if target_cls is None:
        available = ", ".join(sorted(TARGET_REGISTRY))
        raise ValueError(f"Unknown target '{name}'. Available targets: {available}")
    return target_cls()


def available_targets() -> List[str]:
    """Return sorted list of available target names."""
    return sorted(TARGET_REGISTRY.keys())


__all__ = [
    "GAP9Target",
    "SiracusaTarget",
    "MemoryCapabilities",
    "TargetBase",
    "TARGET_REGISTRY",
    "create_target",
    "available_targets",
]
