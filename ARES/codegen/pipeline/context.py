# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Shared mutable state for the codegen pipeline runner."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PipelineContext:
    """Runtime context shared across codegen pipeline passes."""

    generator: Any
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    buffer_annotations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    memory_level_report: Dict[str, Any] = field(default_factory=dict)
    memory_hierarchy: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    stage_order: List[str] = field(default_factory=list)

    def add_timing(self, stage_name: str, elapsed_s: float) -> None:
        """Record elapsed time for one stage."""
        self.stage_timings[stage_name] = elapsed_s
        self.stage_order.append(stage_name)
