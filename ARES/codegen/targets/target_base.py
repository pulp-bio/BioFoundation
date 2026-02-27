# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Target abstraction primitives for codegen planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class MemoryCapabilities:
    """Target memory capacities and planning thresholds."""

    l1_bytes: Optional[int]
    l2_bytes: Optional[int]
    l3_bytes: Optional[int]
    l2_tiling_bytes: Optional[int]
    l3_stage_threshold_bytes: Optional[int]
    # Optional allocator-visible memory totals and fallback planning knobs.
    l1_total_bytes: Optional[int] = None
    l2_total_bytes: Optional[int] = None
    l2_activation_reserved_bytes: Optional[int] = None
    l3_fallback_single_buffer_threshold_bytes: Optional[int] = None


class TargetBase(ABC):
    """Minimal target contract used by codegen decisions."""

    name = "target_base"
    display_name = "Generic Target"

    @property
    @abstractmethod
    def memory(self) -> MemoryCapabilities:
        """Return target memory capabilities."""

    def validate_required_capabilities(self) -> None:
        """
        Validate that required planning capabilities are present.

        These capabilities must be explicit so codegen does not silently fall
        back to GAP9-specific defaults in generic paths.
        """
        memory = self.memory
        missing = []
        if memory.l1_bytes is None:
            missing.append("memory.l1_bytes")
        if memory.l2_bytes is None:
            missing.append("memory.l2_bytes")
        if memory.l2_tiling_bytes is None:
            missing.append("memory.l2_tiling_bytes")
        if memory.l1_total_bytes is None:
            missing.append("memory.l1_total_bytes")
        if memory.l2_total_bytes is None:
            missing.append("memory.l2_total_bytes")
        if memory.l3_fallback_single_buffer_threshold_bytes is None:
            missing.append("memory.l3_fallback_single_buffer_threshold_bytes")

        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"Target '{self.name}' is missing required capabilities: {missing_list}"
            )

    @property
    def l1_budget_bytes(self) -> Optional[int]:
        return self.memory.l1_bytes

    @property
    def l2_budget_bytes(self) -> Optional[int]:
        return self.memory.l2_bytes

    @property
    def l2_tiling_budget_bytes(self) -> Optional[int]:
        return self.memory.l2_tiling_bytes

    @property
    def l3_stage_threshold_bytes(self) -> Optional[int]:
        return self.memory.l3_stage_threshold_bytes

    @property
    def large_weight_threshold_bytes(self) -> int:
        """Threshold above which linear weights use shared slab allocation."""
        l1 = self.memory.l1_bytes
        if l1 is not None:
            return l1
        return 128 * 1024

    @abstractmethod
    def determine_weight_residency(
        self,
        weight_size_bytes: int,
        layer_type: str,
        memory_tier: Optional[str] = None,
        uses_mamba_scratch: bool = False,
    ) -> str:
        """Classify where layer weights should reside at runtime."""

    @abstractmethod
    def supports_ne16_linear(self) -> bool:
        """Return True when NE16 Linear kernels are available."""

    @abstractmethod
    def supports_ne16_conv2d_kernel(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        groups: int,
        in_channels: int,
        use_hwc_layout: bool,
        memory_tier: Optional[str] = None,
    ) -> bool:
        """Return True when target can execute this Conv2D via NE16."""
