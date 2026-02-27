# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""GAP9 target implementation for target-aware codegen."""

from __future__ import annotations

from typing import Optional, Tuple

from ..gap9_model import GAP9HardwareModel
from ..gap9_model import determine_weight_residency as gap9_determine_weight_residency
from .target_base import MemoryCapabilities, TargetBase


class GAP9Target(TargetBase):
    """Current production target used by the code generator."""

    name = "gap9"
    display_name = "GAP9"

    def __init__(self) -> None:
        self._memory = MemoryCapabilities(
            l1_bytes=GAP9HardwareModel.get_l1_budget(),
            l2_bytes=GAP9HardwareModel.get_l2_budget(),
            l3_bytes=None,
            l2_tiling_bytes=GAP9HardwareModel.get_l2_tiling_budget(),
            l3_stage_threshold_bytes=GAP9HardwareModel.get_l3_stage_threshold(),
            l1_total_bytes=GAP9HardwareModel.get_l1_total_bytes(),
            l2_total_bytes=GAP9HardwareModel.get_l2_total_bytes(),
            l2_activation_reserved_bytes=GAP9HardwareModel.get_l2_activation_reserved_bytes(),
            l3_fallback_single_buffer_threshold_bytes=(
                GAP9HardwareModel.get_l3_fallback_single_buffer_threshold_bytes()
            ),
        )

    @property
    def memory(self) -> MemoryCapabilities:
        return self._memory

    def determine_weight_residency(
        self,
        weight_size_bytes: int,
        layer_type: str,
        memory_tier: Optional[str] = None,
        uses_mamba_scratch: bool = False,
    ) -> str:
        return gap9_determine_weight_residency(
            weight_size_bytes=weight_size_bytes,
            layer_type=layer_type,
            memory_tier=memory_tier,
            uses_mamba_scratch=uses_mamba_scratch,
        )

    def supports_ne16_linear(self) -> bool:
        return True

    def supports_ne16_conv2d_kernel(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        groups: int,
        in_channels: int,
        use_hwc_layout: bool,
        memory_tier: Optional[str] = None,
    ) -> bool:
        if memory_tier == "L3_TILED":
            return False
        if stride != (1, 1):
            return False

        kh, kw = kernel_size
        if (kh, kw) == (1, 1):
            return groups == 1

        if (kh, kw) != (3, 3):
            return False

        is_depthwise = (groups > 1) and (groups == in_channels)
        if groups > 1 and not is_depthwise:
            return False

        if not use_hwc_layout:
            return False

        return True
