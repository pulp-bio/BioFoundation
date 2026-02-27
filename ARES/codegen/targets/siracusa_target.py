# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Siracusa target implementation for target-aware codegen."""

from __future__ import annotations

from typing import Optional, Tuple

from ..siracusa_model import SiracusaHardwareModel
from ..siracusa_model import determine_weight_residency as siracusa_determine_weight_residency
from .target_base import MemoryCapabilities, TargetBase


class SiracusaTarget(TargetBase):
    """Siracusa target (PULP SDK) used by codegen planning."""

    name = "siracusa"
    display_name = "Siracusa"

    def __init__(self) -> None:
        self._memory = MemoryCapabilities(
            l1_bytes=SiracusaHardwareModel.get_l1_budget(),
            l2_bytes=SiracusaHardwareModel.get_l2_budget(),
            l3_bytes=None,
            l2_tiling_bytes=SiracusaHardwareModel.get_l2_tiling_budget(),
            l3_stage_threshold_bytes=SiracusaHardwareModel.get_l3_stage_threshold(),
            l1_total_bytes=SiracusaHardwareModel.get_l1_total_bytes(),
            l2_total_bytes=SiracusaHardwareModel.get_l2_total_bytes(),
            l2_activation_reserved_bytes=SiracusaHardwareModel.get_l2_activation_reserved_bytes(),
            l3_fallback_single_buffer_threshold_bytes=(
                SiracusaHardwareModel.get_l3_fallback_single_buffer_threshold_bytes()
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
        return siracusa_determine_weight_residency(
            weight_size_bytes=weight_size_bytes,
            layer_type=layer_type,
            memory_tier=memory_tier,
            uses_mamba_scratch=uses_mamba_scratch,
        )

    def supports_ne16_linear(self) -> bool:
        # Keep disabled until Siracusa NE16 path is validated end-to-end.
        return False

    def supports_ne16_conv2d_kernel(
        self,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        groups: int,
        in_channels: int,
        use_hwc_layout: bool,
        memory_tier: Optional[str] = None,
    ) -> bool:
        del kernel_size, stride, groups, in_channels, use_hwc_layout, memory_tier
        return False
