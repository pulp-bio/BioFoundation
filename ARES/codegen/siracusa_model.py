# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Siracusa hardware model and planning heuristics.

This mirrors the GAP9 model structure while providing Siracusa-specific
memory budgets and residency thresholds.
"""

from __future__ import annotations

from .gap9_model import (
    WEIGHT_RESIDENCY_L2,
    WEIGHT_RESIDENCY_L3_STAGED,
    WEIGHT_RESIDENCY_L3_TILED,
    WEIGHT_RESIDENCY_MAMBA_SCRATCH,
)


class SiracusaHardwareModel:
    """
    Siracusa memory budget constants used by code generation.

    Hardware capacities:
    - L1/TCDM: 256 KiB
    - L2: 2 MiB

    The tiling budgets below remain conservative to leave headroom for runtime
    stacks, metadata, and temporary staging buffers.
    """

    # Hardware totals as seen by allocators/linker scripts.
    L1_TOTAL_BYTES = (256 * 1024) - 4
    L2_TOTAL_BYTES = (2 * 1024 * 1024) - 4

    # Conservative usable budgets for planning.
    L1_SIZE_BYTES = 224 * 1024
    L2_SIZE_BYTES = 2 * 1024 * 1024

    # Reserved memory for code/stack/runtime overhead.
    L1_RESERVED = 8 * 1024
    L2_RESERVED = 128 * 1024

    # Dedicated tiling budget inside L2 after additional runtime overhead.
    L2_TILING_BUDGET = 1536 * 1024

    # Weight staging threshold for non-dynamic L3 tiling paths.
    L3_STAGE_THRESHOLD = 64 * 1024

    # L3 fallback sizing policy for oversized activation buffers.
    L2_ACTIVATION_RESERVED = 768 * 1024
    L3_FALLBACK_SINGLE_BUFFER_THRESHOLD = 900 * 1024

    @classmethod
    def get_l1_total_bytes(cls) -> int:
        return cls.L1_TOTAL_BYTES

    @classmethod
    def get_l2_total_bytes(cls) -> int:
        return cls.L2_TOTAL_BYTES

    @classmethod
    def get_l1_budget(cls) -> int:
        return cls.L1_SIZE_BYTES - cls.L1_RESERVED

    @classmethod
    def get_l2_budget(cls) -> int:
        return cls.L2_SIZE_BYTES - cls.L2_RESERVED

    @classmethod
    def get_l2_tiling_budget(cls) -> int:
        return cls.L2_TILING_BUDGET

    @classmethod
    def get_l3_stage_threshold(cls) -> int:
        return cls.L3_STAGE_THRESHOLD

    @classmethod
    def get_l2_activation_reserved_bytes(cls) -> int:
        return cls.L2_ACTIVATION_RESERVED

    @classmethod
    def get_l3_fallback_single_buffer_threshold_bytes(cls) -> int:
        return cls.L3_FALLBACK_SINGLE_BUFFER_THRESHOLD


def determine_weight_residency(
    weight_size_bytes: int,
    layer_type: str,
    memory_tier: str | None = None,
    uses_mamba_scratch: bool = False,
) -> str:
    """Siracusa weight residency policy."""
    if uses_mamba_scratch:
        return WEIGHT_RESIDENCY_MAMBA_SCRATCH

    if memory_tier == "L3_TILED":
        return WEIGHT_RESIDENCY_L3_TILED

    if layer_type in ("mhsa_projection", "mhsa"):
        if weight_size_bytes >= SiracusaHardwareModel.L3_STAGE_THRESHOLD:
            return WEIGHT_RESIDENCY_L3_STAGED

    return WEIGHT_RESIDENCY_L2
