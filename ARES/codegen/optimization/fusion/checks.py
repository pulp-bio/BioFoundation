# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for fusion scaffolding contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .registry import DEFAULT_PATTERN_ORDER, FusionRegistry, build_default_registry

FusionSpec = Dict[str, Any]

EXPECTED_DEFAULT_PATTERN_ORDER = DEFAULT_PATTERN_ORDER
EXPECTED_FUSION_TYPES = frozenset(
    {
        "conv_relu",
        "linear_relu",
        "pool_quant",
        "conv_relu_quant",
        "linear_relu_quant",
        "conv_relu_maxpool",
    }
)


def validate_default_registry(registry: Optional[FusionRegistry] = None) -> List[str]:
    """Validate registry ordering and minimum-layer constraints."""
    active_registry = registry if registry is not None else build_default_registry()
    errors: List[str] = []

    names = [pattern.name for pattern in active_registry.patterns()]
    if tuple(names) != EXPECTED_DEFAULT_PATTERN_ORDER:
        errors.append(
            "Unexpected default registry order: "
            f"{names} != {list(EXPECTED_DEFAULT_PATTERN_ORDER)}"
        )

    for pattern in active_registry.patterns():
        if pattern.min_layers <= 0:
            errors.append(f"Pattern {pattern.name} has invalid min_layers={pattern.min_layers}")

    return errors


def validate_fusion_payload(fusion: FusionSpec) -> List[str]:
    """Validate shape and required keys for a fusion descriptor payload."""
    errors: List[str] = []
    fusion_type = fusion.get("type")
    layers = fusion.get("layers")
    base_layer = fusion.get("base_layer")

    if fusion_type not in EXPECTED_FUSION_TYPES:
        errors.append(f"Unsupported fusion type: {fusion_type}")

    if not isinstance(layers, list) or not layers:
        errors.append(f"Invalid layers field: {layers}")
    else:
        if not all(isinstance(layer_idx, int) for layer_idx in layers):
            errors.append(f"Non-integer layer index in {layers}")
        if layers != sorted(layers):
            errors.append(f"Layer indices not sorted: {layers}")

    if not isinstance(base_layer, int):
        errors.append(f"Invalid base_layer field: {base_layer}")
    elif isinstance(layers, list) and layers and base_layer != layers[0]:
        errors.append(f"base_layer mismatch: {base_layer} != {layers[0]}")

    if fusion_type == "conv_relu_maxpool" and "pool_spec" not in fusion:
        errors.append("conv_relu_maxpool fusion is missing pool_spec")

    return errors


def validate_fusions(fusions: Sequence[FusionSpec]) -> List[str]:
    """Validate all fusion payloads and return indexed error messages."""
    errors: List[str] = []
    for idx, fusion in enumerate(fusions):
        errors.extend(f"[fusion {idx}] {error}" for error in validate_fusion_payload(fusion))
    return errors

