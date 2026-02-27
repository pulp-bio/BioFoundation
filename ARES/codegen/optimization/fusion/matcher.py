# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Fusion opportunity matcher scaffolding."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .registry import FusionRegistry

LayerSpec = Dict[str, Any]
FusionSpec = Dict[str, Any]


def match_conv_relu_quant(specs: Sequence[LayerSpec], index: int) -> Optional[FusionSpec]:
    """Match Conv2D -> ReLU -> Requantize."""
    if index + 2 >= len(specs):
        return None

    curr = specs[index]
    next1 = specs[index + 1]
    next2 = specs[index + 2]

    if (
        curr.get("op") == "conv2d"
        and next1.get("op") == "relu"
        and next2.get("op") == "requantize"
        and next1.get("buffer") == curr.get("output_buffer")
        and next2.get("buffer") == curr.get("output_buffer")
    ):
        return {
            "type": "conv_relu_quant",
            "layers": [index, index + 1, index + 2],
            "base_layer": index,
        }

    return None


def match_linear_relu_quant(specs: Sequence[LayerSpec], index: int) -> Optional[FusionSpec]:
    """Match Linear -> ReLU -> Requantize."""
    if index + 2 >= len(specs):
        return None

    curr = specs[index]
    next1 = specs[index + 1]
    next2 = specs[index + 2]

    if (
        curr.get("op") == "linear_int8"
        and next1.get("op") == "relu"
        and next2.get("op") == "requantize"
        and next1.get("buffer") == curr.get("output_buffer")
        and next2.get("buffer") == curr.get("output_buffer")
    ):
        return {
            "type": "linear_relu_quant",
            "layers": [index, index + 1, index + 2],
            "base_layer": index,
        }

    return None


def match_conv_relu_maxpool_or_fallback(specs: Sequence[LayerSpec], index: int) -> Optional[FusionSpec]:
    """
    Match Conv2D -> ReLU -> MaxPool with single-tile constraint.

    If Conv is multi-tile, fall back to Conv2D -> ReLU only.
    """
    if index + 2 >= len(specs):
        return None

    curr = specs[index]
    next1 = specs[index + 1]
    next2 = specs[index + 2]

    if not (
        curr.get("op") == "conv2d"
        and next1.get("op") == "relu"
        and next2.get("op") == "maxpool"
        and next1.get("buffer") == curr.get("output_buffer")
        and next2.get("input_buffer") == curr.get("output_buffer")
    ):
        return None

    tile_config = curr.get("tile_config", {})
    num_tiles = tile_config.get("num_tiles", 0)

    if num_tiles == 1:
        return {
            "type": "conv_relu_maxpool",
            "layers": [index, index + 1, index + 2],
            "base_layer": index,
            "pool_spec": next2,
        }

    return {"type": "conv_relu", "layers": [index, index + 1], "base_layer": index}


def match_conv_relu(specs: Sequence[LayerSpec], index: int) -> Optional[FusionSpec]:
    """Match Conv2D -> ReLU."""
    if index + 1 >= len(specs):
        return None

    curr = specs[index]
    next1 = specs[index + 1]

    if (
        curr.get("op") == "conv2d"
        and next1.get("op") == "relu"
        and next1.get("buffer") == curr.get("output_buffer")
    ):
        return {"type": "conv_relu", "layers": [index, index + 1], "base_layer": index}

    return None


def match_linear_relu(specs: Sequence[LayerSpec], index: int) -> Optional[FusionSpec]:
    """Match Linear -> ReLU."""
    if index + 1 >= len(specs):
        return None

    curr = specs[index]
    next1 = specs[index + 1]

    if (
        curr.get("op") == "linear_int8"
        and next1.get("op") == "relu"
        and next1.get("buffer") == curr.get("output_buffer")
    ):
        return {"type": "linear_relu", "layers": [index, index + 1], "base_layer": index}

    return None


def match_pool_quant(specs: Sequence[LayerSpec], index: int) -> Optional[FusionSpec]:
    """Match Pool -> Requantize."""
    if index + 1 >= len(specs):
        return None

    curr = specs[index]
    next1 = specs[index + 1]

    if (
        curr.get("op") in ["maxpool", "avgpool", "global_avgpool"]
        and next1.get("op") == "requantize"
        and next1.get("buffer") == curr.get("output_buffer")
    ):
        return {"type": "pool_quant", "layers": [index, index + 1], "base_layer": index}

    return None


def detect_fusion_opportunities(
    specs: Sequence[LayerSpec],
    registry: Optional[FusionRegistry] = None,
) -> List[FusionSpec]:
    """Detect fusible layer subsequences with registry-defined precedence."""
    if registry is None:
        from .registry import build_default_registry

        registry = build_default_registry()

    fusions: List[FusionSpec] = []
    index = 0
    patterns = registry.patterns()

    while index < len(specs):
        matched: Optional[FusionSpec] = None

        for pattern in patterns:
            if index + pattern.min_layers > len(specs):
                continue
            matched = pattern.match_fn(specs, index)
            if matched is not None:
                break

        if matched is None:
            index += 1
            continue

        fusions.append(matched)
        index += max(1, len(matched.get("layers", [])))

    return fusions

