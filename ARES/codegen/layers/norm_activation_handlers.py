# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Normalization and activation layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..gap9_model import WEIGHT_RESIDENCY_L2, WEIGHT_RESIDENCY_MAMBA_SCRATCH


def handle_groupnorm(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle GroupNorm layer."""
    if len(ctx.current_shape) < 2:
        raise ValueError(f"GroupNorm {layer_name} expects at least 2D [B, C, ...], got {ctx.current_shape}")

    batch = int(ctx.current_shape[0])
    channels = int(ctx.current_shape[1])
    spatial_size = int(np.prod(ctx.current_shape[2:])) if len(ctx.current_shape) > 2 else 1

    num_groups = int(layer_data.get("num_groups", 1))
    if num_groups <= 0:
        raise ValueError(f"GroupNorm {layer_name}: num_groups must be > 0, got {num_groups}")
    if channels % num_groups != 0:
        raise ValueError(f"GroupNorm {layer_name}: channels={channels} not divisible by num_groups={num_groups}")

    total_elements = generator._numel(ctx.current_shape)
    eps = float(layer_data.get("eps", 1e-5))

    scale_output = generator._find_next_scale(idx, ctx.current_scale)

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, "int8_t", total_elements, f"{layer_name} output", spec.get("block_id")
    )

    weight_entry = generator.weight_entries.get(layer_name)
    bias_entry = generator.bias_entries.get(layer_name)
    if weight_entry is None or bias_entry is None:
        raise ValueError(
            f"GroupNorm {layer_name}: missing FP32 weight/bias binaries. "
            f"Expected {layer_name}_weight_fp32.npy and {layer_name}_bias_fp32.npy in weights dir."
        )

    spec.update(
        {
            "op": "groupnorm",
            "input_buffer": generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
            "output_buffer": output_entry["c_name"],
            "batch": batch,
            "channels": channels,
            "spatial_size": spatial_size,
            "num_groups": num_groups,
            "eps": eps,
            "scale_input": ctx.current_scale,
            "scale_output": scale_output,
            "weight_type": "fp32",
            "bias_type": "fp32",
            "weight_elements": channels,
            "bias_elements": channels,
            "weight_index": weight_entry["index"],
            "bias_index": bias_entry["index"],
            "weight_residency": WEIGHT_RESIDENCY_L2,
            "memory_tier": "L2_FULL",
        }
    )

    generator._ctx_attach_golden(ctx, spec, layer_name, total_elements, buffer_name)
    ctx.specs.append(spec)
    ctx.param_layers.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_gelu(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle GELU activation layer (in-place)."""
    num_elements = generator._numel(ctx.current_shape)
    scale_output = generator._find_next_scale(idx, ctx.current_scale)

    spec.update(
        {
            "op": "gelu",
            "buffer": generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
            "num_elements": num_elements,
            "scale_input": ctx.current_scale,
            "scale_output": scale_output,
        }
    )

    memory_tier, tile_config = generator._determine_elementwise_memory_tier(
        layer_name=layer_name,
        num_elements=num_elements,
        in_place=True,
    )
    spec["memory_tier"] = memory_tier
    if tile_config:
        spec["tile_config"] = tile_config.to_dict()

    generator._query_kb_for_layer(
        layer_name=layer_name,
        op_type="gelu_int8",
        shape={"size": num_elements},
    )

    generator._ctx_attach_golden(ctx, spec, layer_name, num_elements, ctx.current_buffer)
    ctx.specs.append(spec)
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_layernorm(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
    has_mamba_layers: bool = False,
) -> bool:
    """Handle LayerNorm layer."""
    normalized_shape = layer_data.get("normalized_shape")
    if isinstance(normalized_shape, (tuple, list)):
        if len(normalized_shape) == 1:
            normalized_dim = normalized_shape[0]
        else:
            normalized_dim = int(np.prod(normalized_shape))
    else:
        normalized_dim = int(normalized_shape)

    total_elements = generator._numel(ctx.current_shape)
    eps = layer_data.get("eps", 1e-5)

    weight_param = None
    bias_param = None
    weight_index = None
    bias_index = None
    if layer_name in generator.weight_entries:
        weight_param = generator.weight_entries[layer_name]["c_symbol"]
        weight_index = generator.weight_entries[layer_name]["index"]
    if layer_name in generator.bias_entries:
        bias_param = generator.bias_entries[layer_name]["c_symbol"]
        bias_index = generator.bias_entries[layer_name]["index"]

    scale_output = generator._find_next_scale(idx, ctx.current_scale)

    buffer_name = f"{layer_name}_out"
    output_numel = total_elements
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, "int8_t", output_numel, f"{layer_name} output", spec.get("block_id")
    )

    uses_mamba_scratch = has_mamba_layers
    num_tokens = max(1, total_elements // max(1, normalized_dim))

    spec.update(
        {
            "op": "layernorm",
            "input_buffer": generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
            "output_buffer": output_entry["c_name"],
            "weight_param": weight_param,
            "bias_param": bias_param,
            "weight_type": "fp32",
            "bias_type": "fp32",
            "weight_index": weight_index,
            "bias_index": bias_index,
            "weight_elements": normalized_dim,
            "bias_elements": normalized_dim,
            "total_elements": total_elements,
            "normalized_dim": normalized_dim,
            "num_tokens": num_tokens,
            "embed_dim": normalized_dim,
            "scale_input": ctx.current_scale,
            "scale_output": scale_output,
            "eps": eps,
            "uses_mamba_scratch": uses_mamba_scratch,
            "weight_residency": WEIGHT_RESIDENCY_MAMBA_SCRATCH if uses_mamba_scratch else WEIGHT_RESIDENCY_L2,
        }
    )

    generator._prepare_kb_config(
        layer_name=layer_name,
        op_type="layernorm_int8",
        shape={
            "num_tokens": num_tokens,
            "embed_dim": normalized_dim,
        },
    )

    memory_tier, tile_config = generator._determine_layernorm_memory_tier(
        layer_name=layer_name,
        num_tokens=num_tokens,
        normalized_dim=normalized_dim,
    )
    spec["memory_tier"] = memory_tier
    if tile_config:
        spec["tile_config"] = tile_config.to_dict()

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.param_layers.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_rmsnorm(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
    has_mamba_layers: bool = False,
) -> bool:
    """Handle RMSNorm layer."""
    normalized_shape = layer_data.get("normalized_shape")
    if isinstance(normalized_shape, (tuple, list)):
        if len(normalized_shape) == 1:
            normalized_dim = normalized_shape[0]
        else:
            normalized_dim = int(np.prod(normalized_shape))
    else:
        normalized_dim = int(normalized_shape)

    total_elements = generator._numel(ctx.current_shape)
    eps = layer_data.get("eps", 1e-5)

    weight_param = None
    weight_index = None
    if layer_name in generator.weight_entries:
        weight_param = generator.weight_entries[layer_name]["c_symbol"]
        weight_index = generator.weight_entries[layer_name]["index"]

    scale_output = generator._find_next_scale(idx, ctx.current_scale)

    buffer_name = f"{layer_name}_out"
    output_numel = total_elements
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, "int8_t", output_numel, f"{layer_name} output", spec.get("block_id")
    )

    uses_mamba_scratch = has_mamba_layers
    num_tokens = max(1, total_elements // max(1, normalized_dim))

    spec.update(
        {
            "op": "rmsnorm",
            "input_buffer": generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
            "output_buffer": output_entry["c_name"],
            "weight_param": weight_param,
            "weight_type": "fp32",
            "weight_index": weight_index,
            "weight_elements": normalized_dim,
            "total_elements": total_elements,
            "normalized_dim": normalized_dim,
            "num_tokens": num_tokens,
            "embed_dim": normalized_dim,
            "scale_input": ctx.current_scale,
            "scale_output": scale_output,
            "eps": eps,
            "uses_mamba_scratch": uses_mamba_scratch,
            "weight_residency": WEIGHT_RESIDENCY_MAMBA_SCRATCH if uses_mamba_scratch else WEIGHT_RESIDENCY_L2,
        }
    )

    generator._prepare_kb_config(
        layer_name=layer_name,
        op_type="rmsnorm_int8",
        shape={
            "num_tokens": num_tokens,
            "embed_dim": normalized_dim,
        },
    )

    memory_tier, tile_config = generator._determine_layernorm_memory_tier(
        layer_name=layer_name,
        num_tokens=num_tokens,
        normalized_dim=normalized_dim,
    )
    spec["memory_tier"] = memory_tier
    if tile_config:
        spec["tile_config"] = tile_config.to_dict()

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.param_layers.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
