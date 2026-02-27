# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Reshape-family layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict


def handle_squeeze(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle Squeeze (dimension removal) layer - no-op, just updates shape."""
    dim = layer_data.get('dim', -1)

    if dim == -1:
        output_shape = [d for d in ctx.current_shape if d != 1]
    else:
        output_shape = [d for i, d in enumerate(ctx.current_shape) if i != dim or d != 1]

    if len(output_shape) == 0:
        output_shape = [1]

    output_numel = generator._numel(output_shape)

    spec.update({
        'op': 'squeeze',
        'buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'input_shape': list(ctx.current_shape),
        'output_shape': output_shape,
        'dim': dim,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, ctx.current_buffer)
    ctx.specs.append(spec)
    ctx.current_shape = output_shape
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_flatten(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle Flatten layer - no-op, just reshapes."""
    output_shape = layer_data.get('output_shape')
    if output_shape is None:
        flatten_dim = generator._numel(ctx.current_shape)
        output_shape = [1, flatten_dim]
    else:
        flatten_dim = generator._numel(output_shape)

    spec.update({
        'op': 'flatten',
        'buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'input_shape': list(ctx.current_shape),
        'output_shape': output_shape,
        'output_dim': flatten_dim,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, flatten_dim, ctx.current_buffer)
    ctx.specs.append(spec)
    ctx.current_shape = output_shape
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_reshape(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle Reshape layer - view operation with shape change."""
    captured = layer_data.get('output_shape')
    if captured is not None:
        new_shape = list(captured)
    else:
        target_shape = layer_data.get('shape', [])
        new_shape = [int(ctx.current_shape[0])] + list(target_shape)

    input_numel = generator._numel(ctx.current_shape)
    output_numel = generator._numel(new_shape)
    if input_numel != output_numel:
        print(f"  [WARN]  WARNING: Reshape {layer_name} element count mismatch: "
              f"{input_numel} -> {output_numel}")

    spec.update({
        'op': 'reshape',
        'buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'input_shape': list(ctx.current_shape),
        'output_shape': new_shape,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, ctx.current_buffer)
    ctx.specs.append(spec)
    ctx.current_shape = new_shape
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_permute(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle Permute (transpose) layer."""
    perm = layer_data.get('dims', [0, 2, 1])
    old_shape = list(ctx.current_shape)
    new_shape = [old_shape[i] for i in perm]

    is_simple_transpose = (perm == [0, 2, 1] and len(old_shape) == 3)

    buffer_name = f"{layer_name}_out"
    output_numel = generator._numel(new_shape)
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    memory_tier, tile_config = generator._determine_transpose2d_memory_tier(
        layer_name, old_shape[1], old_shape[2]
    )

    spec.update({
        'op': 'transpose_2d',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': generator._ctx_buffer_c_name(ctx, buffer_name),
        'input_shape': old_shape,
        'output_shape': new_shape,
        'dims': perm,
        'batch_size': old_shape[0],
        'dim1': old_shape[1],
        'dim2': old_shape[2],
        'in_place': is_simple_transpose,
        'scale': ctx.current_scale,
        'memory_tier': memory_tier,
        'tile_config': tile_config,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_shape = new_shape
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
