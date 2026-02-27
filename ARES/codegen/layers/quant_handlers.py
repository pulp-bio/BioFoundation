# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Quantization layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict

from ..constants import SCALE_EPSILON


def handle_quant_identity(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle QuantIdentity (requantize) layer."""
    # Multi-input branch handling: detect branch entry points
    # and reset shape/buffer to the branch's input
    is_branch_entry = False
    if 'input_quant' in layer_name and layer_name != 'input_quant':
        output_shape = layer_data.get('output_shape')
        if output_shape:
            ctx.current_shape = output_shape
            is_branch_entry = True
            print(f"  [Multi-input] Branch entry '{layer_name}': shape reset to {output_shape}")

            if hasattr(generator, 'branch_input_buffers') and layer_name in generator.branch_input_buffers:
                branch_buffer = generator.branch_input_buffers[layer_name]
                ctx.current_buffer = branch_buffer
                if layer_name in ctx.layer_output_scale:
                    ctx.current_scale = ctx.layer_output_scale[layer_name]
                print(f"  [Multi-input] Branch entry '{layer_name}': buffer switched to '{branch_buffer}'")

    scale_out = layer_data.get('scale', ctx.current_scale)
    num_elements = generator._numel(ctx.current_shape)
    skip_requant = abs(ctx.current_scale - scale_out) <= SCALE_EPSILON

    spec.update({
        'op': 'requantize',
        'buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'numel': num_elements,
        'scale_in': ctx.current_scale,
        'scale_out': scale_out,
        'skip_requant': skip_requant,
        'is_branch_entry': is_branch_entry,
    })

    memory_tier, tile_config = generator._determine_elementwise_memory_tier(
        layer_name=layer_name,
        num_elements=num_elements,
        in_place=True
    )
    spec['memory_tier'] = memory_tier
    if tile_config:
        spec['tile_config'] = tile_config.to_dict()

    generator._ctx_attach_golden(ctx, spec, layer_name, num_elements, ctx.current_buffer)
    ctx.specs.append(spec)
    ctx.current_scale = scale_out
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale

    return True


def handle_quant_relu(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle QuantReLU layer."""
    relu_scale = layer_data.get('scale', ctx.current_scale)
    needs_requant = abs(relu_scale - ctx.current_scale) > SCALE_EPSILON
    num_elements = generator._numel(ctx.current_shape)

    spec.update({
        'op': 'relu',
        'buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'numel': num_elements,
        'scale': relu_scale,
        'scale_in': ctx.current_scale,
        'scale_out': relu_scale,
        'needs_requant': needs_requant,
    })

    memory_tier, tile_config = generator._determine_elementwise_memory_tier(
        layer_name=layer_name,
        num_elements=num_elements,
        in_place=True
    )
    spec['memory_tier'] = memory_tier
    if tile_config:
        spec['tile_config'] = tile_config.to_dict()

    generator._ctx_attach_golden(ctx, spec, layer_name, num_elements, ctx.current_buffer)
    ctx.specs.append(spec)
    ctx.current_scale = relu_scale
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
