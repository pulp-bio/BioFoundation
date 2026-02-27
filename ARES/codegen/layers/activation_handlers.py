# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Activation layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict


def handle_silu(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle SiLU activation layer (in-place with pre-extracted LUT)."""
    num_elements = generator._numel(ctx.current_shape)

    lut_entry = generator.weight_entries.get(f"{layer_name}::lut")
    lut_index = lut_entry['index'] if lut_entry else None

    scale_in = layer_data.get('scale_input', ctx.current_scale)
    scale_out = layer_data.get('scale_output', ctx.current_scale)

    spec.update({
        'op': 'silu',
        'buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'num_elements': num_elements,
        'scale_in': scale_in,
        'scale_out': scale_out,
        'weight_index': lut_index,
        'bias_index': None,
        'bias_type': 'int32',
        'bias_elements': 0,
        'weight_elements': 256,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, num_elements, ctx.current_buffer)
    ctx.specs.append(spec)

    if lut_index is not None:
        ctx.param_layers.append(spec)

    ctx.current_scale = scale_out
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
