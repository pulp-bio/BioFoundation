# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Signal processing layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict


def handle_rfft(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle fixed-point RFFT producing concatenated [magnitude, phase].

    Currently supports patch_size=40.
    """
    if len(ctx.current_shape) < 1:
        raise ValueError(f"RFFT {layer_name}: missing input shape metadata")

    patch_size = int(layer_data.get('patch_size', 40))
    if ctx.current_shape[-1] != patch_size:
        raise ValueError(f"RFFT {layer_name}: expected last dim={patch_size}, got {ctx.current_shape[-1]}")
    if patch_size != 40:
        raise ValueError(f"RFFT {layer_name}: only patch_size=40 supported, got {patch_size}")

    num_bins = patch_size // 2 + 1
    out_features = 2 * num_bins  # [mag, phase]
    out_shape = list(ctx.current_shape[:-1]) + [out_features]
    output_numel = generator._numel(out_shape)

    num_patches = int(output_numel // out_features)

    scale_output = generator._find_next_scale(idx, ctx.current_scale)

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    spec.update({
        'op': 'rfft',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'num_patches': num_patches,
        'patch_size': patch_size,
        'num_bins': num_bins,
        'out_features': out_features,
        'scale_input': ctx.current_scale,
        'scale_output': scale_output,
        'memory_tier': 'L2_FULL',
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)

    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
