# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Elementwise layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def handle_zeropad2d(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle ZeroPad2d layer - pads input tensor with zeros."""
    # Get padding: (left, right, top, bottom)
    padding = layer_data.get('padding', (0, 0, 0, 0))
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])

    pad_left, pad_right, pad_top, pad_bottom = padding

    # Handle 4D input [B, C, H, W]
    if len(ctx.current_shape) == 4:
        batch, in_ch, in_h, in_w = ctx.current_shape
    else:
        # Handle 3D input [C, H, W]
        batch = 1
        in_ch, in_h, in_w = ctx.current_shape

    out_h = in_h + pad_top + pad_bottom
    out_w = in_w + pad_left + pad_right
    out_shape = [batch, in_ch, out_h, out_w] if len(ctx.current_shape) == 4 else [in_ch, out_h, out_w]

    buffer_name = f"{layer_name}_out"
    output_numel = generator._numel(out_shape)
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    spec.update({
        'op': 'zeropad2d',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'channels': in_ch,
        'in_h': in_h,
        'in_w': in_w,
        'out_h': out_h,
        'out_w': out_w,
        'pad_left': pad_left,
        'pad_right': pad_right,
        'pad_top': pad_top,
        'pad_bottom': pad_bottom,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_add(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle Add (element-wise addition) layer.

    Resolves two input tensors by name, computes output scale, and generates
    element-wise add spec with proper scale handling.
    """
    # Use layer's output_shape from metadata instead of current_shape
    # This is critical for ResNet-style networks where current_shape may be stale
    output_shape = layer_data.get('output_shape', ctx.current_shape)
    numel = generator._numel(output_shape)
    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', numel,
        f"{layer_name} output", spec.get('block_id')
    )

    scale_output = layer_data.get('scale_output')
    if scale_output is None:
        scale_output = generator._find_next_scale(idx, ctx.current_scale)

    input_names = layer_data.get('inputs', [])
    if len(input_names) < 2:
        raise ValueError(f"Add layer '{layer_name}' missing inputs metadata")

    # Resolve two unique input buffers
    resolved_inputs = []
    seen_buffers = set()
    for candidate in input_names:
        buf_name = ctx.layer_output_buffer.get(candidate)
        if not buf_name:
            continue
        if buf_name in seen_buffers:
            continue
        resolved_inputs.append(candidate)
        seen_buffers.add(buf_name)
        if len(resolved_inputs) == 2:
            break

    # Fallback: scan prior layers in reverse order for compatible shapes.
    if len(resolved_inputs) < 2:
        target_shape = generator.layer_info.get(layer_name, {}).get('output_shape') or ctx.current_shape
        for prev_name in reversed(generator.layer_order[:idx]):
            buf_name = ctx.layer_output_buffer.get(prev_name)
            if not buf_name or buf_name in seen_buffers:
                continue
            prev_shape = generator.layer_info.get(prev_name, {}).get('output_shape')
            if prev_shape and target_shape and len(prev_shape) == len(target_shape):
                if prev_shape[1:] != target_shape[1:]:
                    continue
            resolved_inputs.append(prev_name)
            seen_buffers.add(buf_name)
            if len(resolved_inputs) == 2:
                break

    if len(resolved_inputs) < 2:
        missing = ', '.join(input_names)
        raise ValueError(
            f"Add layer '{layer_name}' could not resolve two unique inputs "
            f"(candidates: {missing})"
        )

    in1, in2 = resolved_inputs
    buf1 = generator._ctx_buffer_c_name(ctx, ctx.layer_output_buffer[in1])
    buf2 = generator._ctx_buffer_c_name(ctx, ctx.layer_output_buffer[in2])
    scale1 = ctx.buffer_scale.get(buf1, ctx.layer_output_scale[in1])
    scale2 = ctx.buffer_scale.get(buf2, ctx.layer_output_scale[in2])

    spec.update({
        'op': 'add',
        'input1_buffer': buf1,
        'input2_buffer': buf2,
        'output_buffer': output_entry['c_name'],
        'size': numel,
        'scale_x1': scale1,
        'scale_x2': scale2,
        'scale_output': scale_output,
    })

    # Determine memory tier for L1 tiling
    memory_tier, tile_config = generator._determine_add_memory_tier(
        layer_name=layer_name,
        num_elements=numel
    )
    spec['memory_tier'] = memory_tier
    if tile_config:
        spec['tile_config'] = tile_config.to_dict()

    generator._ctx_attach_golden(ctx, spec, layer_name, numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_scale = scale_output
    ctx.current_shape = output_shape
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_concatenate(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle Concatenate (channel concatenation) layer.

    Concatenates multiple input tensors along the channel dimension,
    handling multi-input branch merges for dual-input networks.
    """
    # Channel concatenation
    input_names = layer_data.get('inputs', [])

    # Multi-input branch handling: detect branch merge points
    # For dual-input networks, find the final layers of each branch
    if len(generator.input_quant_layers) > 1 and (not input_names or len(input_names) < 2):
        # This is likely a multi-input branch merge
        # Find the last layer before this concat for each branch
        branch_outputs = generator._find_branch_outputs_before_concat(idx)
        if len(branch_outputs) >= 2:
            input_names = branch_outputs
            print(f"  [Multi-input] Concat '{layer_name}' merging branches: {input_names}")

    in_h, in_w = ctx.current_shape[2], ctx.current_shape[3]
    buffer_name = f"{layer_name}_out"
    scale_output = layer_data.get('scale_output')
    if scale_output is None:
        scale_output = generator._find_next_scale(idx, ctx.current_scale)

    # Get input buffers and scales (computed once)
    input_buffers = []
    input_scales = []
    for name in input_names:
        buf = generator._ctx_buffer_c_name(ctx, ctx.layer_output_buffer[name])
        input_buffers.append(buf)
        input_scales.append(ctx.buffer_scale.get(buf, ctx.layer_output_scale[name]))

    # Get channels per input
    # For multi-input branch merges, always recalculate from actual input layers
    is_multi_input_merge = len(generator.input_quant_layers) > 1 and len(input_names) >= 2
    channels_per_input = layer_data.get('channels_per_input')
    if not channels_per_input or is_multi_input_merge:
        channels_per_input = [
            generator.layer_info[name]['output_shape'][1] for name in input_names
        ]

    total_channels = sum(channels_per_input)
    out_shape = [1, total_channels, in_h, in_w]
    output_numel = generator._numel(out_shape)
    spatial_size = in_h * in_w

    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    spec.update({
        'op': 'concat',
        'input_buffers': input_buffers,
        'input_scales': input_scales,
        'output_buffer': output_entry['c_name'],
        'num_inputs': len(input_names),
        'channels_per_input': channels_per_input,
        'total_channels': total_channels,
        'height': in_h,
        'width': in_w,
        'spatial_size': spatial_size,
        'scale_output': scale_output,
    })

    # Determine memory tier for L1 tiling
    memory_tier, tile_config = generator._determine_concat_memory_tier(
        layer_name=layer_name,
        num_inputs=len(input_names),
        total_channels=total_channels,
        spatial_size=spatial_size
    )
    spec['memory_tier'] = memory_tier
    if tile_config:
        spec['tile_config'] = tile_config.to_dict()

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_mean(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle Mean (mean pooling) layer.

    Computes mean over a specified dimension, typically the sequence dimension
    in transformers (dim=1). Reduces [B, seq_len, features] to [B, features].
    """
    dim = layer_data.get('dim', 1)
    keepdim = layer_data.get('keepdim', False)
    scale_input = layer_data.get('scale_input', ctx.current_scale)
    scale_output = layer_data.get('scale_output', ctx.current_scale)

    # Validate shape: expect at least [B, seq_len, features]
    if len(ctx.current_shape) < 3:
        raise ValueError(
            f"Mean {layer_name}: expected 3D input [B, seq_len, features], "
            f"got shape {ctx.current_shape}"
        )

    batch = int(ctx.current_shape[0])
    seq_len = int(ctx.current_shape[dim])
    # Features are the last dimension for dim=1
    features = int(ctx.current_shape[-1]) if dim != len(ctx.current_shape) - 1 else int(ctx.current_shape[1])

    # Compute output shape
    if keepdim:
        output_shape = list(ctx.current_shape)
        output_shape[dim] = 1
    else:
        output_shape = [d for i, d in enumerate(ctx.current_shape) if i != dim]

    output_numel = int(np.prod(output_shape))
    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    spec.update({
        'op': 'mean',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'batch': batch,
        'seq_len': seq_len,
        'features': features,
        'dim': dim,
        'keepdim': 1 if keepdim else 0,
        'scale_input': scale_input,
        'scale_output': scale_output,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_shape = output_shape
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
