# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Pool layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict


def handle_maxpool2d(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle MaxPool2d layer."""
    kernel_h, kernel_w = generator._as_pair(layer_data.get('kernel_size', 2))
    stride_h, stride_w = generator._as_pair(layer_data.get('stride', 2))
    pad_h, pad_w = generator._as_pair(layer_data.get('padding', 0))
    in_ch, in_h, in_w = ctx.current_shape[1], ctx.current_shape[2], ctx.current_shape[3]
    out_h = generator._compute_output_dim(in_h, kernel_h, stride_h, pad_h)
    out_w = generator._compute_output_dim(in_w, kernel_w, stride_w, pad_w)
    out_shape = [1, in_ch, out_h, out_w]

    buffer_name = f"{layer_name}_out"
    output_numel = generator._numel(out_shape)
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    spec.update({
        'op': 'maxpool',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'in_h': in_h,
        'in_w': in_w,
        'channels': in_ch,
        'out_h': out_h,
        'out_w': out_w,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
        'stride_h': stride_h,
        'stride_w': stride_w,
        'pad_h': pad_h,
        'pad_w': pad_w,
    })

    memory_tier, tile_config = generator._determine_maxpool_memory_tier(
        layer_name=layer_name, in_h=in_h, in_w=in_w, channels=in_ch,
        kernel_size=kernel_h, stride=stride_h, padding=pad_h
    )
    spec['memory_tier'] = memory_tier
    if tile_config:
        spec['tile_config'] = tile_config.to_dict()

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_avgpool2d(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle AvgPool2d layer."""
    kernel_h, kernel_w = generator._as_pair(layer_data.get('kernel_size', 2))
    stride_h, stride_w = generator._as_pair(layer_data.get('stride', 2))
    pad_h, pad_w = generator._as_pair(layer_data.get('padding', 0))
    in_ch, in_h, in_w = ctx.current_shape[1], ctx.current_shape[2], ctx.current_shape[3]
    out_h = generator._compute_output_dim(in_h, kernel_h, stride_h, pad_h)
    out_w = generator._compute_output_dim(in_w, kernel_w, stride_w, pad_w)
    out_shape = [1, in_ch, out_h, out_w]

    buffer_name = f"{layer_name}_out"
    output_numel = generator._numel(out_shape)
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    scale_output = layer_data.get('scale_output')
    if scale_output is None:
        scale_output = generator._find_next_scale(idx, ctx.current_scale)

    # If output is 1x1 (global pooling), use global_avgpool op which handles
    # non-square kernels correctly. This handles AdaptiveAvgPool2d((1,1)) cases.
    if out_h == 1 and out_w == 1:
        memory_tier, tile_config = generator._determine_globalavgpool_memory_tier(
            layer_name=layer_name, in_h=in_h, in_w=in_w, channels=in_ch
        )
        spec.update({
            'op': 'global_avgpool',
            'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
            'output_buffer': output_entry['c_name'],
            'batch': 1,
            'channels': in_ch,
            'height': in_h,
            'width': in_w,
            'scale_input': ctx.current_scale,
            'scale_output': scale_output,
            'memory_tier': memory_tier,
        })
        if tile_config is not None:
            spec['tile_config'] = tile_config.to_dict()
    else:
        memory_tier, tile_config = generator._determine_avgpool_memory_tier(
            layer_name=layer_name, in_h=in_h, in_w=in_w, channels=in_ch,
            kernel_size=kernel_h, stride=stride_h, padding=pad_h
        )
        spec.update({
            'op': 'avgpool',
            'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
            'output_buffer': output_entry['c_name'],
            'in_h': in_h,
            'in_w': in_w,
            'channels': in_ch,
            'out_h': out_h,
            'out_w': out_w,
            'kernel_h': kernel_h,
            'kernel_w': kernel_w,
            'stride_h': stride_h,
            'stride_w': stride_w,
            'scale_input': ctx.current_scale,
            'scale_output': scale_output,
            'memory_tier': memory_tier,
        })
        if tile_config is not None:
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


def handle_globalavgpool(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle GlobalAvgPool layer."""
    in_ch, in_h, in_w = ctx.current_shape[1], ctx.current_shape[2], ctx.current_shape[3]
    out_shape = [1, in_ch, 1, 1]

    buffer_name = f"{layer_name}_out"
    output_numel = generator._numel(out_shape)
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    scale_output = layer_data.get('scale_output')
    if scale_output is None:
        scale_output = generator._find_next_scale(idx, ctx.current_scale)

    memory_tier, tile_config = generator._determine_globalavgpool_memory_tier(
        layer_name=layer_name, in_h=in_h, in_w=in_w, channels=in_ch
    )

    spec.update({
        'op': 'global_avgpool',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'batch': 1,
        'channels': in_ch,
        'height': in_h,
        'width': in_w,
        'scale_input': ctx.current_scale,
        'scale_output': scale_output,
        'memory_tier': memory_tier,
    })
    if tile_config is not None:
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


def handle_adaptive_avgpool1d(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle AdaptiveAvgPool1d layer.

    Pools the last dimension to output_size (usually 1). Handles implicit
    transpose when detected from expected output shape.
    """
    output_size = layer_data.get('output_size', 1)

    if len(ctx.current_shape) != 3:
        raise ValueError(f"AdaptiveAvgPool1d expects 3D input, got {ctx.current_shape}")

    batch, dim1, dim2 = ctx.current_shape
    current_buffer = ctx.current_buffer
    current_shape = list(ctx.current_shape)

    expected_output_shape = layer_data.get('output_shape', None)
    needs_transpose = False

    if expected_output_shape and len(expected_output_shape) == 3:
        if expected_output_shape[1] != dim1 and expected_output_shape[1] == dim2:
            needs_transpose = True
            channels = dim2
            seq_len = dim1
            output_shape = expected_output_shape
        else:
            channels = dim1
            seq_len = dim2
            output_shape = [batch, channels, output_size]
    else:
        channels = dim1
        seq_len = dim2
        output_shape = [batch, channels, output_size]

    # Insert transpose if needed
    if needs_transpose:
        transpose_name = f"{layer_name}_transpose"
        transposed_shape = [batch, dim2, dim1]
        transpose_numel = dim2 * dim1 * batch

        transpose_buffer_name = f"{transpose_name}_out"
        transpose_entry = generator._ctx_register_buffer(
            ctx, transpose_buffer_name, 'int8_t', transpose_numel,
            f"{transpose_name} output", spec.get('block_id')
        )

        memory_tier, tile_config = generator._determine_transpose2d_memory_tier(
            transpose_name, dim1, dim2
        )

        transpose_spec = {
            'c_name': transpose_name,
            'name': transpose_name,
            'op': 'transpose_2d',
            'input_buffer': generator._ctx_buffer_c_name(ctx, current_buffer),
            'output_buffer': transpose_entry['c_name'],
            'input_shape': current_shape,
            'output_shape': transposed_shape,
            'dims': [0, 2, 1],
            'batch_size': batch,
            'dim1': dim1,
            'dim2': dim2,
            'in_place': False,
            'scale': ctx.current_scale,
            'memory_tier': memory_tier,
            'tile_config': tile_config,
        }
        ctx.specs.append(transpose_spec)

        current_buffer = transpose_buffer_name
        current_shape = transposed_shape
        ctx.layer_output_buffer[transpose_name] = current_buffer
        ctx.layer_output_scale[transpose_name] = ctx.current_scale
        ctx.buffer_scale[current_buffer] = ctx.current_scale

    output_numel = generator._numel(output_shape)
    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    spec.update({
        'op': 'adaptive_avgpool1d',
        'input_buffer': generator._ctx_buffer_c_name(ctx, current_buffer),
        'output_buffer': output_entry['c_name'],
        'input_shape': current_shape,
        'output_shape': output_shape,
        'batch': batch,
        'channels': channels,
        'input_len': seq_len,
        'output_size': output_size,
        'scale': ctx.current_scale,
        'needs_transpose': False,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_shape = output_shape
    ctx.current_buffer = buffer_name
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
