# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Convolution layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict

from ..gap9_model import (
    WEIGHT_RESIDENCY_L3_STAGED,
    calculate_ne16_depthwise_tile_size,
)


def handle_quantconv2d(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle QuantConv2d layer.

    Handles 2D convolution with INT8 weights, including ResNet shortcut
    connections and L3 tiling for large activations.
    """
    if len(ctx.current_shape) != 4:
        raise ValueError(f"Conv layer {layer_name} expects 4D input.")

    # ResNet block input tracking
    if '.conv1' in layer_name:
        block_name = layer_name.rsplit('.conv1', 1)[0]
        ctx.block_input_shape[block_name] = ctx.current_shape
        ctx.block_input_scale[block_name] = ctx.current_scale
        ctx.block_input_buffer[block_name] = ctx.current_buffer

    # Handle shortcut connections
    if '.shortcut' in layer_name:
        block_name = layer_name.rsplit('.shortcut', 1)[0]
        if block_name in ctx.block_input_shape:
            input_shape = ctx.block_input_shape[block_name]
            in_ch, in_h, in_w = input_shape[1], input_shape[2], input_shape[3]
            input_buf = ctx.block_input_buffer[block_name]
            input_scale = ctx.block_input_scale.get(block_name, ctx.current_scale)
        else:
            in_ch, in_h, in_w = ctx.current_shape[1], ctx.current_shape[2], ctx.current_shape[3]
            input_buf = ctx.current_buffer
            input_scale = ctx.current_scale
    else:
        in_ch, in_h, in_w = ctx.current_shape[1], ctx.current_shape[2], ctx.current_shape[3]
        input_buf = ctx.current_buffer
        input_scale = ctx.current_scale

    # Conv parameters
    kernel_h, kernel_w = generator._as_pair(layer_data.get('kernel_size', 3))
    stride_h, stride_w = generator._as_pair(layer_data.get('stride', 1))
    pad_h, pad_w = generator._as_pair(layer_data.get('padding', 0))
    out_ch = layer_data['out_channels']
    groups = layer_data.get('groups', 1)  # 1=standard, in_ch=depthwise
    out_h = generator._compute_output_dim(in_h, kernel_h, stride_h, pad_h)
    out_w = generator._compute_output_dim(in_w, kernel_w, stride_w, pad_w)
    out_shape = [1, out_ch, out_h, out_w]

    buffer_name = f"{layer_name}_out"
    output_numel = generator._numel(out_shape)
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    scale_output = layer_data.get('scale_output')
    if scale_output is None:
        scale_output = generator._find_next_scale(idx, ctx.current_scale)

    # Query KB before tiling
    generator._prepare_kb_config(
        layer_name=layer_name,
        op_type='conv2d_int8',
        shape={
            'in_h': in_h, 'in_w': in_w,
            'in_channels': in_ch, 'out_channels': out_ch,
            'kernel_h': kernel_h, 'kernel_w': kernel_w,
            'stride_h': stride_h, 'stride_w': stride_w,
        }
    )

    # Determine memory tier (pass groups for depthwise weight size calculation)
    memory_tier, tile_config = generator._determine_conv2d_memory_tier(
        layer_name=layer_name,
        in_h=in_h, in_w=in_w, in_ch=in_ch, out_ch=out_ch,
        kernel_h=kernel_h, kernel_w=kernel_w,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        groups=groups
    )

    final_input_c_name = generator._ctx_buffer_c_name(ctx, input_buf)
    final_output_c_name = output_entry['c_name']

    # Handle L3 tiling buffers
    if memory_tier == 'L3_TILED':
        output_entry['use_l3_fallback'] = True

        slab_out_h = tile_config.l3_tile_h
        slab_out_size = slab_out_h * out_w * out_ch
        slab_out_entry = generator._ctx_register_buffer(
            ctx, f"{layer_name}_out_slab", 'int8_t', slab_out_size,
            "L2 Output Slab", None
        )
        spec['output_slab_buffer'] = slab_out_entry['c_name']
        final_output_c_name = slab_out_entry['c_name']

        input_buf_obj = ctx.buffer_map.get(input_buf)
        if input_buf_obj and input_buf_obj.get('use_l3_fallback', False):
            slab_in_h = tile_config.l3_tile_h_halo
            slab_in_size = slab_in_h * in_w * in_ch
            slab_in_entry = generator._ctx_register_buffer(
                ctx, f"{layer_name}_in_slab", 'int8_t', slab_in_size,
                "L2 Input Slab", None
            )
            spec['input_slab_buffer'] = slab_in_entry['c_name']
            final_input_c_name = slab_in_entry['c_name']

    # Check if this conv should use NE16 accelerator.
    # Kernel availability and layout constraints are target-specific.
    weight_size_bytes = out_ch * (in_ch // groups) * kernel_h * kernel_w
    default_weight_residency = generator._determine_weight_residency(
        weight_size_bytes=weight_size_bytes,
        layer_type='conv2d',
        memory_tier=memory_tier,
    )
    weight_residency = layer_data.get('weight_residency', default_weight_residency)

    use_ne16 = (
        generator.use_ne16
        and layer_name in generator.ne16_eligible_layers
        and generator.target.supports_ne16_conv2d_kernel(
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            groups=groups,
            in_channels=in_ch,
            use_hwc_layout=generator.use_hwc_layout,
            memory_tier=memory_tier,
        )
    )

    # Check for depthwise convolution (groups was extracted earlier)
    is_depthwise = (groups > 1) and (groups == in_ch)

    if use_ne16:
        if kernel_h == 1 and kernel_w == 1:
            op_type = 'conv2d_1x1_ne16'
        elif is_depthwise:  # Depthwise 3x3
            op_type = 'conv2d_3x3_dw_ne16'
        else:  # Regular 3x3
            op_type = 'conv2d_3x3_ne16'
    else:
        op_type = 'conv2d'

    spec.update({
        'op': op_type,
        'input_buffer': final_input_c_name,
        'output_buffer': final_output_c_name,
        'in_h': in_h, 'in_w': in_w, 'in_ch': in_ch,
        'out_h': out_h, 'out_w': out_w, 'out_ch': out_ch,
        'groups': groups,  # 1=standard, in_ch=depthwise
        'kernel_h': kernel_h, 'kernel_w': kernel_w,
        'stride_h': stride_h, 'stride_w': stride_w,
        'pad_h': pad_h, 'pad_w': pad_w,
        'scale_input': input_scale,
        'scale_weight': layer_data['scale_weight'],
        'scale_output': scale_output,
        # For depthwise (groups=in_ch), each channel has (in_ch/groups)=1 input
        'weight_elements': out_ch * (in_ch // groups) * kernel_h * kernel_w,
        'weight_size_bytes': out_ch * (in_ch // groups) * kernel_h * kernel_w,
        'bias_type': 'int32',
        'bias_elements': out_ch,
        'weight_index': generator.weight_entries[layer_name]['index'],
        'bias_index': generator.bias_entries[layer_name]['index'],
        'weight_residency': weight_residency,
        'activation_residency': layer_data.get('activation_residency', 'L2'),
        'memory_tier': memory_tier,
    })

    # Add NE16-specific fields for conv (1x1 or 3x3)
    if use_ne16:
        spec['ne16_eligible'] = True
        spec['ne16_packed_weight_index'] = generator.ne16_weight_entries[layer_name]['index']
        spec['ne16_bias_corr_index'] = generator.ne16_bias_entries[layer_name]['index']
        if kernel_h == 1 and kernel_w == 1:
            # For 1x1 conv, weight elements = out_ch * ceil(in_ch/16) * 16
            nb_ki = (in_ch + 15) // 16
            spec['ne16_packed_weight_elements'] = out_ch * nb_ki * 16
        elif is_depthwise:
            # For depthwise 3x3, packed_size = ceil(channels/32) * 8 * 3 * 3 * 4
            # Each Ko group of 32 channels has 8 bits * 9 spatial * 4 bytes = 288 bytes
            nb_ko = (out_ch + 31) // 32
            spec['ne16_packed_weight_elements'] = nb_ko * 8 * 3 * 3 * 4  # 288 per Ko group

            # Calculate spatial tiling for large depthwise layers
            dw_tile_config = calculate_ne16_depthwise_tile_size(
                in_h, in_w, in_ch, pad_h, pad_w
            )
            if dw_tile_config and dw_tile_config.spatial_tiling_enabled:
                spec['ne16_dw_spatial_tiling'] = 1
                spec['ne16_dw_num_tiles'] = dw_tile_config.num_tiles
                spec['ne16_dw_tile_h_out'] = dw_tile_config.tile_h_out
                spec['ne16_dw_tile_h_in'] = dw_tile_config.tile_h_in
                print(f"  -> Layer '{layer_name}' uses NE16 DW 3x3 with spatial tiling: "
                      f"{dw_tile_config.num_tiles} tiles, tile_h_out={dw_tile_config.tile_h_out}")
            else:
                spec['ne16_dw_spatial_tiling'] = 0
                spec['ne16_dw_num_tiles'] = 1
                spec['ne16_dw_tile_h_out'] = in_h
                spec['ne16_dw_tile_h_in'] = in_h + 2 * pad_h
        else:
            # For regular 3x3 conv, weight elements = out_ch * ceil(in_ch/16) * 16 * 9 spatial
            nb_ki = (in_ch + 15) // 16
            spec['ne16_packed_weight_elements'] = out_ch * nb_ki * 16 * 9
        spec['ne16_bias_corr_elements'] = out_ch
        # HW outquant scale indices (if available - for 1x1 or depthwise)
        if layer_name in generator.ne16_scale_entries:
            spec['ne16_hw_scale_index'] = generator.ne16_scale_entries[layer_name]['index']
            spec['ne16_hw_scale_shift_index'] = generator.ne16_scale_shift_entries[layer_name]['index']
            spec['ne16_use_hw_requant'] = True
        elif is_depthwise and hasattr(generator, 'ne16_hw_scale_entries') and layer_name in generator.ne16_hw_scale_entries:
            # Depthwise HW requant uses separate entries
            spec['ne16_hw_scale_index'] = generator.ne16_hw_scale_entries[layer_name]['index']
            spec['ne16_hw_scale_shift_index'] = generator.ne16_hw_scale_shift_entries[layer_name]['index']
            spec['ne16_use_hw_requant'] = True
        else:
            spec['ne16_use_hw_requant'] = False
        # L3 streaming for NE16: load packed weights from L3 to shared slab
        is_streamed = (weight_residency == WEIGHT_RESIDENCY_L3_STAGED)
        spec['ne16_with_streaming'] = is_streamed
        if is_streamed:
            print(f"  -> Layer '{layer_name}' uses NE16 {kernel_h}x{kernel_w} with L3 streaming (packed_size={spec['ne16_packed_weight_elements']} bytes)")

    if tile_config:
        spec['tile_config'] = tile_config.to_dict()

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.param_layers.append(spec)

    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True


def handle_conv1d_depthwise(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> bool:
    """Handle Conv1D Depthwise layer (MAMBA-style).

    Input: [batch, channels, length], Output: same shape.
    Uses depthwise convolution with causal padding.
    """
    if len(ctx.current_shape) != 3:
        raise ValueError(f"Conv1dDepthwise {layer_name} expects 3D input [B, C, L], got {ctx.current_shape}")

    batch, channels, length = ctx.current_shape
    kernel_size = layer_data.get('kernel_size', 4)
    causal = layer_data.get('causal', True)

    out_shape = [batch, channels, length]
    output_numel = generator._numel(out_shape)

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    weight_entry = generator.weight_entries.get(layer_name)
    bias_entry = generator.bias_entries.get(layer_name)
    weight_index = weight_entry['index'] if weight_entry else None
    bias_index = bias_entry['index'] if bias_entry else None

    spec.update({
        'op': 'conv1d_depthwise',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': generator._ctx_buffer_c_name(ctx, buffer_name),
        'channels': channels,
        'length': length,
        'kernel_size': kernel_size,
        'causal': 1 if causal else 0,
        'scale_input': layer_data.get('scale_input', ctx.current_scale),
        'scale_weight': layer_data.get('scale_weight', 1.0),
        'scale_output': layer_data.get('scale_output', ctx.current_scale),
        'weight_elements': channels * kernel_size,
        'weight_index': weight_index,
        'bias_index': bias_index,
        'bias_type': 'int32',
        'bias_elements': channels,
    })

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    if weight_index is not None:
        ctx.param_layers.append(spec)

    ctx.current_buffer = buffer_name
    ctx.current_shape = out_shape
    ctx.current_scale = layer_data.get('scale_output', ctx.current_scale)
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
