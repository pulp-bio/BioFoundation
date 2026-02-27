# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Linear layer handlers extracted from generate_c_code."""

from __future__ import annotations

from typing import Any, Dict

from ..gap9_model import (
    WEIGHT_RESIDENCY_L3_STAGED,
    WEIGHT_RESIDENCY_L3_TILED,
)


def handle_quantlinear(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle QuantLinear layer.

    Handles linear layers with INT8 or FP32 output, including L3 tiling
    with double-buffered slabs for large weight matrices.
    """
    # SwiGLU parallel branch detection: W1 and W3 should share the same input
    # When we see 'w3' in a SwiGLU FFN, restore the input buffer from 'w1'
    if '.w3' in layer_name and '.ffn.' in layer_name:
        w1_name = layer_name.replace('.w3', '.w1')
        if w1_name in ctx.layer_input_buffer:
            # Restore the same input buffer and scale that W1 used
            shared_input_buffer = ctx.layer_input_buffer[w1_name]
            shared_input_scale = ctx.layer_input_scale.get(w1_name, ctx.current_scale)
            ctx.current_buffer = shared_input_buffer
            ctx.current_scale = shared_input_scale
            print(f"  [SwiGLU] {layer_name}: Sharing input buffer with {w1_name} ({ctx.current_buffer})")

    # Record this layer's input buffer and scale for potential parallel branches
    ctx.layer_input_buffer[layer_name] = ctx.current_buffer
    ctx.layer_input_scale[layer_name] = ctx.current_scale

    in_features = layer_data['in_features']
    out_features = layer_data['out_features']
    is_final = (layer_name == generator.final_linear_name)

    # Detect 3D inputs
    is_3d_input = False
    seq_len = 1
    current_shape = list(ctx.current_shape)
    if len(current_shape) == 3 and current_shape[-1] == in_features:
        is_3d_input = True
        seq_len = current_shape[1]
    elif current_shape[-1] != in_features:
        current_shape = [1, in_features]

    batch_tokens = seq_len
    output_numel = batch_tokens * out_features
    scale_output = layer_data.get('scale_output')

    # Determine if this is a FP32 or INT8 classifier output
    use_fp32_output = is_final and not generator.int8_classifier_output

    if use_fp32_output:
        buffer_name = "output_fp32"
        output_entry = generator._ctx_register_buffer(
            ctx, buffer_name, 'float', output_numel,
            "Final FP32 logits", spec.get('block_id')
        )
        scale_output = None
    else:
        buffer_name = "output_int8" if is_final else f"{layer_name}_out"
        if scale_output is None:
            scale_output = generator._find_next_scale(idx, ctx.current_scale)
        desc = "Final INT8 output" if is_final else f"{layer_name} INT8 output"
        output_entry = generator._ctx_register_buffer(
            ctx, buffer_name, 'int8_t', output_numel, desc, spec.get('block_id')
        )

    memory_tier = 'L2_FULL'
    tile_config = None

    generator._prepare_kb_config(
        layer_name=layer_name,
        op_type='linear_fp32' if use_fp32_output else 'linear_int8',
        shape={'M': batch_tokens, 'N': out_features, 'K': in_features}
    )

    if (not use_fp32_output) or (not is_3d_input):
        memory_tier, tile_config = generator._determine_linear_memory_tier(
            layer_name, in_features, out_features, batch_tokens, is_final=use_fp32_output
        )

    final_output_c_name = output_entry['c_name']
    weight_size_bytes = out_features * in_features
    final_weight_residency = layer_data.get(
        'weight_residency',
        generator._determine_weight_residency(
            weight_size_bytes=weight_size_bytes,
            layer_type='linear',
            memory_tier=memory_tier,
        ),
    )

    is_streamed = False
    output_uses_l3_fallback = output_entry.get('use_l3_fallback', False)
    uses_shared_slab = False

    if memory_tier == 'L3_TILED':
        final_weight_residency = WEIGHT_RESIDENCY_L3_TILED
        is_streamed = True

        slab_out_features = tile_config.l3_tile_out_features
        slab_out_size = batch_tokens * slab_out_features
        if use_fp32_output:
            slab_out_size *= 4

        slab_out_entry = generator._ctx_register_buffer(
            ctx, f"{layer_name}_out_slab",
            'float' if use_fp32_output else 'int8_t',
            slab_out_size * 2, "L2 Output Slab (Double)",
            None, l2_required=True
        )
        spec['output_slab_buffer'] = slab_out_entry['c_name']
        final_output_c_name = slab_out_entry['c_name']

        slab_weight_size = slab_out_features * in_features
        weight_slab_entry = generator._ctx_register_buffer(
            ctx, f"{layer_name}_weight_slab", 'int8_t',
            slab_weight_size * 2, "L2 Weight Slab (Double)",
            None, l2_required=True
        )
        spec['weight_slab_buffer'] = weight_slab_entry['c_name']

        slab_bias_size = slab_out_features
        bias_slab_entry = generator._ctx_register_buffer(
            ctx, f"{layer_name}_bias_slab",
            'float' if use_fp32_output else 'int32_t',
            slab_bias_size * 2, "L2 Bias Slab (Double)",
            None, l2_required=True
        )
        spec['bias_slab_buffer'] = bias_slab_entry['c_name']
        spec['l3_slab_out_stride'] = slab_out_features

    # Check if layer has bias (Llama SwiGLU layers typically don't have bias)
    has_bias = layer_name in generator.bias_entries

    # Check if this large linear layer should use the shared block_weight_slab
    # to avoid L2 exhaustion. This applies to non-block layers with weights
    # above the target's large weight threshold.
    large_weight_threshold = generator.target.large_weight_threshold_bytes
    if (weight_size_bytes >= large_weight_threshold and
        memory_tier != 'L3_TILED' and
        not use_fp32_output):
        # Mark this layer to use the shared block_weight_slab
        uses_shared_slab = True
        is_streamed = True
        final_weight_residency = WEIGHT_RESIDENCY_L3_STAGED
        print(f"  -> Layer '{layer_name}' uses shared block_weight_slab (weight={weight_size_bytes} bytes)")

    # Determine if this layer should use NE16 accelerator
    # NE16 now supports L3 streaming via weight tiling
    use_ne16 = (generator.use_ne16 and
                layer_name in generator.ne16_eligible_layers and
                generator.target.supports_ne16_linear() and
                not use_fp32_output)

    # For NE16 with streaming, we need to generate packed weights for L3
    # Block layers (with block_id) also use the shared slab, so they need NE16 streaming
    is_block_layer = spec.get('block_id') is not None
    ne16_with_streaming = use_ne16 and (is_streamed or uses_shared_slab or is_block_layer)

    if use_ne16:
        op_type = 'linear_ne16'
    elif use_fp32_output:
        op_type = 'linear_fp32'
    else:
        op_type = 'linear_int8'

    spec.update({
        'op': op_type,
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': final_output_c_name,
        'in_features': in_features,
        'out_features': out_features,
        'batch_tokens': batch_tokens,
        'scale_input': ctx.current_scale,
        'scale_weight': layer_data['scale_weight'],
        'scale_output': scale_output,
        'weight_elements': out_features * in_features,
        'weight_size_bytes': weight_size_bytes,
        'bias_type': 'fp32' if use_fp32_output else 'int32',
        'bias_elements': out_features if has_bias else 0,
        'has_bias': has_bias,
        'weight_index': generator.weight_entries[layer_name]['index'],
        'bias_index': generator.bias_entries[layer_name]['index'] if has_bias else None,
        'weight_residency': final_weight_residency,
        'memory_tier': memory_tier,
        'is_streamed': is_streamed,
        'uses_shared_slab': uses_shared_slab,
        'output_uses_l3_fallback': output_uses_l3_fallback,
        'is_classifier': is_final,
        'int8_classifier_output': is_final and generator.int8_classifier_output,
    })

    # Add NE16-specific fields
    if use_ne16:
        spec['ne16_eligible'] = True
        spec['ne16_packed_weight_index'] = generator.ne16_weight_entries[layer_name]['index']
        spec['ne16_bias_corr_index'] = generator.ne16_bias_entries[layer_name]['index']
        # Calculate packed weight size (padded to 16-byte Ki groups)
        nb_ki = (in_features + 15) // 16
        packed_weight_size = out_features * nb_ki * 16
        spec['ne16_packed_weight_elements'] = packed_weight_size
        spec['ne16_bias_corr_elements'] = out_features
        # Default tile size - can be tuned based on L1 budget
        spec['ne16_tile_tokens'] = min(64, batch_tokens)
        # Track if NE16 uses streaming (weights from L3 via shared slab)
        spec['ne16_with_streaming'] = ne16_with_streaming
        # HW outquant scale indices (if available)
        if layer_name in generator.ne16_scale_entries:
            spec['ne16_hw_scale_index'] = generator.ne16_scale_entries[layer_name]['index']
            spec['ne16_hw_scale_shift_index'] = generator.ne16_scale_shift_entries[layer_name]['index']
            spec['ne16_use_hw_requant'] = True
        else:
            spec['ne16_use_hw_requant'] = False
        if ne16_with_streaming:
            print(f"  -> Layer '{layer_name}' uses NE16 with L3 streaming (packed_size={packed_weight_size} bytes)")

    if tile_config:
        spec['tile_config'] = tile_config.to_dict()

    if not is_final:
        generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)

    ctx.specs.append(spec)
    ctx.param_layers.append(spec)

    ctx.current_buffer = buffer_name
    if is_3d_input:
        ctx.current_shape = [1, seq_len, out_features]
    else:
        ctx.current_shape = [1, out_features]
    ctx.current_scale = scale_output if scale_output else 1.0
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale

    if is_final:
        generator.num_classes = out_features

    return True
