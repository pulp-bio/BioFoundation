# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""Attention layer handlers extracted from generate_c_code."""

from __future__ import annotations

import math
from typing import Any, Dict

from ..gap9_model import WEIGHT_RESIDENCY_L2


def handle_alternating_attention(
    generator: Any,
    ctx: Any,
    layer_name: str,
    layer_data: Dict[str, Any],
    spec: Dict[str, Any],
    idx: int,
) -> bool:
    """Handle AlternatingAttention layer (Cerebro transformer).

    Alternating pattern based on block_idx:
    - Even blocks: Channel attention (attend over channels for each time step)
    - Odd blocks: Temporal attention (attend over time for each channel)
    """
    embed_dim = layer_data.get('embed_dim')
    num_heads = layer_data.get('num_heads', 1)
    head_dim = layer_data.get('head_dim') or (embed_dim // num_heads if embed_dim and num_heads else None)
    num_channels = layer_data.get('num_channels')
    temporal_len = layer_data.get('temporal_len')
    block_idx = layer_data.get('block_idx', 0)
    scaling_factor = layer_data.get('scaling_factor') or (1.0 / math.sqrt(head_dim) if head_dim else 1.0)

    if embed_dim is None or num_channels is None or temporal_len is None:
        raise ValueError(
            f"AlternatingAttention {layer_name}: missing required params "
            f"(embed_dim={embed_dim}, num_channels={num_channels}, temporal_len={temporal_len})"
        )

    seq_len = num_channels * temporal_len
    scale_input = layer_data.get('scale_input', ctx.current_scale)
    scale_output = layer_data.get('scale_output', ctx.current_scale)

    # Output shape is same as input: [B, seq_len, embed_dim]
    output_shape = [1, seq_len, embed_dim]
    output_numel = seq_len * embed_dim

    buffer_name = f"{layer_name}_out"
    output_entry = generator._ctx_register_buffer(
        ctx, buffer_name, 'int8_t', output_numel,
        f"{layer_name} output", spec.get('block_id')
    )

    # Register weight parameters
    def register_attn_param(prefix: str):
        weight_key = f"{layer_name}::{prefix}"
        weight_entry = generator.weight_entries.get(weight_key)
        bias_entry = generator.bias_entries.get(weight_key)
        if weight_entry is None:
            raise ValueError(f"Missing {prefix} weights for {layer_name}")

        if prefix == 'qkv':
            in_features = layer_data.get('qkv_in_features', embed_dim)
            out_features = layer_data.get('qkv_out_features', 3 * embed_dim)
        else:  # 'out'
            in_features = layer_data.get('out_in_features', embed_dim)
            out_features = layer_data.get('out_out_features', embed_dim)

        weight_size_bytes = in_features * out_features
        weight_residency = WEIGHT_RESIDENCY_L2  # For simplicity, use L2
        param = {
            'name': f"{layer_name}_{prefix}",
            'c_name': generator._unique_layer_c_name(f"{layer_name}_{prefix}"),
            'weight_elements': in_features * out_features,
            'bias_type': 'int32',
            'bias_elements': out_features,
            'weight_index': weight_entry['index'],
            'bias_index': bias_entry['index'] if bias_entry else None,
            'weight_residency': weight_residency,
        }
        ctx.param_layers.append(param)
        return param['c_name']

    qkv_c_name = register_attn_param('qkv')
    out_c_name = register_attn_param('out')

    # Get scales
    qkv_scale_weight = layer_data.get('qkv_scale_weight', 1.0)
    qkv_scale_output = layer_data.get('qkv_scale_output', 1.0)
    q_scale_output = layer_data.get('q_scale_output', qkv_scale_output)
    k_scale_output = layer_data.get('k_scale_output', qkv_scale_output)
    v_scale_output = layer_data.get('v_scale_output', qkv_scale_output)
    out_scale_weight = layer_data.get('out_scale_weight', 1.0)

    # Check for NE16 packed weights
    qkv_ne16_key = f"{layer_name}::qkv"
    out_ne16_key = f"{layer_name}::out"
    use_ne16_qkv = qkv_ne16_key in generator.ne16_weight_entries
    use_ne16_out = out_ne16_key in generator.ne16_weight_entries

    spec.update({
        'op': 'alternating_attention',
        'input_buffer': generator._ctx_buffer_c_name(ctx, ctx.current_buffer),
        'output_buffer': output_entry['c_name'],
        'batch': 1,
        'seq_len': seq_len,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'num_channels': num_channels,
        'temporal_len': temporal_len,
        'block_idx': block_idx,
        'scaling_factor': scaling_factor,
        'scale_input': scale_input,
        'qkv_scale_weight': qkv_scale_weight,
        'qkv_scale_output': qkv_scale_output,
        'scale_q': q_scale_output,
        'scale_k': k_scale_output,
        'scale_v': v_scale_output,
        'out_scale_weight': out_scale_weight,
        'scale_output': scale_output,
        'qkv_param': qkv_c_name,
        'out_param': out_c_name,
        # NE16 support for attention projections
        'use_ne16_qkv': use_ne16_qkv,
        'use_ne16_out': use_ne16_out,
        'qkv_ne16_packed_index': generator.ne16_weight_entries.get(qkv_ne16_key, {}).get('index', -1) if use_ne16_qkv else -1,
        'qkv_ne16_bias_index': generator.ne16_bias_entries.get(qkv_ne16_key, {}).get('index', -1) if use_ne16_qkv else -1,
        'qkv_ne16_scale_index': generator.ne16_scale_entries.get(qkv_ne16_key, {}).get('index', -1) if use_ne16_qkv else -1,
        'qkv_ne16_scale_shift_index': generator.ne16_scale_shift_entries.get(qkv_ne16_key, {}).get('index', -1) if use_ne16_qkv else -1,
        'out_ne16_packed_index': generator.ne16_weight_entries.get(out_ne16_key, {}).get('index', -1) if use_ne16_out else -1,
        'out_ne16_bias_index': generator.ne16_bias_entries.get(out_ne16_key, {}).get('index', -1) if use_ne16_out else -1,
        'out_ne16_scale_index': generator.ne16_scale_entries.get(out_ne16_key, {}).get('index', -1) if use_ne16_out else -1,
        'out_ne16_scale_shift_index': generator.ne16_scale_shift_entries.get(out_ne16_key, {}).get('index', -1) if use_ne16_out else -1,
    })

    attn_type = "channel" if block_idx % 2 == 0 else "temporal"
    ne16_status = ""
    if use_ne16_qkv or use_ne16_out:
        ne16_status = f" [NE16: qkv={'Y' if use_ne16_qkv else 'N'}, out={'Y' if use_ne16_out else 'N'}]"
        # Create an entry for template L3 handle setup
        alt_attn_entry = {
            'c_name': generator.sanitize_c_name(layer_name),
            'layer_name': layer_name,
            'embed_dim': embed_dim,
            'use_ne16_qkv': use_ne16_qkv,
            'use_ne16_out': use_ne16_out,
            'qkv_ne16_packed_index': spec.get('qkv_ne16_packed_index', -1),
            'qkv_ne16_bias_index': spec.get('qkv_ne16_bias_index', -1),
            'out_ne16_packed_index': spec.get('out_ne16_packed_index', -1),
            'out_ne16_bias_index': spec.get('out_ne16_bias_index', -1),
        }
        generator.alt_attn_ne16_entries.append(alt_attn_entry)
    print(f"  [ALT_ATTN] {layer_name}: block={block_idx} ({attn_type}), "
          f"heads={num_heads}, seq={seq_len}, embed={embed_dim}{ne16_status}")

    generator._ctx_attach_golden(ctx, spec, layer_name, output_numel, buffer_name)
    ctx.specs.append(spec)
    ctx.current_buffer = buffer_name
    ctx.current_shape = output_shape
    ctx.current_scale = scale_output
    ctx.layer_output_buffer[layer_name] = ctx.current_buffer
    ctx.layer_output_scale[layer_name] = ctx.current_scale
    ctx.buffer_scale[ctx.current_buffer] = ctx.current_scale
    return True
