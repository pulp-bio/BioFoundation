# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Classification Head with MLP — INT8 Reference Implementation

Matches QuantClassificationHeadWithMLP from brevitas_custom_layers.py:
1. Learned aggregation query (1 token) cross-attends to full sequence
2. MLP classifier: hidden_dim -> 4*hidden_dim -> num_classes
"""

from typing import Any, Dict, Tuple

import numpy as np

try:
    from .quantize import quantize_linear, dequantize_linear
    from .linear import linear_int8
    from .gelu import gelu_int8_lut
    from .mhsa import i_softmax_int32_to_uint8
except ImportError:
    from quantize import quantize_linear, dequantize_linear  # type: ignore
    from linear import linear_int8  # type: ignore
    from gelu import gelu_int8_lut  # type: ignore
    from mhsa import i_softmax_int32_to_uint8  # type: ignore


def _convert_bias_int32(bias_fp32, scale_input, scale_weight):
    """Convert FP32 bias to INT32 domain."""
    if bias_fp32 is None:
        return None
    return np.round(bias_fp32 / (scale_input * scale_weight)).astype(np.int32)


def classification_head_with_mlp_int8(
    x_int8: np.ndarray,
    layer_info: Dict[str, Any],
    scale_input: float,
    *,
    use_i_softmax: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    INT8 Classification Head with learned aggregation query and MLP.

    Args:
        x_int8: Input tensor [B, seq_len, hidden_dim] INT8.
        layer_info: Flat dict with weights, biases, and scales.
        scale_input: Quantization scale for x_int8.
        use_i_softmax: Use integer-only softmax.

    Returns:
        (logits_int8, logit_scale, logits_fp32) where shape is [B, num_classes].
    """
    hidden_dim = layer_info['hidden_dim']
    num_heads = layer_info['num_heads']
    head_dim = layer_info['head_dim']
    num_classes = layer_info['num_classes']
    softmax_scale = layer_info['softmax_scale']

    batch_size = x_int8.shape[0]
    seq_len = x_int8.shape[1]

    # --- Learned aggregation query ---
    agg_int8 = np.array(layer_info['learned_agg_int8'])
    if agg_int8.ndim == 2:
        # [1, hidden_dim] -> [B, 1, hidden_dim]
        agg_int8 = np.tile(agg_int8[np.newaxis] if agg_int8.ndim == 2 else agg_int8, (batch_size, 1, 1))
    elif agg_int8.ndim == 3 and agg_int8.shape[0] == 1:
        agg_int8 = np.tile(agg_int8, (batch_size, 1, 1))
    agg_scale = layer_info['agg_scale']

    # --- Q/K/V projections ---
    # Q from aggregation query [B, 1, hidden_dim]
    q = linear_int8(
        agg_int8.reshape(batch_size, hidden_dim),
        layer_info['q_weight_int8'],
        _convert_bias_int32(layer_info['q_bias_fp32'], agg_scale, layer_info['q_scale_weight']),
        agg_scale, layer_info['q_scale_weight'], layer_info['q_scale_output'],
    ).reshape(batch_size, 1, hidden_dim)

    # K/V from sequence [B, seq_len, hidden_dim]
    k = linear_int8(
        x_int8.reshape(batch_size * seq_len, hidden_dim),
        layer_info['k_weight_int8'],
        _convert_bias_int32(layer_info['k_bias_fp32'], scale_input, layer_info['k_scale_weight']),
        scale_input, layer_info['k_scale_weight'], layer_info['k_scale_output'],
    ).reshape(batch_size, seq_len, hidden_dim)

    v = linear_int8(
        x_int8.reshape(batch_size * seq_len, hidden_dim),
        layer_info['v_weight_int8'],
        _convert_bias_int32(layer_info['v_bias_fp32'], scale_input, layer_info['v_scale_weight']),
        scale_input, layer_info['v_scale_weight'], layer_info['v_scale_output'],
    ).reshape(batch_size, seq_len, hidden_dim)

    # --- Multi-head attention ---
    # Reshape to [B, H, SeqLen, Dh]
    qh = q.reshape(batch_size, 1, num_heads, head_dim).transpose(0, 2, 1, 3)
    kh = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    vh = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Scores: INT8 x INT8 -> INT32
    scores_int32 = np.matmul(qh.astype(np.int32), kh.astype(np.int32).transpose(0, 1, 3, 2))

    # Softmax
    if use_i_softmax:
        attn_uint8 = i_softmax_int32_to_uint8(
            scores_int32,
            scale_q=layer_info['q_scale_output'],
            scale_k=layer_info['k_scale_output'],
            softmax_scale=softmax_scale, axis=-1,
        )
    else:
        scores_fp32 = scores_int32.astype(np.float32) * layer_info['q_scale_output'] * layer_info['k_scale_output'] * softmax_scale
        scores_max = np.max(scores_fp32, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_fp32 - scores_max).astype(np.float32)
        attn_fp32 = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        attn_uint8 = np.clip(np.round(attn_fp32 * 255.0), 0, 255).astype(np.uint8)

    # Context: UINT8 x INT8 -> INT32 -> (acc+128)>>8 -> INT8
    context_int32 = np.matmul(attn_uint8.astype(np.int32), vh.astype(np.int32))
    context_int8 = np.clip((context_int32 + 128) >> 8, -128, 127).astype(np.int8)

    # Reshape: [B, H, 1, Dh] -> [B, hidden_dim]
    context_flat = context_int8.transpose(0, 2, 1, 3).reshape(batch_size, hidden_dim)
    scale_v = layer_info['v_scale_output']

    # --- Output projection ---
    out = linear_int8(
        context_flat, layer_info['out_weight_int8'],
        _convert_bias_int32(layer_info['out_bias_fp32'], scale_v, layer_info['out_scale_weight']),
        scale_v, layer_info['out_scale_weight'], layer_info['out_scale_output'],
    )
    scale_out = layer_info['out_scale_output']

    # --- MLP classifier ---
    mlp_h = linear_int8(
        out, layer_info['mlp_fc1_weight_int8'],
        _convert_bias_int32(layer_info['mlp_fc1_bias_fp32'], scale_out, layer_info['mlp_fc1_scale_weight']),
        scale_out, layer_info['mlp_fc1_scale_weight'], layer_info['mlp_gelu_scale'],
    )
    mlp_h = gelu_int8_lut(mlp_h, layer_info['mlp_gelu_scale'], layer_info['mlp_gelu_scale'])

    # Final classifier: output as FP32 logits
    # Use an INT8 intermediate scale, then dequantize
    mlp_fc2_scale = layer_info['mlp_fc2_scale_weight']
    logit_int8_scale = layer_info['mlp_gelu_scale'] * mlp_fc2_scale  # approximate
    logits_int8 = linear_int8(
        mlp_h, layer_info['mlp_fc2_weight_int8'],
        _convert_bias_int32(layer_info.get('mlp_fc2_bias_fp32'), layer_info['mlp_gelu_scale'], mlp_fc2_scale),
        layer_info['mlp_gelu_scale'], mlp_fc2_scale, logit_int8_scale,
    )
    logits_fp32 = dequantize_linear(logits_int8, scale=logit_int8_scale, zero_point=0)

    return logits_int8, logit_int8_scale, logits_fp32


def test_classification_head_with_mlp_int8():
    """Smoke test for classification head."""
    batch_size = 2
    seq_len = 8
    hidden_dim = 32
    num_heads = 4
    head_dim = hidden_dim // num_heads
    num_classes = 3

    np.random.seed(42)
    scale_input = 0.02
    x_fp32 = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32) * 0.1
    x_int8 = quantize_linear(x_fp32, scale=scale_input, zero_point=0)

    def rand_w(rows, cols):
        return quantize_linear(
            np.random.randn(rows, cols).astype(np.float32) * 0.1, scale=0.01, zero_point=0
        )

    def rand_bias(size):
        return np.random.randn(size).astype(np.float32) * 0.01

    layer_info = {
        'embed_dim': hidden_dim // 4,  # 8
        'num_queries': 4,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'num_classes': num_classes,
        'softmax_scale': 1.0 / np.sqrt(head_dim),
    }

    # Learned aggregation query
    layer_info['learned_agg_int8'] = quantize_linear(
        np.random.randn(1, hidden_dim).astype(np.float32) * 0.1, scale=0.03, zero_point=0
    )
    layer_info['agg_scale'] = 0.03

    # Attention projections
    for prefix in ('q', 'k', 'v', 'out'):
        layer_info[f'{prefix}_weight_int8'] = rand_w(hidden_dim, hidden_dim)
        layer_info[f'{prefix}_bias_fp32'] = rand_bias(hidden_dim)
        layer_info[f'{prefix}_scale_weight'] = 0.01
        layer_info[f'{prefix}_scale_output'] = 0.02

    # MLP
    layer_info['mlp_fc1_weight_int8'] = rand_w(4 * hidden_dim, hidden_dim)
    layer_info['mlp_fc1_bias_fp32'] = rand_bias(4 * hidden_dim)
    layer_info['mlp_fc1_scale_weight'] = 0.01
    layer_info['mlp_gelu_scale'] = 0.02
    layer_info['mlp_fc2_weight_int8'] = rand_w(num_classes, 4 * hidden_dim)
    layer_info['mlp_fc2_bias_fp32'] = rand_bias(num_classes)
    layer_info['mlp_fc2_scale_weight'] = 0.01

    logits_int8, logit_scale, logits_fp32 = classification_head_with_mlp_int8(
        x_int8, layer_info, scale_input, use_i_softmax=True,
    )

    assert logits_fp32.shape == (batch_size, num_classes), f"Bad shape: {logits_fp32.shape}"
    assert logits_int8.shape == (batch_size, num_classes)
    print(f"PASS: logits shape {logits_fp32.shape}, scale={logit_scale:.6f}")
    print(f"      logits_fp32: {logits_fp32}")


if __name__ == '__main__':
    test_classification_head_with_mlp_int8()
