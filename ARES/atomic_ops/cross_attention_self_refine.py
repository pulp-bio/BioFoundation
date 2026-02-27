# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Cross-Attention with Self-Refinement — INT8 Reference Implementation

Matches QuantCrossAttentionWithSelfRefine from brevitas_custom_layers.py:
1. LayerNorm on queries/keys/values
2. Multi-head cross-attention (learned queries attend to input)
3. FFN with residual
4. 3x self-attention refinement blocks (each: LN→MHSA→residual→LN→MLP→residual)
"""

from typing import Any, Dict, Tuple

import numpy as np

try:
    from .quantize import quantize_linear, dequantize_linear
    from .linear import linear_int8
    from .layernorm import layernorm_int8
    from .gelu import gelu_int8_lut
    from .add import add_int8
    from .mhsa import i_softmax_int32_to_uint8
except ImportError:
    from quantize import quantize_linear, dequantize_linear  # type: ignore
    from linear import linear_int8  # type: ignore
    from layernorm import layernorm_int8  # type: ignore
    from gelu import gelu_int8_lut  # type: ignore
    from add import add_int8  # type: ignore
    from mhsa import i_softmax_int32_to_uint8  # type: ignore


def _convert_bias_int32(bias_fp32, scale_input, scale_weight):
    """Convert FP32 bias to INT32 domain: bias_int32 = round(bias_fp32 / (s_in * s_w))."""
    if bias_fp32 is None:
        return None
    return np.round(bias_fp32 / (scale_input * scale_weight)).astype(np.int32)


def _run_mhsa_core(
    q_int8, k_int8, v_int8,
    out_weight_int8, out_bias_fp32,
    scale_q, scale_k, scale_v, scale_out_weight, scale_out,
    num_heads, head_dim, softmax_scale,
    batch_size, q_len, kv_len, embed_dim,
    use_i_softmax=False,
):
    """Run multi-head attention core: reshape → scores → softmax → context → output proj."""
    # Reshape to multi-head: [B, SeqLen, D] -> [B, H, SeqLen, Dh]
    q = q_int8.reshape(batch_size, q_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k_int8.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v_int8.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention scores: INT8 x INT8 -> INT32
    scores_int32 = np.matmul(q.astype(np.int32), k.astype(np.int32).transpose(0, 1, 3, 2))

    # Softmax
    if use_i_softmax:
        attn_uint8 = i_softmax_int32_to_uint8(
            scores_int32, scale_q=scale_q, scale_k=scale_k,
            softmax_scale=softmax_scale, axis=-1,
        )
    else:
        # FP32 fallback softmax
        scores_fp32 = scores_int32.astype(np.float32) * scale_q * scale_k * softmax_scale
        scores_max = np.max(scores_fp32, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_fp32 - scores_max).astype(np.float32)
        attn_fp32 = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        attn_uint8 = np.clip(np.round(attn_fp32 * 255.0), 0, 255).astype(np.uint8)

    # Context: UINT8 x INT8 -> INT32 -> (acc+128)>>8 -> INT8
    context_int32 = np.matmul(attn_uint8.astype(np.int32), v.astype(np.int32))
    context_int8 = np.clip((context_int32 + 128) >> 8, -128, 127).astype(np.int8)

    # Reshape back: [B, H, Q, Dh] -> [B*Q, D]
    context_flat = context_int8.transpose(0, 2, 1, 3).reshape(batch_size * q_len, embed_dim)

    # Output projection
    out_bias_int32 = _convert_bias_int32(out_bias_fp32, scale_v, scale_out_weight)
    out_flat = linear_int8(context_flat, out_weight_int8, out_bias_int32,
                           scale_v, scale_out_weight, scale_out)
    return out_flat.reshape(batch_size, q_len, embed_dim)


def cross_attention_with_self_refine_int8(
    kv_int8: np.ndarray,
    layer_info: Dict[str, Any],
    scale_kv_input: float,
    *,
    use_i_softmax: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    INT8 Cross-Attention with Self-Refinement reference implementation.

    Args:
        kv_int8: Input K/V tensor [B, N, D] INT8.
        layer_info: Flat dict with all weights, biases, and scales.
        scale_kv_input: Quantization scale for kv_int8.
        use_i_softmax: Use integer-only softmax (recommended for bit-exact).

    Returns:
        (out_int8, out_scale, out_fp32) where out_int8 is [B, num_queries, D].
    """
    embed_dim = layer_info['embed_dim']
    num_heads = layer_info['num_heads']
    head_dim = layer_info['head_dim']
    num_queries = layer_info['num_queries']
    softmax_scale = layer_info['softmax_scale']
    num_sa_blocks = layer_info.get('num_self_attn_blocks', 3)

    batch_size = kv_int8.shape[0]
    kv_len = kv_int8.shape[1]

    # --- Stage 1: LayerNorm on queries/keys/values ---
    query_embed_int8 = np.array(layer_info['query_embed_int8'])
    if query_embed_int8.ndim == 2:
        query_embed_int8 = np.tile(query_embed_int8[np.newaxis], (batch_size, 1, 1))
    scale_query = layer_info['query_scale']

    q_normed = layernorm_int8(
        query_embed_int8, layer_info['queries_norm_weight'], layer_info['queries_norm_bias'],
        scale_query, layer_info['queries_norm_scale_output'], embed_dim,
    )
    scale_q_normed = layer_info['queries_norm_scale_output']

    k_normed = layernorm_int8(
        kv_int8, layer_info['keys_norm_weight'], layer_info['keys_norm_bias'],
        scale_kv_input, layer_info['keys_norm_scale_output'], embed_dim,
    )
    scale_k_normed = layer_info['keys_norm_scale_output']

    v_normed = layernorm_int8(
        kv_int8, layer_info['values_norm_weight'], layer_info['values_norm_bias'],
        scale_kv_input, layer_info['values_norm_scale_output'], embed_dim,
    )
    scale_v_normed = layer_info['values_norm_scale_output']

    # --- Stage 2: Cross-attention projections ---
    q_flat = linear_int8(
        q_normed.reshape(batch_size * num_queries, embed_dim),
        layer_info['q_weight_int8'],
        _convert_bias_int32(layer_info['q_bias_fp32'], scale_q_normed, layer_info['q_scale_weight']),
        scale_q_normed, layer_info['q_scale_weight'], layer_info['q_scale_output'],
    ).reshape(batch_size, num_queries, embed_dim)

    k_flat = linear_int8(
        k_normed.reshape(batch_size * kv_len, embed_dim),
        layer_info['k_weight_int8'],
        _convert_bias_int32(layer_info['k_bias_fp32'], scale_k_normed, layer_info['k_scale_weight']),
        scale_k_normed, layer_info['k_scale_weight'], layer_info['k_scale_output'],
    ).reshape(batch_size, kv_len, embed_dim)

    v_flat = linear_int8(
        v_normed.reshape(batch_size * kv_len, embed_dim),
        layer_info['v_weight_int8'],
        _convert_bias_int32(layer_info['v_bias_fp32'], scale_v_normed, layer_info['v_scale_weight']),
        scale_v_normed, layer_info['v_scale_weight'], layer_info['v_scale_output'],
    ).reshape(batch_size, kv_len, embed_dim)

    # --- Stage 3: Attention core ---
    cross_out = _run_mhsa_core(
        q_flat, k_flat, v_flat,
        layer_info['out_weight_int8'], layer_info['out_bias_fp32'],
        layer_info['q_scale_output'], layer_info['k_scale_output'],
        layer_info['v_scale_output'], layer_info['out_scale_weight'],
        layer_info['out_scale_output'],
        num_heads, head_dim, softmax_scale,
        batch_size, num_queries, kv_len, embed_dim,
        use_i_softmax=use_i_softmax,
    )
    scale_cross_out = layer_info['out_scale_output']

    # --- Stage 4: FFN with residual ---
    ffn_in = cross_out.reshape(batch_size * num_queries, embed_dim)
    ffn_h = linear_int8(
        ffn_in,
        layer_info['ffn_fc1_weight_int8'],
        _convert_bias_int32(layer_info['ffn_fc1_bias_fp32'], scale_cross_out, layer_info['ffn_fc1_scale_weight']),
        scale_cross_out, layer_info['ffn_fc1_scale_weight'], layer_info['ffn_gelu_scale'],
    )
    ffn_h = gelu_int8_lut(ffn_h, layer_info['ffn_gelu_scale'], layer_info['ffn_gelu_scale'])
    ffn_fc2_scale_out = layer_info['ffn_add_scale']  # fc2 output scale = add output scale
    ffn_out = linear_int8(
        ffn_h,
        layer_info['ffn_fc2_weight_int8'],
        _convert_bias_int32(layer_info['ffn_fc2_bias_fp32'], layer_info['ffn_gelu_scale'], layer_info['ffn_fc2_scale_weight']),
        layer_info['ffn_gelu_scale'], layer_info['ffn_fc2_scale_weight'], ffn_fc2_scale_out,
    ).reshape(batch_size, num_queries, embed_dim)

    current = add_int8(ffn_out, cross_out, ffn_fc2_scale_out, scale_cross_out, layer_info['ffn_add_scale'])
    current_scale = layer_info['ffn_add_scale']

    # --- Stage 5: 3x Self-attention refinement ---
    for i in range(num_sa_blocks):
        pfx = f'sa{i}'
        residual = current
        residual_scale = current_scale

        # Norm1
        normed = layernorm_int8(
            current, layer_info[f'{pfx}_norm1_weight'], layer_info[f'{pfx}_norm1_bias'],
            current_scale, layer_info[f'{pfx}_norm1_scale_output'], embed_dim,
        )
        norm_scale = layer_info[f'{pfx}_norm1_scale_output']

        # Self-attention projections
        flat = normed.reshape(batch_size * num_queries, embed_dim)
        sq = linear_int8(
            flat, layer_info[f'{pfx}_q_weight_int8'],
            _convert_bias_int32(layer_info[f'{pfx}_q_bias_fp32'], norm_scale, layer_info[f'{pfx}_q_scale_weight']),
            norm_scale, layer_info[f'{pfx}_q_scale_weight'], layer_info[f'{pfx}_q_scale_output'],
        ).reshape(batch_size, num_queries, embed_dim)

        sk = linear_int8(
            flat, layer_info[f'{pfx}_k_weight_int8'],
            _convert_bias_int32(layer_info[f'{pfx}_k_bias_fp32'], norm_scale, layer_info[f'{pfx}_k_scale_weight']),
            norm_scale, layer_info[f'{pfx}_k_scale_weight'], layer_info[f'{pfx}_k_scale_output'],
        ).reshape(batch_size, num_queries, embed_dim)

        sv = linear_int8(
            flat, layer_info[f'{pfx}_v_weight_int8'],
            _convert_bias_int32(layer_info[f'{pfx}_v_bias_fp32'], norm_scale, layer_info[f'{pfx}_v_scale_weight']),
            norm_scale, layer_info[f'{pfx}_v_scale_weight'], layer_info[f'{pfx}_v_scale_output'],
        ).reshape(batch_size, num_queries, embed_dim)

        # Self-attention core
        attn_out = _run_mhsa_core(
            sq, sk, sv,
            layer_info[f'{pfx}_out_weight_int8'], layer_info[f'{pfx}_out_bias_fp32'],
            layer_info[f'{pfx}_q_scale_output'], layer_info[f'{pfx}_k_scale_output'],
            layer_info[f'{pfx}_v_scale_output'], layer_info[f'{pfx}_out_scale_weight'],
            layer_info[f'{pfx}_out_scale_output'],
            num_heads, head_dim, softmax_scale,
            batch_size, num_queries, num_queries, embed_dim,
            use_i_softmax=use_i_softmax,
        )
        attn_out_scale = layer_info[f'{pfx}_out_scale_output']

        # Residual add1
        current = add_int8(attn_out, residual, attn_out_scale, residual_scale, layer_info[f'{pfx}_add1_scale'])
        current_scale = layer_info[f'{pfx}_add1_scale']

        # Norm2 + MLP
        residual = current
        residual_scale = current_scale

        normed2 = layernorm_int8(
            current, layer_info[f'{pfx}_norm2_weight'], layer_info[f'{pfx}_norm2_bias'],
            current_scale, layer_info[f'{pfx}_norm2_scale_output'], embed_dim,
        )
        norm2_scale = layer_info[f'{pfx}_norm2_scale_output']

        flat2 = normed2.reshape(batch_size * num_queries, embed_dim)
        mlp_h = linear_int8(
            flat2, layer_info[f'{pfx}_mlp_fc1_weight_int8'],
            _convert_bias_int32(layer_info[f'{pfx}_mlp_fc1_bias_fp32'], norm2_scale, layer_info[f'{pfx}_mlp_fc1_scale_weight']),
            norm2_scale, layer_info[f'{pfx}_mlp_fc1_scale_weight'], layer_info[f'{pfx}_mlp_gelu_scale'],
        )
        mlp_h = gelu_int8_lut(mlp_h, layer_info[f'{pfx}_mlp_gelu_scale'], layer_info[f'{pfx}_mlp_gelu_scale'])
        mlp_fc2_scale_out = layer_info[f'{pfx}_add2_scale']
        mlp_out = linear_int8(
            mlp_h, layer_info[f'{pfx}_mlp_fc2_weight_int8'],
            _convert_bias_int32(layer_info[f'{pfx}_mlp_fc2_bias_fp32'], layer_info[f'{pfx}_mlp_gelu_scale'], layer_info[f'{pfx}_mlp_fc2_scale_weight']),
            layer_info[f'{pfx}_mlp_gelu_scale'], layer_info[f'{pfx}_mlp_fc2_scale_weight'], mlp_fc2_scale_out,
        ).reshape(batch_size, num_queries, embed_dim)

        # Residual add2
        current = add_int8(mlp_out, residual, mlp_fc2_scale_out, residual_scale, layer_info[f'{pfx}_add2_scale'])
        current_scale = layer_info[f'{pfx}_add2_scale']

    # Final output
    out_scale = layer_info['scale_output']
    # Requantize to output scale if different
    if abs(current_scale - out_scale) > 1e-10:
        out_fp32 = current.astype(np.float32) * current_scale
        out_int8 = np.clip(np.round(out_fp32 / out_scale), -128, 127).astype(np.int8)
    else:
        out_int8 = current
    out_fp32 = dequantize_linear(out_int8, scale=out_scale, zero_point=0)

    return out_int8, out_scale, out_fp32


def test_cross_attention_with_self_refine_int8():
    """Smoke test for the full cross-attention with self-refinement pipeline."""
    batch_size = 2
    kv_len = 7
    num_queries = 4
    embed_dim = 16
    num_heads = 2
    head_dim = embed_dim // num_heads
    ff_dim = 32

    np.random.seed(42)
    scale_kv = 0.02
    kv_fp32 = np.random.randn(batch_size, kv_len, embed_dim).astype(np.float32) * 0.1
    kv_int8 = quantize_linear(kv_fp32, scale=scale_kv, zero_point=0)

    def rand_w(rows, cols):
        return quantize_linear(
            np.random.randn(rows, cols).astype(np.float32) * 0.1, scale=0.01, zero_point=0
        )

    def rand_bias(size):
        return np.random.randn(size).astype(np.float32) * 0.01

    def rand_ln(size):
        return np.ones(size, dtype=np.float32), np.zeros(size, dtype=np.float32)

    layer_info = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'num_queries': num_queries,
        'ff_dim': ff_dim,
        'softmax_scale': 1.0 / np.sqrt(head_dim),
        'num_self_attn_blocks': 3,
    }

    # Query embedding
    layer_info['query_embed_int8'] = quantize_linear(
        np.random.randn(num_queries, embed_dim).astype(np.float32) * 0.1, scale=0.03, zero_point=0
    )
    layer_info['query_scale'] = 0.03

    # Layer norms
    for ln in ('queries_norm', 'keys_norm', 'values_norm'):
        w, b = rand_ln(embed_dim)
        layer_info[f'{ln}_weight'] = w
        layer_info[f'{ln}_bias'] = b
        layer_info[f'{ln}_scale_output'] = 0.02

    # Cross-attention projections
    for prefix in ('q', 'k', 'v', 'out'):
        layer_info[f'{prefix}_weight_int8'] = rand_w(embed_dim, embed_dim)
        layer_info[f'{prefix}_bias_fp32'] = rand_bias(embed_dim)
        layer_info[f'{prefix}_scale_weight'] = 0.01
        layer_info[f'{prefix}_scale_output'] = 0.02
        layer_info[f'{prefix}_in_features'] = embed_dim
        layer_info[f'{prefix}_out_features'] = embed_dim

    # FFN
    layer_info['ffn_fc1_weight_int8'] = rand_w(ff_dim, embed_dim)
    layer_info['ffn_fc1_bias_fp32'] = rand_bias(ff_dim)
    layer_info['ffn_fc1_scale_weight'] = 0.01
    layer_info['ffn_gelu_scale'] = 0.02
    layer_info['ffn_fc2_weight_int8'] = rand_w(embed_dim, ff_dim)
    layer_info['ffn_fc2_bias_fp32'] = rand_bias(embed_dim)
    layer_info['ffn_fc2_scale_weight'] = 0.01
    layer_info['ffn_add_scale'] = 0.025

    # 3x self-attention blocks
    for i in range(3):
        pfx = f'sa{i}'
        # Norms
        for norm_idx in ('1', '2'):
            w, b = rand_ln(embed_dim)
            layer_info[f'{pfx}_norm{norm_idx}_weight'] = w
            layer_info[f'{pfx}_norm{norm_idx}_bias'] = b
            layer_info[f'{pfx}_norm{norm_idx}_scale_output'] = 0.02
        # Self-attn projections
        for prefix in ('q', 'k', 'v', 'out'):
            layer_info[f'{pfx}_{prefix}_weight_int8'] = rand_w(embed_dim, embed_dim)
            layer_info[f'{pfx}_{prefix}_bias_fp32'] = rand_bias(embed_dim)
            layer_info[f'{pfx}_{prefix}_scale_weight'] = 0.01
            layer_info[f'{pfx}_{prefix}_scale_output'] = 0.02
            layer_info[f'{pfx}_{prefix}_in_features'] = embed_dim
            layer_info[f'{pfx}_{prefix}_out_features'] = embed_dim
        layer_info[f'{pfx}_add1_scale'] = 0.025
        # MLP
        layer_info[f'{pfx}_mlp_fc1_weight_int8'] = rand_w(ff_dim, embed_dim)
        layer_info[f'{pfx}_mlp_fc1_bias_fp32'] = rand_bias(ff_dim)
        layer_info[f'{pfx}_mlp_fc1_scale_weight'] = 0.01
        layer_info[f'{pfx}_mlp_gelu_scale'] = 0.02
        layer_info[f'{pfx}_mlp_fc2_weight_int8'] = rand_w(embed_dim, ff_dim)
        layer_info[f'{pfx}_mlp_fc2_bias_fp32'] = rand_bias(embed_dim)
        layer_info[f'{pfx}_mlp_fc2_scale_weight'] = 0.01
        layer_info[f'{pfx}_add2_scale'] = 0.025

    layer_info['scale_output'] = 0.025

    out_int8, out_scale, out_fp32 = cross_attention_with_self_refine_int8(
        kv_int8, layer_info, scale_kv, use_i_softmax=True,
    )

    assert out_int8.shape == (batch_size, num_queries, embed_dim), f"Bad shape: {out_int8.shape}"
    assert abs(out_scale - 0.025) < 1e-12
    assert out_fp32.shape == (batch_size, num_queries, embed_dim)
    print(f"PASS: output shape {out_int8.shape}, scale={out_scale:.4f}")
    print(f"      range=[{out_int8.min()}, {out_int8.max()}]")


if __name__ == '__main__':
    test_cross_attention_with_self_refine_int8()
