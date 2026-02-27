# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Atomic Cross-Attention Operation - INT8 Hybrid Precision

Cross-attention differs from self-attention only in the source of Q:
- Q comes from a learned query embedding table (stored as parameters)
- K/V come from the input activation tensor

The implementation mirrors the existing MHSA integer-softmax path to enable
bit-exact matching with the GAP9 C kernels:
- Q/K/V projections: INT8 → INT8 via linear_int8
- Scores: INT8xINT8 → INT32
- Softmax: integer-only LUT path producing UINT8 weights in [0,255]
- Context: UINT8xINT8 → INT32 → (acc+128)>>8 → INT8 (same scale as V)
- Output projection: INT8 → INT8 via linear_int8
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

# Handle both module import and standalone execution
try:
    from .quantize import quantize_linear, dequantize_linear
    from .linear import linear_int8
    from .mhsa import i_softmax_int32_to_uint8
except ImportError:  # pragma: no cover
    from quantize import quantize_linear, dequantize_linear  # type: ignore
    from linear import linear_int8  # type: ignore
    from mhsa import i_softmax_int32_to_uint8  # type: ignore


def cross_attention_int8_hybrid(
    kv_int8: np.ndarray,
    layer_info: Dict[str, Any],
    scale_kv_input: float,
    *,
    verbose: bool = False,
    use_i_softmax: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    INT8 Cross-Attention (hybrid precision) reference implementation.

    Args:
        kv_int8: Input K/V source tensor [B, N, D] (INT8).
        layer_info: Dict containing weights, biases, scales, and metadata.
            Required keys:
              - 'embed_dim', 'num_heads'
              - 'query_embed_int8': [Q, D] or [1, Q, D]
              - 'query_scale': quantization scale for query embeddings
              - 'q_weight_int8', 'k_weight_int8', 'v_weight_int8', 'out_weight_int8'
              - 'q_scale_weight', 'k_scale_weight', 'v_scale_weight', 'out_scale_weight'
              - 'q_scale_output', 'k_scale_output', 'v_scale_output' (for INT8 projections path)
              - 'scale_output' (final output scale)
            Optional:
              - 'q_bias_fp32', 'k_bias_fp32', 'v_bias_fp32', 'out_bias_fp32'
              - 'softmax_scale' (defaults to 1/sqrt(head_dim))
        scale_kv_input: Input activation scale for kv_int8.
        verbose: Print debug info.
        use_i_softmax: If True, use C-compatible i-softmax path (recommended).

    Returns:
        (out_int8, out_scale, out_fp32):
          - out_int8: [B, Q, D] INT8
          - out_scale: float, equals layer_info['scale_output']
          - out_fp32: dequantized output for verification
    """
    kv = np.asarray(kv_int8, dtype=np.int8)
    if kv.ndim != 3:
        raise ValueError(f"Cross-attention expects kv_int8 as [B, N, D], got {kv.shape}")

    batch_size, kv_len, embed_dim_actual = kv.shape
    embed_dim = int(layer_info.get("embed_dim", embed_dim_actual))
    if embed_dim_actual != embed_dim:
        raise ValueError(f"kv embed_dim mismatch: expected {embed_dim}, got {embed_dim_actual}")

    num_heads = int(layer_info.get("num_heads", 1))
    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")
    head_dim = int(layer_info.get("head_dim") or (embed_dim // num_heads))

    # Learned queries (INT8)
    query_embed = np.asarray(layer_info["query_embed_int8"], dtype=np.int8)
    if query_embed.ndim == 3 and query_embed.shape[0] == 1:
        query_embed = query_embed[0]
    if query_embed.ndim != 2 or query_embed.shape[1] != embed_dim:
        raise ValueError(f"query_embed_int8 must be [Q, D]=[*, {embed_dim}], got {query_embed.shape}")
    num_queries = int(query_embed.shape[0])

    scale_query = float(layer_info.get("query_scale", 1.0))
    scale_output = float(layer_info.get("scale_output", 1.0))

    # Projections (INT8 weights + FP32 biases)
    q_weight_int8 = np.asarray(layer_info["q_weight_int8"], dtype=np.int8)
    k_weight_int8 = np.asarray(layer_info["k_weight_int8"], dtype=np.int8)
    v_weight_int8 = np.asarray(layer_info["v_weight_int8"], dtype=np.int8)
    out_weight_int8 = np.asarray(layer_info["out_weight_int8"], dtype=np.int8)

    q_bias_fp32 = layer_info.get("q_bias_fp32")
    k_bias_fp32 = layer_info.get("k_bias_fp32")
    v_bias_fp32 = layer_info.get("v_bias_fp32")
    out_bias_fp32 = layer_info.get("out_bias_fp32")
    if q_bias_fp32 is not None and not isinstance(q_bias_fp32, np.ndarray):
        q_bias_fp32 = np.asarray(q_bias_fp32, dtype=np.float32)
    if k_bias_fp32 is not None and not isinstance(k_bias_fp32, np.ndarray):
        k_bias_fp32 = np.asarray(k_bias_fp32, dtype=np.float32)
    if v_bias_fp32 is not None and not isinstance(v_bias_fp32, np.ndarray):
        v_bias_fp32 = np.asarray(v_bias_fp32, dtype=np.float32)
    if out_bias_fp32 is not None and not isinstance(out_bias_fp32, np.ndarray):
        out_bias_fp32 = np.asarray(out_bias_fp32, dtype=np.float32)

    scale_q_weight = float(layer_info["q_scale_weight"])
    scale_k_weight = float(layer_info["k_scale_weight"])
    scale_v_weight = float(layer_info["v_scale_weight"])
    scale_out_weight = float(layer_info["out_scale_weight"])

    scale_q_output = layer_info.get("q_scale_output")
    scale_k_output = layer_info.get("k_scale_output")
    scale_v_output = layer_info.get("v_scale_output")

    use_int8_projections = (
        scale_q_output is not None and scale_k_output is not None and scale_v_output is not None
    )
    if not use_int8_projections and use_i_softmax:
        raise ValueError("C-compatible i-softmax requires INT8 projection outputs (q/k/v_scale_output)")

    scale_q_output = float(scale_q_output) if scale_q_output is not None else None
    scale_k_output = float(scale_k_output) if scale_k_output is not None else None
    scale_v_output = float(scale_v_output) if scale_v_output is not None else None

    softmax_scale = float(layer_info.get("softmax_scale", 1.0 / np.sqrt(head_dim)))

    # ---------------------------------------------------------------------
    # 1) Q/K/V projections
    # ---------------------------------------------------------------------
    # Flatten for projections
    queries = np.broadcast_to(query_embed[None, :, :], (batch_size, num_queries, embed_dim))
    q_in = queries.reshape(batch_size * num_queries, embed_dim)
    kv_in = kv.reshape(batch_size * kv_len, embed_dim)

    if use_int8_projections:
        q_bias_int32 = None if q_bias_fp32 is None else (q_bias_fp32 / (scale_query * scale_q_weight)).astype(np.int32)
        k_bias_int32 = None if k_bias_fp32 is None else (k_bias_fp32 / (scale_kv_input * scale_k_weight)).astype(np.int32)
        v_bias_int32 = None if v_bias_fp32 is None else (v_bias_fp32 / (scale_kv_input * scale_v_weight)).astype(np.int32)

        q_flat = linear_int8(q_in, q_weight_int8, q_bias_int32, scale_query, scale_q_weight, scale_q_output)
        k_flat = linear_int8(kv_in, k_weight_int8, k_bias_int32, scale_kv_input, scale_k_weight, scale_k_output)
        v_flat = linear_int8(kv_in, v_weight_int8, v_bias_int32, scale_kv_input, scale_v_weight, scale_v_output)

        q = q_flat.reshape(batch_size, num_queries, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k_flat.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v_flat.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # -----------------------------------------------------------------
        # 2) Attention scores + integer softmax
        # -----------------------------------------------------------------
        scores_int32 = np.matmul(
            q.astype(np.int32),
            np.transpose(k.astype(np.int32), (0, 1, 3, 2)),
        )  # [B, H, Q, N]

        attn_uint8 = i_softmax_int32_to_uint8(
            scores_int32,
            scale_q=scale_q_output,
            scale_k=scale_k_output,
            softmax_scale=softmax_scale,
            axis=-1,
        )

        # Context: UINT8xINT8 -> INT32 -> requant -> INT8 (same scale as V)
        context_int32 = np.matmul(attn_uint8.astype(np.int32), v.astype(np.int32))
        context_int8 = np.clip((context_int32 + 128) >> 8, -128, 127).astype(np.int8)

        # Reshape context for output projection: [B, H, Q, Dh] -> [B, Q, D]
        context_seq_int8 = context_int8.transpose(0, 2, 1, 3).reshape(batch_size, num_queries, embed_dim)

        # -----------------------------------------------------------------
        # 3) Output projection (INT8 -> INT8)
        # -----------------------------------------------------------------
        out_bias_int32 = (
            None
            if out_bias_fp32 is None
            else (out_bias_fp32 / (scale_v_output * scale_out_weight)).astype(np.int32)
        )
        out_flat = linear_int8(
            context_seq_int8.reshape(batch_size * num_queries, embed_dim),
            out_weight_int8,
            out_bias_int32,
            scale_v_output,
            scale_out_weight,
            scale_output,
        )
        out_int8 = out_flat.reshape(batch_size, num_queries, embed_dim)
        out_fp32 = dequantize_linear(out_int8, scale=scale_output, zero_point=0)

    else:
        # FP32 fallback: dequantize inputs/weights and run attention in FP32
        q_in_fp32 = dequantize_linear(q_in, scale=scale_query, zero_point=0)
        kv_in_fp32 = dequantize_linear(kv_in, scale=scale_kv_input, zero_point=0)

        q_w_fp32 = dequantize_linear(q_weight_int8, scale=scale_q_weight, zero_point=0)
        k_w_fp32 = dequantize_linear(k_weight_int8, scale=scale_k_weight, zero_point=0)
        v_w_fp32 = dequantize_linear(v_weight_int8, scale=scale_v_weight, zero_point=0)
        out_w_fp32 = dequantize_linear(out_weight_int8, scale=scale_out_weight, zero_point=0)

        q_flat_fp32 = q_in_fp32 @ q_w_fp32.T
        k_flat_fp32 = kv_in_fp32 @ k_w_fp32.T
        v_flat_fp32 = kv_in_fp32 @ v_w_fp32.T
        if q_bias_fp32 is not None:
            q_flat_fp32 += q_bias_fp32
        if k_bias_fp32 is not None:
            k_flat_fp32 += k_bias_fp32
        if v_bias_fp32 is not None:
            v_flat_fp32 += v_bias_fp32

        q_fp32 = q_flat_fp32.reshape(batch_size, num_queries, num_heads, head_dim).transpose(0, 2, 1, 3)
        k_fp32 = k_flat_fp32.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v_fp32 = v_flat_fp32.reshape(batch_size, kv_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(q_fp32, np.transpose(k_fp32, (0, 1, 3, 2))) * softmax_scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max).astype(np.float32)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        context_fp32 = np.matmul(attn, v_fp32).transpose(0, 2, 1, 3).reshape(batch_size, num_queries, embed_dim)

        out_fp32 = context_fp32.reshape(batch_size * num_queries, embed_dim) @ out_w_fp32.T
        if out_bias_fp32 is not None:
            out_fp32 += out_bias_fp32
        out_fp32 = out_fp32.reshape(batch_size, num_queries, embed_dim)
        out_int8 = quantize_linear(out_fp32, scale=scale_output, zero_point=0)

    if verbose:
        print(f"[CrossAttn] B={batch_size} KV={kv_len} Q={num_queries} D={embed_dim} H={num_heads} Dh={head_dim}")
        print(f"[CrossAttn] scales: kv_in={scale_kv_input:.6f}, q_in={scale_query:.6f}, out={scale_output:.6f}")
        print(f"[CrossAttn] out_int8 range: [{out_int8.min()}, {out_int8.max()}]")

    return out_int8, scale_output, out_fp32


def test_cross_attention_int8_hybrid():
    """Smoke test for INT8 Cross-Attention."""
    batch_size = 3
    kv_len = 7
    num_queries = 5
    embed_dim = 16
    num_heads = 4
    head_dim = embed_dim // num_heads

    kv_fp32 = (np.random.randn(batch_size, kv_len, embed_dim).astype(np.float32) * 0.1)
    scale_kv = 0.02
    kv_int8 = quantize_linear(kv_fp32, scale=scale_kv, zero_point=0)

    query_fp32 = (np.random.randn(num_queries, embed_dim).astype(np.float32) * 0.1)
    scale_query = 0.03
    query_int8 = quantize_linear(query_fp32, scale=scale_query, zero_point=0)

    scale_weight = 0.01
    scale_q_out = 0.02
    scale_k_out = 0.02
    scale_v_out = 0.02
    scale_out = 0.025

    def rand_w():
        return (np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1)

    q_w_i8 = quantize_linear(rand_w(), scale=scale_weight, zero_point=0)
    k_w_i8 = quantize_linear(rand_w(), scale=scale_weight, zero_point=0)
    v_w_i8 = quantize_linear(rand_w(), scale=scale_weight, zero_point=0)
    o_w_i8 = quantize_linear(rand_w(), scale=scale_weight, zero_point=0)

    q_b = (np.random.randn(embed_dim).astype(np.float32) * 0.01)
    k_b = (np.random.randn(embed_dim).astype(np.float32) * 0.01)
    v_b = (np.random.randn(embed_dim).astype(np.float32) * 0.01)
    o_b = (np.random.randn(embed_dim).astype(np.float32) * 0.01)

    layer_info = {
        "type": "CrossAttention",
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "query_embed_int8": query_int8,
        "query_scale": scale_query,
        "q_weight_int8": q_w_i8,
        "k_weight_int8": k_w_i8,
        "v_weight_int8": v_w_i8,
        "out_weight_int8": o_w_i8,
        "q_bias_fp32": q_b,
        "k_bias_fp32": k_b,
        "v_bias_fp32": v_b,
        "out_bias_fp32": o_b,
        "q_scale_weight": scale_weight,
        "k_scale_weight": scale_weight,
        "v_scale_weight": scale_weight,
        "out_scale_weight": scale_weight,
        "q_scale_output": scale_q_out,
        "k_scale_output": scale_k_out,
        "v_scale_output": scale_v_out,
        "scale_output": scale_out,
        "softmax_scale": 1.0 / np.sqrt(head_dim),
    }

    out_int8, out_scale, out_fp32 = cross_attention_int8_hybrid(
        kv_int8, layer_info, scale_kv, use_i_softmax=True
    )

    assert out_int8.shape == (batch_size, num_queries, embed_dim)
    assert abs(out_scale - scale_out) < 1e-12
    assert out_fp32.shape == (batch_size, num_queries, embed_dim)


if __name__ == "__main__":  # pragma: no cover
    test_cross_attention_int8_hybrid()

