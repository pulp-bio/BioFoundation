# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Alternating Attention Operation - INT8

Implements the alternating attention mechanism from Cerebro transformer
for EEG signal processing using true INT8 arithmetic.

Alternating Attention Pattern:
- Even blocks: Channel attention (attend across channels for each time step)
- Odd blocks: Temporal attention (attend across time for each channel)

This atomic operation implements ONLY the attention mechanism:
    Input -> Reshape -> QKV -> AttnScores -> Softmax -> Context -> OutputProj -> Reshape -> Output

LayerNorms, residuals, and MLP blocks are handled as separate layers.

Reference:
    Cerebro Transformer (Zhang et al.)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    from .linear import linear_int8
    from .softmax import softmax_int8_lut, softmax_int8
    from .requantize import requantize_int8
except ImportError:
    from linear import linear_int8
    from softmax import softmax_int8_lut, softmax_int8
    from requantize import requantize_int8


def matmul_int8(
    a_int8: np.ndarray,
    b_int8: np.ndarray,
    scale_a: float,
    scale_b: float,
    scale_out: float
) -> np.ndarray:
    """
    INT8 matrix multiplication with scale handling.

    Computes C = A @ B where A, B are INT8 tensors with associated scales.
    Uses INT32 accumulation to prevent overflow.

    Formula:
        C_fp32 = (A_int8 * scale_a) @ (B_int8 * scale_b)
        C_int8 = quantize(C_fp32, scale_out)

    Combined:
        C_int8 = round((A_int8 @ B_int8) * (scale_a * scale_b / scale_out))

    Args:
        a_int8: Left matrix (INT8), shape [..., M, K]
        b_int8: Right matrix (INT8), shape [..., K, N]
        scale_a: Quantization scale for A
        scale_b: Quantization scale for B
        scale_out: Output quantization scale

    Returns:
        Output matrix (INT8), shape [..., M, N]
    """
    # Compute combined scale factor
    scale_factor = (scale_a * scale_b) / scale_out

    # INT8 x INT8 -> INT32 accumulation
    a_int32 = a_int8.astype(np.int32)
    b_int32 = b_int8.astype(np.int32)

    # Matrix multiply with INT32 accumulation
    c_int32 = np.matmul(a_int32, b_int32)

    # Scale and round
    c_float = c_int32.astype(np.float32) * np.float32(scale_factor)
    c_rounded = np.floor(c_float + 0.5).astype(np.int32)

    # Clip to INT8 range
    return np.clip(c_rounded, -128, 127).astype(np.int8)


def alternating_attention_int8(
    x_int8: np.ndarray,
    qkv_weight_int8: np.ndarray,
    qkv_bias_int32: Optional[np.ndarray],
    proj_weight_int8: np.ndarray,
    proj_bias_int32: Optional[np.ndarray],
    # Scales
    scale_x: float,
    scale_qkv_weight: float,
    scale_qkv_out: float,
    scale_q: float,
    scale_k: float,
    scale_v: float,
    scale_attn_scores: float,
    scale_softmax: float,
    scale_attn_out: float,
    scale_proj_weight: float,
    scale_output: float,
    # Configuration
    block_idx: int,
    num_heads: int,
    head_dim: int,
    num_channels: int,
    temporal_len: int,
    scaling_factor: float = None,
    use_lut_softmax: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    INT8 Alternating Attention for Cerebro Transformer.

    Executes the attention mechanism with channel/temporal alternation based
    on block index.

    Args:
        x_int8: Input tensor [B, SeqTotal, EmbedDim] in INT8
            where SeqTotal = num_channels * temporal_len
        qkv_weight_int8: QKV projection weights [3*D, D] in INT8
        qkv_bias_int32: QKV projection bias [3*D] in INT32 (optional)
        proj_weight_int8: Output projection weights [D, D] in INT8
        proj_bias_int32: Output projection bias [D] in INT32 (optional)
        scale_x: Input quantization scale
        scale_qkv_weight: QKV weight quantization scale
        scale_qkv_out: QKV output quantization scale (before Q/K/V split)
        scale_q: Q quantization scale (after 1/sqrt(d) scaling)
        scale_k: K quantization scale
        scale_v: V quantization scale
        scale_attn_scores: Attention scores scale (Q @ K^T output)
        scale_softmax: Softmax output scale
        scale_attn_out: Attention output scale (softmax @ V output)
        scale_proj_weight: Output projection weight scale
        scale_output: Final output scale
        block_idx: Block index (even=channel attn, odd=temporal attn)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_channels: Number of EEG channels
        temporal_len: Number of temporal patches
        scaling_factor: 1/sqrt(head_dim), applied to Q before attention
        use_lut_softmax: Use LUT-based softmax (more accurate)

    Returns:
        Tuple of (output_int8, debug_info)
    """
    debug_info = {}
    embed_dim = num_heads * head_dim
    is_channel_attn = (block_idx % 2 == 0)

    # Calculate scaling factor if not provided
    if scaling_factor is None:
        scaling_factor = 1.0 / np.sqrt(head_dim)

    # ---------------------------------------------------
    # 1. Reshape based on attention type
    # ---------------------------------------------------
    B, SeqTotal, D = x_int8.shape

    # View as [B, C, T, D]
    x_3d = x_int8.reshape(B, num_channels, temporal_len, D)

    if is_channel_attn:
        # Channel Attention: Attend over channels for each time step
        # Transpose to [B, T, C, D] -> Reshape to [B*T, C, D]
        x_attn = x_3d.transpose(0, 2, 1, 3).reshape(B * temporal_len, num_channels, D)
        local_batch = B * temporal_len
        local_seq = num_channels
    else:
        # Temporal Attention: Attend over time for each channel
        # Reshape to [B*C, T, D]
        x_attn = x_3d.reshape(B * num_channels, temporal_len, D)
        local_batch = B * num_channels
        local_seq = temporal_len

    # ---------------------------------------------------
    # 2. QKV Projection
    # ---------------------------------------------------
    x_flat = x_attn.reshape(-1, D)

    qkv_out = linear_int8(
        x_flat,
        qkv_weight_int8,
        qkv_bias_int32,
        scale_x,
        scale_qkv_weight,
        scale_qkv_out
    )

    # ---------------------------------------------------
    # 3. Split into Q, K, V and apply scaling to Q
    # ---------------------------------------------------
    qkv = qkv_out.reshape(local_batch, local_seq, 3, num_heads, head_dim)

    # Extract Q, K, V: [LocalBatch, Heads, LocalSeq, HeadDim]
    q = qkv[:, :, 0].transpose(0, 2, 1, 3)
    k = qkv[:, :, 1].transpose(0, 2, 1, 3)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)

    # Apply 1/sqrt(d) scaling to Q via requantization
    # Q_int8_final = Q_int8_raw * (scale_qkv_out * scaling_factor) / scale_q
    q = requantize_int8(q, scale_qkv_out * scaling_factor, scale_q)

    # Requantize K if scales differ
    if abs(scale_k - scale_qkv_out) > 1e-9:
        k = requantize_int8(k, scale_qkv_out, scale_k)

    # Requantize V if scales differ
    if abs(scale_v - scale_qkv_out) > 1e-9:
        v = requantize_int8(v, scale_qkv_out, scale_v)

    # ---------------------------------------------------
    # 4. Attention Scores (Q @ K^T)
    # ---------------------------------------------------
    k_t = k.transpose(0, 1, 3, 2)

    # Flatten batch and heads for matmul
    q_rs = q.reshape(local_batch * num_heads, local_seq, head_dim)
    k_t_rs = k_t.reshape(local_batch * num_heads, head_dim, local_seq)

    attn_scores = matmul_int8(
        q_rs, k_t_rs,
        scale_q, scale_k, scale_attn_scores
    )

    debug_info['scores'] = attn_scores

    # ---------------------------------------------------
    # 5. Softmax
    # ---------------------------------------------------
    if use_lut_softmax:
        attn_probs = softmax_int8_lut(
            attn_scores,
            scale_attn_scores,
            scale_softmax,
            axis=-1
        )
    else:
        attn_probs = softmax_int8(
            attn_scores,
            scale_attn_scores,
            scale_softmax,
            axis=-1
        )

    debug_info['attn_probs'] = attn_probs

    # ---------------------------------------------------
    # 6. Context (Attn @ V)
    # ---------------------------------------------------
    v_rs = v.reshape(local_batch * num_heads, local_seq, head_dim)

    context_rs = matmul_int8(
        attn_probs, v_rs,
        scale_softmax, scale_v, scale_attn_out
    )

    # ---------------------------------------------------
    # 7. Output Projection
    # ---------------------------------------------------
    # Reshape: [LocalBatch, Heads, LocalSeq, HeadDim] -> [LocalBatch*LocalSeq, D]
    context = context_rs.reshape(local_batch, num_heads, local_seq, head_dim)
    context = context.transpose(0, 2, 1, 3).reshape(local_batch * local_seq, embed_dim)

    output_int8 = linear_int8(
        context,
        proj_weight_int8,
        proj_bias_int32,
        scale_attn_out,
        scale_proj_weight,
        scale_output
    )

    # ---------------------------------------------------
    # 8. Final Reshape to Original [B, SeqTotal, D]
    # ---------------------------------------------------
    if is_channel_attn:
        # From [B*T, C, D] -> [B, T, C, D] -> [B, C, T, D]
        output_final = output_int8.reshape(B, temporal_len, num_channels, embed_dim)
        output_final = output_final.transpose(0, 2, 1, 3)
    else:
        # From [B*C, T, D] -> [B, C, T, D]
        output_final = output_int8.reshape(B, num_channels, temporal_len, embed_dim)

    # Flatten to [B, C*T, D]
    output_final = output_final.reshape(B, SeqTotal, embed_dim)

    return output_final, debug_info


def alternating_attention_fp32_reference(
    x: np.ndarray,
    qkv_weight: np.ndarray,
    qkv_bias: Optional[np.ndarray],
    proj_weight: np.ndarray,
    proj_bias: Optional[np.ndarray],
    block_idx: int,
    num_heads: int,
    head_dim: int,
    num_channels: int,
    temporal_len: int,
) -> np.ndarray:
    """FP32 reference implementation for testing."""
    embed_dim = num_heads * head_dim
    is_channel_attn = (block_idx % 2 == 0)
    scaling_factor = 1.0 / np.sqrt(head_dim)

    B, SeqTotal, D = x.shape
    x_3d = x.reshape(B, num_channels, temporal_len, D)

    if is_channel_attn:
        x_attn = x_3d.transpose(0, 2, 1, 3).reshape(B * temporal_len, num_channels, D)
        local_batch = B * temporal_len
        local_seq = num_channels
    else:
        x_attn = x_3d.reshape(B * num_channels, temporal_len, D)
        local_batch = B * num_channels
        local_seq = temporal_len

    # QKV projection
    x_flat = x_attn.reshape(-1, D)
    qkv = x_flat @ qkv_weight.T
    if qkv_bias is not None:
        qkv = qkv + qkv_bias

    qkv = qkv.reshape(local_batch, local_seq, 3, num_heads, head_dim)
    q = qkv[:, :, 0].transpose(0, 2, 1, 3)
    k = qkv[:, :, 1].transpose(0, 2, 1, 3)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)

    # Apply scaling to Q
    q = q * scaling_factor

    # Attention
    scores = q @ k.transpose(0, 1, 3, 2)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    context = attn @ v

    # Output projection
    context = context.transpose(0, 2, 1, 3).reshape(local_batch * local_seq, embed_dim)
    output = context @ proj_weight.T
    if proj_bias is not None:
        output = output + proj_bias

    # Reshape back
    if is_channel_attn:
        output = output.reshape(B, temporal_len, num_channels, embed_dim)
        output = output.transpose(0, 2, 1, 3)
    else:
        output = output.reshape(B, num_channels, temporal_len, embed_dim)

    return output.reshape(B, SeqTotal, embed_dim)


# --- Tests ---

def test_alternating_attention():
    """Unit tests for alternating attention."""
    print("=" * 70)
    print("Testing Alternating Attention (INT8)")
    print("=" * 70)

    # Configuration
    B, C, T, D = 1, 4, 8, 32
    Heads = 4
    head_dim = D // Heads

    np.random.seed(42)
    x = np.random.randint(-50, 50, (B, C * T, D)).astype(np.int8)

    # Weights
    qkv_w = np.random.randint(-10, 10, (3 * D, D)).astype(np.int8)
    proj_w = np.random.randint(-10, 10, (D, D)).astype(np.int8)

    # Scales
    scale_x = 0.0625
    scale_qkv_w = 0.0078125
    scale_qkv_out = 0.03125
    scale_q = 0.0078125
    scale_k = 0.03125
    scale_v = 0.03125
    scale_attn = 0.0625
    scale_softmax = 0.0078125
    scale_attn_out = 0.03125
    scale_proj_w = 0.0078125
    scale_out = 0.0625
    scaling_factor = 1.0 / np.sqrt(head_dim)

    # Test 1: Channel Attention
    print("\nTest 1: Channel Attention (Block 0)")
    print("-" * 60)

    out, debug = alternating_attention_int8(
        x, qkv_w, None, proj_w, None,
        scale_x, scale_qkv_w, scale_qkv_out,
        scale_q, scale_k, scale_v,
        scale_attn, scale_softmax, scale_attn_out,
        scale_proj_w, scale_out,
        0, Heads, head_dim, C, T, scaling_factor
    )

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min()}, {out.max()}]")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print("  OK: Channel attention passed")

    # Test 2: Temporal Attention
    print("\nTest 2: Temporal Attention (Block 1)")
    print("-" * 60)

    out, debug = alternating_attention_int8(
        x, qkv_w, None, proj_w, None,
        scale_x, scale_qkv_w, scale_qkv_out,
        scale_q, scale_k, scale_v,
        scale_attn, scale_softmax, scale_attn_out,
        scale_proj_w, scale_out,
        1, Heads, head_dim, C, T, scaling_factor
    )

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min()}, {out.max()}]")
    assert out.shape == x.shape
    print("  OK: Temporal attention passed")

    # Test 3: With bias
    print("\nTest 3: With QKV and projection bias")
    print("-" * 60)

    qkv_b = np.random.randint(-100, 100, (3 * D,)).astype(np.int32)
    proj_b = np.random.randint(-100, 100, (D,)).astype(np.int32)

    out, _ = alternating_attention_int8(
        x, qkv_w, qkv_b, proj_w, proj_b,
        scale_x, scale_qkv_w, scale_qkv_out,
        scale_q, scale_k, scale_v,
        scale_attn, scale_softmax, scale_attn_out,
        scale_proj_w, scale_out,
        0, Heads, head_dim, C, T, scaling_factor
    )

    assert out.shape == x.shape
    print("  OK: Attention with bias passed")

    # Test 4: Multiple block indices
    print("\nTest 4: Block indices 0-5")
    print("-" * 60)

    for block_idx in range(6):
        attn_type = "Channel" if block_idx % 2 == 0 else "Temporal"
        out, _ = alternating_attention_int8(
            x, qkv_w, None, proj_w, None,
            scale_x, scale_qkv_w, scale_qkv_out,
            scale_q, scale_k, scale_v,
            scale_attn, scale_softmax, scale_attn_out,
            scale_proj_w, scale_out,
            block_idx, Heads, head_dim, C, T, scaling_factor
        )
        assert out.shape == x.shape
        print(f"  Block {block_idx} ({attn_type:8}): range=[{out.min():4}, {out.max():4}]")

    print("\n" + "=" * 70)
    print("All Alternating Attention tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_alternating_attention()
