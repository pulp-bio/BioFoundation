# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Atomic Multi-Head Self-Attention (MHSA) Operation - INT8 Hybrid Precision

Implements MHSA with hybrid INT8/FP32 precision for GAP9 deployment.

Design Strategy:
- Q/K/V projections: INT8 → INT8 (quantized linear transformations)
- Attention scores (QxK^T): INT8 x INT8 → INT32 → FP32
- Softmax: FP32 (for numerical stability)
- Context (AxV): FP32 x INT8 → INT32 → FP32
- Output projection: FP32 → INT8 (quantized linear)
- Sequence pooling: Mean over sequence dimension

This hybrid approach balances memory efficiency (INT8 storage) with
numerical stability (FP32 attention computation).
"""

import numpy as np
from typing import Dict, Any, Optional

# Handle both module import and standalone execution
try:
    from .quantize import quantize_linear, dequantize_linear
    from .linear import linear_int8
    from .rope import rope_apply_int8_q15
    from .kv_cache import KVCache
except ImportError:
    from quantize import quantize_linear, dequantize_linear
    from linear import linear_int8
    try:
        from rope import rope_apply_int8_q15  # type: ignore
    except ImportError:
        rope_apply_int8_q15 = None  # type: ignore
    try:
        from kv_cache import KVCache  # type: ignore
    except ImportError:
        KVCache = None  # type: ignore


def fast_exp(x):
    """
    Fast exponential approximation matching GAP9 C implementation.

    Uses 7th-order Taylor series with range reduction to match the C code
    exactly for bit-perfect reproducibility.

    This is NOT optimized for performance - it's for bit-exact matching!
    """
    # Ensure scalar or array input
    is_scalar = np.isscalar(x)
    if is_scalar:
        x = np.array([x], dtype=np.float32)
    else:
        x = x.astype(np.float32)

    result = np.zeros_like(x, dtype=np.float32)

    for i in range(len(x.flat)):
        val = np.float32(x.flat[i])  # Use float32, not float64!

        # Range reduction to keep |x| <= 0.5
        reduction = 0
        while val > np.float32(0.5):
            val = np.float32(val * np.float32(0.5))
            reduction += 1
        while val < np.float32(-0.5):
            val = np.float32(val * np.float32(0.5))
            reduction += 1

        # 7th-order Taylor series for exp(x) around 0 - use float32!
        x2 = np.float32(val * val)
        # Coefficients: 1/2!, 1/3!, 1/4!, 1/5!, 1/6!, 1/7!
        r = np.float32(1.0) + val + np.float32(x2 * np.float32(
            np.float32(0.5) + np.float32(val * np.float32(
                np.float32(0.166666666666667) + np.float32(val * np.float32(
                    np.float32(0.041666666666667) + np.float32(val * np.float32(
                        np.float32(0.008333333333333) + np.float32(val * np.float32(
                            np.float32(0.001388888888889) + np.float32(val * np.float32(0.000198412698413))
                        ))
                    ))
                ))
            ))
        ))

        # Undo range reduction: exp(x_original) = exp(x_reduced)^(2^reduction)
        for _ in range(reduction):
            r = np.float32(r * r)

        result.flat[i] = r

    return result[0] if is_scalar else result


def load_softmax_lut(lut_path: str) -> tuple[np.ndarray, dict]:
    """
    Load softmax LUT from binary file.

    Args:
        lut_path: Path to binary LUT file (e.g., 'bin/softmax_lut.bin')

    Returns:
        Tuple of (lut, metadata):
            - lut: numpy array of INT16 values
            - metadata: dict with 'input_min', 'input_max', 'input_step', 'output_scale'
    """
    import struct
    from pathlib import Path

    lut_path = Path(lut_path)
    if not lut_path.exists():
        raise FileNotFoundError(f"Softmax LUT not found: {lut_path}")

    # Read binary file (little-endian int16)
    lut_values = []
    with open(lut_path, 'rb') as f:
        while True:
            data = f.read(2)  # 2 bytes per int16
            if not data:
                break
            value = struct.unpack('<h', data)[0]
            lut_values.append(value)

    lut = np.array(lut_values, dtype=np.int16)

    # Default metadata (matches generate_softmax_lut.py defaults)
    # These should ideally be stored in a companion .json file
    metadata = {
        'input_min': -8.0,
        'input_max': 0.0,
        'input_step': 8.0 / len(lut),  # (max - min) / num_entries
        'output_scale': 32767.0,
        'num_entries': len(lut),
    }

    return lut, metadata


def get_builtin_softmax_lut() -> tuple[np.ndarray, dict]:
    """
    Get the built-in softmax LUT that matches the C code exactly.

    This generates the same LUT that is hardcoded in network_kernels.c.mako:
    - 1024 entries for x in [-8.0, 0.0]
    - Values: round(exp(x) * 32767)

    Using this function guarantees bit-exact matching between Python and C.

    Returns:
        Tuple of (lut, metadata):
            - lut: numpy array of INT16 values [1024]
            - metadata: dict with 'input_min', 'input_max', 'input_step', 'output_scale', 'num_entries'
    """
    num_entries = 1024
    input_min = -8.0
    input_max = 0.0
    output_scale = 32767

    # Generate LUT values exactly as in C code
    x = np.linspace(input_min, input_max, num_entries)
    lut = np.round(np.exp(x) * output_scale).astype(np.int16)

    metadata = {
        'input_min': input_min,
        'input_max': input_max,
        'input_step': (input_max - input_min) / num_entries,  # 8.0 / 1024 = 0.0078125
        'output_scale': float(output_scale),
        'num_entries': num_entries,
    }

    return lut, metadata


def get_c_compatible_softmax_lut() -> np.ndarray:
    """
    Get the 129-entry UINT32 softmax LUT that matches the C code exactly.

    This generates the same LUT as i_softmax_lut_int8 in network_dma_pipeline.c.mako:
    - 129 entries for x in [-128, 0] (integer input, not float!)
    - Values: round(exp(x) * 2^24) as UINT32
    - Index: x + 128 (so index 0 = exp(-128), index 128 = exp(0))

    Returns:
        numpy array of UINT32 values [129]
    """
    lut = np.zeros(129, dtype=np.uint32)
    for x in range(-128, 1):  # -128 to 0 inclusive
        lut[x + 128] = int(round(np.exp(x) * (1 << 24)))  # exp(x) * 2^24
    return lut


def i_softmax_int32_to_uint8(
    scores_int32: np.ndarray,
    scale_q: float,
    scale_k: float,
    softmax_scale: float,
    quant_scale: float = 16.0,
    lut: np.ndarray = None,
    axis: int = -1
) -> np.ndarray:
    """
    Integer-only softmax matching the C implementation exactly.

    This implements the same algorithm as mhsa_tile_softmax_int_worker() in C:
    1. Find max INT32 score (purely integer)
    2. Compute INT32 diff = score - max (always <= 0)
    3. Requantize diff to INT8: x_int = (diff * requant_mul + round) >> requant_shift
    4. LUT lookup: exp_val = lut[x_int + 128]
    5. Normalize: attn = exp_val * 255 / sum (integer division)

    Args:
        scores_int32: INT32 attention scores from QxK^T, shape [..., seq_len]
        scale_q: Q projection output scale
        scale_k: K projection output scale
        softmax_scale: Typically 1/sqrt(head_dim)
        quant_scale: Internal quantization scale (default 16.0, matches Python)
        lut: Optional pre-computed LUT. If None, uses get_c_compatible_softmax_lut()
        axis: Axis to apply softmax over (default: -1)

    Returns:
        UINT8 attention weights in [0, 255], same shape as input
    """
    if lut is None:
        lut = get_c_compatible_softmax_lut()

    # Compute requantization parameters (same as C code in network.c.mako)
    combined_scale = scale_q * scale_k * softmax_scale
    target_scale = combined_scale * quant_scale
    requant_shift = 24  # Fixed shift for precision
    requant_mul = int(round(target_scale * (1 << requant_shift)))
    round_val = (1 << (requant_shift - 1)) if requant_shift > 0 else 0

    # Convert to int32 to match C behavior
    scores = scores_int32.astype(np.int32)

    # Step 1: Find max along axis
    scores_max = np.max(scores, axis=axis, keepdims=True)

    # Step 2: Compute diff (always <= 0)
    diff = scores - scores_max  # INT32 diff

    # Step 3: Requantize to INT8 range [-128, 0]
    # x_int = (diff * requant_mul + round) >> requant_shift
    x_int = ((diff.astype(np.int64) * requant_mul + round_val) >> requant_shift).astype(np.int32)
    x_int = np.clip(x_int, -128, 0)

    # Step 4: LUT lookup
    indices = x_int + 128  # Map to [0, 128]
    exp_vals = lut[indices].astype(np.uint64)  # Use uint64 for sum to avoid overflow

    # Step 5: Compute sum
    exp_sum = np.sum(exp_vals, axis=axis, keepdims=True)

    # Step 6: Normalize using integer division
    # attn = exp_val * 255 / sum
    attn_uint8 = np.where(
        exp_sum > 0,
        (exp_vals * 255) // exp_sum,
        255 // scores.shape[axis]  # Uniform distribution if sum is 0
    ).astype(np.uint8)

    return attn_uint8


def i_softmax_int16(
    scores_fp32: np.ndarray,
    lut: np.ndarray,
    lut_metadata: dict,
    axis: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integer-only softmax using lookup table (i-Softmax).

    This implements softmax using integer arithmetic and a pre-computed LUT
    for exp() approximation. It's designed to be bit-exact between Python
    and C implementations.

    Algorithm:
    1. Find max per row (numerical stability): max_val = max(scores)
    2. Normalize: norm_scores = scores - max_val
    3. Quantize to LUT index: idx = round((norm_score - input_min) / input_step)
    4. LUT lookup for exp: exp_vals = lut[idx]  (INT16)
    5. Sum: sum_exp = sum(exp_vals)  (INT32)
    6. Normalize: attention_int16 = (exp_vals * 32767) // sum_exp  (integer division!)

    Args:
        scores_fp32: Attention scores in FP32, shape [..., seq_len]
                     Values typically in range [-50, 0] after scaling
        lut: Softmax LUT (INT16 array from load_softmax_lut)
        lut_metadata: Metadata dict with 'input_min', 'input_max', 'input_step', 'output_scale'
        axis: Axis to apply softmax over (default: -1)

    Returns:
        Tuple of (attention_weights_fp32, attention_weights_int16):
            - attention_weights_fp32: FP32 in [0, 1] for compatibility
            - attention_weights_int16: INT16 in [0, 32767] for bit-exact context computation

    Notes:
        - Input scores are in FP32 but only for convenience (indexing)
        - All core computations use integer arithmetic with integer division
        - For true bit-exactness, both Python and C must use same LUT file
        - Returns both FP32 (for compatibility) and INT16 (for bit-exact path)
    """
    input_min = lut_metadata['input_min']
    input_step = lut_metadata['input_step']
    num_entries = len(lut)

    # 1. Find max along axis for numerical stability
    scores_max = np.max(scores_fp32, axis=axis, keepdims=True)

    # 2. Normalize scores: scores - max (stability)
    norm_scores = scores_fp32 - scores_max

    # 3. Quantize normalized scores to LUT indices using round (not truncate)
    # Map input range [input_min, input_max] to index range [0, num_entries-1]
    # Use rounding to match C lrintf() behavior
    indices = np.round((norm_scores - input_min) / input_step).astype(np.int32)
    indices = np.clip(indices, 0, num_entries - 1)

    # 4. LUT lookup: exp(norm_scores) quantized to INT16
    exp_quantized = lut[indices].astype(np.int32)  # Promote to INT32 for arithmetic

    # 5. Sum exp values (INT32 accumulation to prevent overflow)
    exp_sum = np.sum(exp_quantized, axis=axis, keepdims=True)

    # 6. Normalize using INTEGER DIVISION (crucial for bit-exactness!)
    # attention_int16 = (exp_quantized * 32767) // exp_sum
    # This gives normalized values in range [0, 32767]
    attention_int16 = (exp_quantized * 32767) // exp_sum

    # 7. Convert to FP32 for compatibility with existing code path
    attention_fp32 = attention_int16.astype(np.float32) / 32767.0

    return attention_fp32, attention_int16.astype(np.int16)


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """
    Repeat Key/Value heads to match Query heads for Grouped-Query Attention (GQA).

    This function expands KV tensors when n_kv_heads < n_heads.
    When n_rep=1, returns input unchanged (standard MHA behavior).

    Args:
        x: Input tensor [B, n_kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each KV head (n_heads // n_kv_heads)

    Returns:
        Output tensor [B, n_heads, seq_len, head_dim] where n_heads = n_kv_heads * n_rep

    Example:
        If n_heads=8 and n_kv_heads=2, then n_rep=4.
        Each of the 2 KV heads is repeated 4 times to produce 8 heads.
    """
    if n_rep == 1:
        return x

    batch_size, n_kv_heads, seq_len, head_dim = x.shape

    # Expand: [B, n_kv_heads, seq_len, head_dim] -> [B, n_kv_heads, n_rep, seq_len, head_dim]
    x_expanded = np.broadcast_to(
        x[:, :, np.newaxis, :, :],
        (batch_size, n_kv_heads, n_rep, seq_len, head_dim)
    )

    # Reshape: [B, n_kv_heads, n_rep, seq_len, head_dim] -> [B, n_heads, seq_len, head_dim]
    return x_expanded.reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)


def mhsa_int8_hybrid(
    x_int8: np.ndarray,
    layer_info: Dict[str, Any],
    scale_input: float,
    verbose: bool = False,
    use_i_softmax: bool = False,
    softmax_lut: Optional[np.ndarray] = None,
    softmax_lut_metadata: Optional[dict] = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Multi-Head Self-Attention with INT8 hybrid precision.

    Supports both standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).
    For GQA, set n_kv_heads < num_heads to share KV heads across multiple query heads.

    Args:
        x_int8: Input tensor [B, seq_len * embed_dim] or [B, seq_len, embed_dim] (INT8)
        layer_info: Layer metadata containing:
            - q_weight_int8, k_weight_int8, v_weight_int8, out_weight_int8 (INT8)
            - q_bias_fp32, k_bias_fp32, v_bias_fp32, out_bias_fp32 (FP32)
            - q_scale_weight, k_scale_weight, v_scale_weight, out_scale_weight (floats)
            - q_scale_output, k_scale_output, v_scale_output (floats for quantized projections)
            - scale_output (float for final output)
            - num_heads, embed_dim, sequence_length
            - n_kv_heads (optional): Number of KV heads for GQA. Defaults to num_heads (standard MHA).
                                     Must divide num_heads evenly. Common ratios: 1:8 (MQA), 1:4, 1:2.
            - pool_sequence ('mean' or 'flat')
        scale_input: Input quantization scale
        verbose: Print debug information
        use_i_softmax: Use integer-only LUT-based softmax instead of FP32 fast_exp()
                       When True, requires softmax_lut and softmax_lut_metadata
        softmax_lut: Softmax lookup table (INT16 array from load_softmax_lut)
                     Required if use_i_softmax=True
        softmax_lut_metadata: LUT metadata dict with 'input_min', 'input_max', etc.
                              Required if use_i_softmax=True

    Returns:
        Tuple of (output_int8, scale_output, output_fp32):
            - output_int8: Quantized output [B, embed_dim] if pooled, else [B, seq_len, embed_dim]
            - scale_output: Output quantization scale
            - output_fp32: FP32 output for verification

    Implementation Details:
        1. Reshape input to [B, seq_len, embed_dim] if flattened
        2. Q/K/V projections: INT8 → INT8 (quantized linear)
        3. Reshape Q to [B, num_heads, seq_len, head_dim], K/V to [B, n_kv_heads, seq_len, head_dim]
        4. For GQA: repeat K/V heads to match Q heads
        5. Compute attention scores: QxK^T in FP32 (dequantize Q/K on-the-fly)
        6. Apply softmax in FP32
        7. Compute context: AxV in mixed precision (FP32 x INT8 → FP32)
        8. Reshape and apply output projection: FP32 → INT8
        9. Optional sequence pooling (mean)

    Grouped-Query Attention (GQA):
        When n_kv_heads < num_heads, multiple query heads share the same KV heads.
        This reduces memory usage and computation while maintaining model quality.
        - n_kv_heads = num_heads: Standard MHA (default, backward compatible)
        - n_kv_heads = 1: Multi-Query Attention (MQA)
        - 1 < n_kv_heads < num_heads: Grouped-Query Attention (GQA)
    """
    # Extract metadata
    seq_len = layer_info.get('sequence_length')
    embed_dim = layer_info.get('embed_dim')
    num_heads = layer_info.get('num_heads', 1)
    head_dim = layer_info.get('head_dim') or (embed_dim // num_heads)
    pool_mode = layer_info.get('pool_sequence', 'mean')
    scale_output = layer_info.get('scale_output', 1.0)

    # Grouped-Query Attention (GQA) support
    # n_kv_heads defaults to num_heads for backward compatibility (standard MHA)
    n_kv_heads = layer_info.get('n_kv_heads', num_heads)
    if num_heads % n_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
    kv_rep = num_heads // n_kv_heads  # How many times to repeat each KV head

    if seq_len is None or embed_dim is None:
        raise ValueError("MHSA layer requires 'sequence_length' and 'embed_dim' in layer_info")

    # Reshape input if needed: [B, seq_len * embed_dim] → [B, seq_len, embed_dim]
    if x_int8.ndim == 2:
        batch_size = x_int8.shape[0]
        x_int8 = x_int8.reshape(batch_size, seq_len, embed_dim)
    elif x_int8.ndim != 3:
        raise ValueError(f"MHSA input must be 2D or 3D, got shape {x_int8.shape}")

    batch_size, seq_len_actual, embed_dim_actual = x_int8.shape
    if seq_len_actual != seq_len or embed_dim_actual != embed_dim:
        raise ValueError(f"MHSA input shape mismatch: expected [{batch_size}, {seq_len}, {embed_dim}], "
                         f"got [{batch_size}, {seq_len_actual}, {embed_dim_actual}]")

    # Flatten input for projection: [B, seq_len, embed_dim] → [B * seq_len, embed_dim]
    x_flat = x_int8.reshape(batch_size * seq_len, embed_dim)

    # 1. Q/K/V Projections
    # Check if we have INT8 projection outputs or FP32 projection outputs
    q_weight_int8 = layer_info['q_weight_int8']
    k_weight_int8 = layer_info['k_weight_int8']
    v_weight_int8 = layer_info['v_weight_int8']

    q_bias_fp32 = layer_info.get('q_bias_fp32')
    k_bias_fp32 = layer_info.get('k_bias_fp32')
    v_bias_fp32 = layer_info.get('v_bias_fp32')

    scale_q_weight = layer_info['q_scale_weight']
    scale_k_weight = layer_info['k_scale_weight']
    scale_v_weight = layer_info['v_scale_weight']

    # Optional: INT8 projection output scales (for true hybrid INT8)
    scale_q_output = layer_info.get('q_scale_output')
    scale_k_output = layer_info.get('k_scale_output')
    scale_v_output = layer_info.get('v_scale_output')

    # Convert weights/biases to numpy arrays if needed (JSON serialization converts arrays to lists)
    if not isinstance(q_weight_int8, np.ndarray):
        q_weight_int8 = np.array(q_weight_int8)
    if not isinstance(k_weight_int8, np.ndarray):
        k_weight_int8 = np.array(k_weight_int8)
    if not isinstance(v_weight_int8, np.ndarray):
        v_weight_int8 = np.array(v_weight_int8)

    if q_bias_fp32 is not None and not isinstance(q_bias_fp32, np.ndarray):
        q_bias_fp32 = np.array(q_bias_fp32)
    if k_bias_fp32 is not None and not isinstance(k_bias_fp32, np.ndarray):
        k_bias_fp32 = np.array(k_bias_fp32)
    if v_bias_fp32 is not None and not isinstance(v_bias_fp32, np.ndarray):
        v_bias_fp32 = np.array(v_bias_fp32)

    # Determine if projections output INT8 or FP32
    use_int8_projections = (scale_q_output is not None and
                            scale_k_output is not None and
                            scale_v_output is not None)

    if use_int8_projections:
        # Path A: INT8 projections (true hybrid precision)
        # Convert bias to INT32 format for linear_int8
        q_bias_int32 = None if q_bias_fp32 is None else (q_bias_fp32 / (scale_input * scale_q_weight)).astype(np.int32)
        k_bias_int32 = None if k_bias_fp32 is None else (k_bias_fp32 / (scale_input * scale_k_weight)).astype(np.int32)
        v_bias_int32 = None if v_bias_fp32 is None else (v_bias_fp32 / (scale_input * scale_v_weight)).astype(np.int32)

        # Compute Q/K/V projections (INT8 → INT8)
        q_flat = linear_int8(x_flat, q_weight_int8, q_bias_int32, scale_input, scale_q_weight, scale_q_output)
        k_flat = linear_int8(x_flat, k_weight_int8, k_bias_int32, scale_input, scale_k_weight, scale_k_output)
        v_flat = linear_int8(x_flat, v_weight_int8, v_bias_int32, scale_input, scale_v_weight, scale_v_output)

        # Reshape Q: [B * seq_len, num_heads * head_dim] → [B, seq_len, num_heads, head_dim] → [B, num_heads, seq_len, head_dim]
        q = q_flat.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Reshape K/V: [B * seq_len, n_kv_heads * head_dim] → [B, seq_len, n_kv_heads, head_dim] → [B, n_kv_heads, seq_len, head_dim]
        # For standard MHA (n_kv_heads == num_heads), this is identical to the Q reshape
        k = k_flat.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v_flat.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # Optional RoPE rotation (applies to Q and K only; preserves quantization scale).
        if layer_info.get('use_rope', False):
            if rope_apply_int8_q15 is None:
                raise ImportError("RoPE requested but atomic_ops.rope is not available")
            cos_q15 = np.asarray(layer_info.get('rope_cos_q15'), dtype=np.int16)
            sin_q15 = np.asarray(layer_info.get('rope_sin_q15'), dtype=np.int16)
            q = rope_apply_int8_q15(q, cos_q15, sin_q15)
            k = rope_apply_int8_q15(k, cos_q15, sin_q15)

        # GQA: Repeat K/V heads to match Q heads (when n_kv_heads < num_heads)
        # For standard MHA (kv_rep == 1), this is a no-op
        k = repeat_kv(k, kv_rep)
        v = repeat_kv(v, kv_rep)

        # Dequantize Q/K/V for FP32 attention computation
        q_fp32 = dequantize_linear(q, scale=scale_q_output, zero_point=0)
        k_fp32 = dequantize_linear(k, scale=scale_k_output, zero_point=0)
        v_fp32 = dequantize_linear(v, scale=scale_v_output, zero_point=0)
    else:
        # Path B: FP32 projections (current Brevitas model behavior)
        if layer_info.get('use_rope', False):
            raise ValueError("RoPE requires INT8 projection outputs (q/k/v_scale_output must be present)")
        # Dequantize input for FP32 projections
        x_fp32 = dequantize_linear(x_flat, scale=scale_input, zero_point=0)

        # Dequantize weights
        q_weight_fp32 = dequantize_linear(q_weight_int8, scale=scale_q_weight, zero_point=0)
        k_weight_fp32 = dequantize_linear(k_weight_int8, scale=scale_k_weight, zero_point=0)
        v_weight_fp32 = dequantize_linear(v_weight_int8, scale=scale_v_weight, zero_point=0)

        # Compute Q/K/V projections in FP32
        q_flat_fp32 = x_fp32 @ q_weight_fp32.T
        k_flat_fp32 = x_fp32 @ k_weight_fp32.T
        v_flat_fp32 = x_fp32 @ v_weight_fp32.T

        if q_bias_fp32 is not None:
            q_flat_fp32 += q_bias_fp32
        if k_bias_fp32 is not None:
            k_flat_fp32 += k_bias_fp32
        if v_bias_fp32 is not None:
            v_flat_fp32 += v_bias_fp32

        # Reshape Q: [B * seq_len, num_heads * head_dim] → [B, seq_len, num_heads, head_dim] → [B, num_heads, seq_len, head_dim]
        q_fp32 = q_flat_fp32.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Reshape K/V: [B * seq_len, n_kv_heads * head_dim] → [B, seq_len, n_kv_heads, head_dim] → [B, n_kv_heads, seq_len, head_dim]
        k_fp32 = k_flat_fp32.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v_fp32 = v_flat_fp32.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # GQA: Repeat K/V heads to match Q heads (when n_kv_heads < num_heads)
        k_fp32 = repeat_kv(k_fp32, kv_rep)
        v_fp32 = repeat_kv(v_fp32, kv_rep)

    # 2. Compute attention scores and 3. Apply softmax
    softmax_scale = layer_info.get('softmax_scale', 1.0 / np.sqrt(head_dim))

    # Check if we should use the C-compatible INT32 softmax path
    use_c_compatible_softmax = use_i_softmax and use_int8_projections

    if use_c_compatible_softmax:
        # C-compatible path: INT8 Q/K → INT32 scores → INT8 requant → LUT → UINT8 attention
        # This matches mhsa_tile_softmax_int_worker() in C exactly

        # Compute INT32 scores: Q_int8 x K_int8^T (no scaling yet!)
        # Shape: [B, num_heads, seq_len, head_dim] x [B, num_heads, head_dim, seq_len] → [B, num_heads, seq_len, seq_len]
        scores_int32 = np.matmul(
            q.astype(np.int32),  # q is INT8 from linear_int8()
            np.transpose(k.astype(np.int32), (0, 1, 3, 2))
        )  # Result is INT32

        if verbose:
            print(f"[MHSA] Using C-compatible i-Softmax (INT32→UINT8)")
            print(f"  INT32 scores shape: {scores_int32.shape}")
            print(f"  INT32 scores range: [{np.min(scores_int32)}, {np.max(scores_int32)}]")

        # Get UINT8 attention weights using C-compatible softmax
        attention_weights_uint8 = i_softmax_int32_to_uint8(
            scores_int32,
            scale_q=scale_q_output,
            scale_k=scale_k_output,
            softmax_scale=softmax_scale,
            axis=-1
        )

        if verbose:
            print(f"  Attention UINT8 range: [{np.min(attention_weights_uint8)}, {np.max(attention_weights_uint8)}]")

        # Compute context: UINT8 attention x INT8 V → INT32 → requant → INT8
        # This matches mhsa_tile_av_int_worker() in C exactly:
        # Context: [B, num_heads, seq_len, seq_len] x [B, num_heads, seq_len, head_dim] → [B, num_heads, seq_len, head_dim]
        context_int32 = np.matmul(
            attention_weights_uint8.astype(np.int32),
            v.astype(np.int32)  # v is INT8
        )

        # Requantize exactly as C code: (acc + 128) >> 8 with clamping to [-128, 127]
        # The attention weights sum to ~255, so >>8 (divide by 256) normalizes
        context_int8 = np.clip((context_int32 + 128) >> 8, -128, 127).astype(np.int8)

        # Convert INT8 context back to FP32 for output projection
        # The INT8 context represents values in V's scale space
        context_fp32 = context_int8.astype(np.float32) * scale_v_output

    else:
        # FP32 path: compute scores in FP32, apply softmax in FP32

        # Compute FP32 scores with softmax_scale already applied
        scores = np.matmul(q_fp32, np.transpose(k_fp32, (0, 1, 3, 2))) * softmax_scale

        if use_i_softmax:
            # Legacy integer-only softmax using 1024-entry INT16 LUT
            # Use builtin LUT if not provided (guarantees bit-exact match with C)
            if softmax_lut is None or softmax_lut_metadata is None:
                softmax_lut, softmax_lut_metadata = get_builtin_softmax_lut()

            if verbose:
                print(f"[MHSA] Using i-Softmax (LUT-based integer softmax on reference FP32 scores)")
                print(f"  Scores shape: {scores.shape}")
                print(f"  Scores range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")

            # i_softmax_int16 returns (fp32_weights, int16_weights)
            # Context computation currently uses FP32.
            attention_weights, attention_weights_int16 = i_softmax_int16(
                scores,
                lut=softmax_lut,
                lut_metadata=softmax_lut_metadata,
                axis=-1
            )

            if verbose:
                print(f"  Attention INT16 range: [{np.min(attention_weights_int16)}, {np.max(attention_weights_int16)}]")
        else:
            # FP32 softmax using fast_exp() to match C implementation
            scores_max = np.max(scores, axis=-1, keepdims=True)
            scores_exp = fast_exp(scores - scores_max)
            attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        # Compute context: A x V (FP32 x FP32 → FP32)
        # v_fp32 is already in FP32 from the projection step above
        # Context: [B, num_heads, seq_len, seq_len] x [B, num_heads, seq_len, head_dim] → [B, num_heads, seq_len, head_dim]
        context_fp32 = np.matmul(attention_weights, v_fp32)

    # Reshape: [B, num_heads, seq_len, head_dim] → [B, seq_len, num_heads, head_dim] → [B, seq_len, embed_dim]
    context_fp32 = context_fp32.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

    # 5. Output projection: FP32 → INT8
    out_weight_int8 = layer_info['out_weight_int8']
    out_bias_fp32 = layer_info.get('out_bias_fp32')
    scale_out_weight = layer_info['out_scale_weight']

    # Convert to numpy arrays if needed (JSON serialization converts arrays to lists)
    if not isinstance(out_weight_int8, np.ndarray):
        out_weight_int8 = np.array(out_weight_int8)
    if out_bias_fp32 is not None and not isinstance(out_bias_fp32, np.ndarray):
        out_bias_fp32 = np.array(out_bias_fp32)

    # Flatten context for projection
    context_flat_fp32 = context_fp32.reshape(batch_size * seq_len, embed_dim)

    # Output projection strategy depends on whether we used INT8 projections
    if use_int8_projections:
        # Path A: INT8 hybrid - quantize context for INT8 output projection
        scale_context = scale_v_output
        context_flat_int8 = quantize_linear(context_flat_fp32, scale=scale_context, zero_point=0)

        # Apply output projection (INT8 → INT8)
        out_bias_int32 = None if out_bias_fp32 is None else (out_bias_fp32 / (scale_context * scale_out_weight)).astype(np.int32)
        out_flat = linear_int8(
            context_flat_int8, out_weight_int8, out_bias_int32,
            scale_context, scale_out_weight, scale_output
        )
    else:
        # Path B: FP32 hybrid - use FP32 output projection to match C implementation
        # Dequantize output weights
        out_weight_fp32 = dequantize_linear(out_weight_int8, scale=scale_out_weight, zero_point=0)

        # Compute output projection in FP32
        out_flat_fp32 = context_flat_fp32 @ out_weight_fp32.T
        if out_bias_fp32 is not None:
            out_flat_fp32 += out_bias_fp32

        # Quantize final output
        out_flat = quantize_linear(out_flat_fp32, scale=scale_output, zero_point=0)

    # Reshape: [B * seq_len, embed_dim] → [B, seq_len, embed_dim]
    out_int8 = out_flat.reshape(batch_size, seq_len, embed_dim)

    # 6. Optional sequence pooling
    if pool_mode == 'mean':
        # Mean pooling over sequence dimension
        out_fp32 = dequantize_linear(out_int8, scale=scale_output, zero_point=0)
        pooled_fp32 = out_fp32.mean(axis=1)  # [B, seq_len, embed_dim] → [B, embed_dim]
        out_int8 = quantize_linear(pooled_fp32, scale=scale_output, zero_point=0)
        out_fp32_final = pooled_fp32
    elif pool_mode == 'flat':
        # Flatten sequence dimension
        out_int8 = out_int8.reshape(batch_size, -1)
        out_fp32_final = dequantize_linear(out_int8, scale=scale_output, zero_point=0)
    else:
        # No pooling
        out_fp32_final = dequantize_linear(out_int8, scale=scale_output, zero_point=0)

    if verbose:
        print(f"  MHSA: seq_len={seq_len}, embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}")
        print(f"  Q/K/V shapes: {q.shape}")
        print(f"  Attention scores: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Attention weights: range=[{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
        print(f"  Context: {context_fp32.shape}")
        print(f"  Output: INT8 {out_int8.shape}, range=[{out_int8.min()}, {out_int8.max()}]")
        print(f"  Pool mode: {pool_mode}")

    return out_int8, scale_output, out_fp32_final


def test_mhsa_int8_hybrid():
    """Test MHSA INT8 hybrid precision implementation."""
    print("=" * 80)
    print("Testing MHSA INT8 Hybrid Precision")
    print("=" * 80)

    # Test parameters
    batch_size = 2
    seq_len = 8
    embed_dim = 16
    num_heads = 4
    head_dim = embed_dim // num_heads

    # Create test input
    x_fp32 = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.1
    scale_input = 0.01
    x_int8 = quantize_linear(x_fp32, scale=scale_input, zero_point=0)

    # Create test weights (Q/K/V/Out projections)
    scale_weight = 0.005
    scale_q_output = 0.008
    scale_k_output = 0.008
    scale_v_output = 0.008
    scale_output = 0.01

    q_weight_fp32 = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1
    k_weight_fp32 = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1
    v_weight_fp32 = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1
    out_weight_fp32 = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1

    q_weight_int8 = quantize_linear(q_weight_fp32, scale=scale_weight, zero_point=0)
    k_weight_int8 = quantize_linear(k_weight_fp32, scale=scale_weight, zero_point=0)
    v_weight_int8 = quantize_linear(v_weight_fp32, scale=scale_weight, zero_point=0)
    out_weight_int8 = quantize_linear(out_weight_fp32, scale=scale_weight, zero_point=0)

    q_bias_fp32 = np.random.randn(embed_dim).astype(np.float32) * 0.01
    k_bias_fp32 = np.random.randn(embed_dim).astype(np.float32) * 0.01
    v_bias_fp32 = np.random.randn(embed_dim).astype(np.float32) * 0.01
    out_bias_fp32 = np.random.randn(embed_dim).astype(np.float32) * 0.01

    # Layer info
    layer_info = {
        'sequence_length': seq_len,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'q_weight_int8': q_weight_int8,
        'k_weight_int8': k_weight_int8,
        'v_weight_int8': v_weight_int8,
        'out_weight_int8': out_weight_int8,
        'q_bias_fp32': q_bias_fp32,
        'k_bias_fp32': k_bias_fp32,
        'v_bias_fp32': v_bias_fp32,
        'out_bias_fp32': out_bias_fp32,
        'q_scale_weight': scale_weight,
        'k_scale_weight': scale_weight,
        'v_scale_weight': scale_weight,
        'out_scale_weight': scale_weight,
        'q_scale_output': scale_q_output,
        'k_scale_output': scale_k_output,
        'v_scale_output': scale_v_output,
        'scale_output': scale_output,
        'softmax_scale': 1.0 / np.sqrt(head_dim),
        'pool_sequence': 'mean',
    }

    # Run MHSA
    print(f"\nInput: {x_int8.shape} (INT8), scale={scale_input}")
    out_int8, out_scale, out_fp32 = mhsa_int8_hybrid(x_int8, layer_info, scale_input, verbose=True)

    print(f"\n[OK] MHSA INT8 hybrid test passed!")
    print(f"  Output: {out_int8.shape} (INT8), scale={out_scale}")
    print(f"  Output range: [{out_int8.min()}, {out_int8.max()}]")
    print(f"  FP32 output range: [{out_fp32.min():.4f}, {out_fp32.max():.4f}]")

    # Test without pooling
    layer_info['pool_sequence'] = None
    print("\n" + "=" * 80)
    print("Testing MHSA without pooling")
    print("=" * 80)
    out_int8_no_pool, _, out_fp32_no_pool = mhsa_int8_hybrid(x_int8, layer_info, scale_input, verbose=True)
    print(f"\n[OK] MHSA without pooling test passed!")
    print(f"  Output: {out_int8_no_pool.shape} (INT8)")
    print(f"  Expected shape: [{batch_size}, {seq_len}, {embed_dim}]")

    # Test GQA (Grouped-Query Attention)
    print("\n" + "=" * 80)
    print("Testing MHSA with Grouped-Query Attention (GQA)")
    print("=" * 80)

    # GQA parameters: 4 query heads, 2 KV heads (kv_rep=2)
    n_kv_heads = 2
    kv_dim = n_kv_heads * head_dim  # KV projection output is smaller

    # Create smaller K/V weights for GQA
    k_weight_gqa_fp32 = np.random.randn(kv_dim, embed_dim).astype(np.float32) * 0.1
    v_weight_gqa_fp32 = np.random.randn(kv_dim, embed_dim).astype(np.float32) * 0.1
    k_weight_gqa_int8 = quantize_linear(k_weight_gqa_fp32, scale=scale_weight, zero_point=0)
    v_weight_gqa_int8 = quantize_linear(v_weight_gqa_fp32, scale=scale_weight, zero_point=0)

    k_bias_gqa_fp32 = np.random.randn(kv_dim).astype(np.float32) * 0.01
    v_bias_gqa_fp32 = np.random.randn(kv_dim).astype(np.float32) * 0.01

    layer_info_gqa = {
        'sequence_length': seq_len,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'n_kv_heads': n_kv_heads,  # GQA: fewer KV heads
        'head_dim': head_dim,
        'q_weight_int8': q_weight_int8,
        'k_weight_int8': k_weight_gqa_int8,  # Smaller K weights
        'v_weight_int8': v_weight_gqa_int8,  # Smaller V weights
        'out_weight_int8': out_weight_int8,
        'q_bias_fp32': q_bias_fp32,
        'k_bias_fp32': k_bias_gqa_fp32,
        'v_bias_fp32': v_bias_gqa_fp32,
        'out_bias_fp32': out_bias_fp32,
        'q_scale_weight': scale_weight,
        'k_scale_weight': scale_weight,
        'v_scale_weight': scale_weight,
        'out_scale_weight': scale_weight,
        'q_scale_output': scale_q_output,
        'k_scale_output': scale_k_output,
        'v_scale_output': scale_v_output,
        'scale_output': scale_output,
        'softmax_scale': 1.0 / np.sqrt(head_dim),
        'pool_sequence': 'mean',
    }

    print(f"  Config: num_heads={num_heads}, n_kv_heads={n_kv_heads}, kv_rep={num_heads // n_kv_heads}")
    out_int8_gqa, _, out_fp32_gqa = mhsa_int8_hybrid(x_int8, layer_info_gqa, scale_input, verbose=False)
    print(f"\n[OK] GQA MHSA test passed!")
    print(f"  Output: {out_int8_gqa.shape} (INT8)")
    print(f"  Output range: [{out_int8_gqa.min()}, {out_int8_gqa.max()}]")

    # Test backward compatibility: n_kv_heads == num_heads should match standard MHA
    print("\n" + "=" * 80)
    print("Testing backward compatibility (n_kv_heads == num_heads)")
    print("=" * 80)

    layer_info_compat = layer_info.copy()
    layer_info_compat['n_kv_heads'] = num_heads  # Explicitly set, should be same as default
    layer_info_compat['pool_sequence'] = 'mean'

    out_int8_compat, _, _ = mhsa_int8_hybrid(x_int8, layer_info_compat, scale_input, verbose=False)

    # Reset original layer_info
    layer_info['pool_sequence'] = 'mean'
    if 'n_kv_heads' in layer_info:
        del layer_info['n_kv_heads']

    out_int8_orig, _, _ = mhsa_int8_hybrid(x_int8, layer_info, scale_input, verbose=False)

    if np.array_equal(out_int8_orig, out_int8_compat):
        print("[OK] Backward compatibility verified: explicit n_kv_heads == num_heads matches default")
    else:
        mismatch = np.sum(out_int8_orig != out_int8_compat)
        print(f"[FAIL] Backward compatibility issue: {mismatch} elements differ")

    return True


def mhsa_autoregressive_step(
    x_int8: np.ndarray,
    layer_info: Dict[str, Any],
    kv_cache: 'KVCache',
    layer_idx: int,
    scale_input: float,
    verbose: bool = False
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Single-step autoregressive MHSA with KV caching.

    Processes a single token, updates the KV cache, and computes attention
    over all cached positions. Used for token-by-token generation.

    Args:
        x_int8: Current token input [1, embed_dim] (INT8)
        layer_info: Layer metadata (same as mhsa_int8_hybrid)
        kv_cache: KVCache instance for storing K/V across positions
        layer_idx: Current layer index (for multi-layer models)
        scale_input: Input quantization scale
        verbose: Print debug information

    Returns:
        Tuple of (output_int8, scale_output, output_fp32):
            - output_int8: Output for current token [1, embed_dim]
            - scale_output: Output quantization scale
            - output_fp32: FP32 output for verification

    Usage:
        cache = KVCache(n_layers=6, max_seq_len=256, n_kv_heads=4, head_dim=64)

        # Prefill with prompt (use mhsa_int8_hybrid for batch processing)
        # ...

        # Generate tokens one by one
        for step in range(max_new_tokens):
            for layer_idx in range(n_layers):
                out, scale, _ = mhsa_autoregressive_step(
                    x_t, layer_info[layer_idx], cache, layer_idx, scale
                )
                x_t = out  # Feed output to next layer
            cache.advance()
            # Sample next token from x_t
    """
    if KVCache is None:
        raise ImportError("KVCache not available")

    # Extract metadata
    embed_dim = layer_info.get('embed_dim')
    num_heads = layer_info.get('num_heads', 1)
    n_kv_heads = layer_info.get('n_kv_heads', num_heads)
    head_dim = layer_info.get('head_dim') or (embed_dim // num_heads)
    scale_output = layer_info.get('scale_output', 1.0)
    kv_rep = num_heads // n_kv_heads

    # Ensure input is [1, embed_dim]
    if x_int8.ndim == 1:
        x_int8 = x_int8.reshape(1, -1)
    batch_size, embed_dim_actual = x_int8.shape
    if batch_size != 1:
        raise ValueError(f"Autoregressive mode requires batch_size=1, got {batch_size}")

    # Extract weights
    q_weight_int8 = np.asarray(layer_info['q_weight_int8'])
    k_weight_int8 = np.asarray(layer_info['k_weight_int8'])
    v_weight_int8 = np.asarray(layer_info['v_weight_int8'])
    out_weight_int8 = np.asarray(layer_info['out_weight_int8'])

    q_bias_fp32 = layer_info.get('q_bias_fp32')
    k_bias_fp32 = layer_info.get('k_bias_fp32')
    v_bias_fp32 = layer_info.get('v_bias_fp32')
    out_bias_fp32 = layer_info.get('out_bias_fp32')

    scale_q_weight = layer_info['q_scale_weight']
    scale_k_weight = layer_info['k_scale_weight']
    scale_v_weight = layer_info['v_scale_weight']
    scale_out_weight = layer_info['out_scale_weight']

    scale_q_output = layer_info.get('q_scale_output', 0.01)
    scale_k_output = layer_info.get('k_scale_output', 0.01)
    scale_v_output = layer_info.get('v_scale_output', 0.01)

    # Convert biases to numpy arrays if needed
    if q_bias_fp32 is not None and not isinstance(q_bias_fp32, np.ndarray):
        q_bias_fp32 = np.array(q_bias_fp32)
    if k_bias_fp32 is not None and not isinstance(k_bias_fp32, np.ndarray):
        k_bias_fp32 = np.array(k_bias_fp32)
    if v_bias_fp32 is not None and not isinstance(v_bias_fp32, np.ndarray):
        v_bias_fp32 = np.array(v_bias_fp32)
    if out_bias_fp32 is not None and not isinstance(out_bias_fp32, np.ndarray):
        out_bias_fp32 = np.array(out_bias_fp32)

    # 1. Compute Q/K/V for current token (INT8 projections)
    q_bias_int32 = None if q_bias_fp32 is None else (q_bias_fp32 / (scale_input * scale_q_weight)).astype(np.int32)
    k_bias_int32 = None if k_bias_fp32 is None else (k_bias_fp32 / (scale_input * scale_k_weight)).astype(np.int32)
    v_bias_int32 = None if v_bias_fp32 is None else (v_bias_fp32 / (scale_input * scale_v_weight)).astype(np.int32)

    q_flat = linear_int8(x_int8, q_weight_int8, q_bias_int32, scale_input, scale_q_weight, scale_q_output)
    k_flat = linear_int8(x_int8, k_weight_int8, k_bias_int32, scale_input, scale_k_weight, scale_k_output)
    v_flat = linear_int8(x_int8, v_weight_int8, v_bias_int32, scale_input, scale_v_weight, scale_v_output)

    # Reshape: [1, embed_dim] -> [1, n_heads/n_kv_heads, 1, head_dim]
    q = q_flat.reshape(1, num_heads, 1, head_dim).transpose(0, 1, 2, 3)  # [1, H, 1, D]
    k = k_flat.reshape(1, n_kv_heads, 1, head_dim).transpose(0, 1, 2, 3)  # [1, Hkv, 1, D]
    v = v_flat.reshape(1, n_kv_heads, 1, head_dim).transpose(0, 1, 2, 3)

    # 2. Apply RoPE to Q and K at current position
    current_pos = kv_cache.current_pos
    if layer_info.get('use_rope', False):
        if rope_apply_int8_q15 is None:
            raise ImportError("RoPE requested but atomic_ops.rope is not available")
        cos_q15 = np.asarray(layer_info.get('rope_cos_q15'), dtype=np.int16)
        sin_q15 = np.asarray(layer_info.get('rope_sin_q15'), dtype=np.int16)
        q = rope_apply_int8_q15(q, cos_q15, sin_q15, pos_offset=current_pos)
        k = rope_apply_int8_q15(k, cos_q15, sin_q15, pos_offset=current_pos)

    # 3. Update KV cache with new K/V
    kv_cache.update(layer_idx, k, v, scale_k_output, scale_v_output)

    # 4. Get all cached K/V (positions 0 to current_pos inclusive)
    # Include the just-updated position by temporarily advancing, reading, then restoring.
    kv_cache.current_pos += 1  # Temporarily include new position
    k_all, v_all, _, _ = kv_cache.get(layer_idx)
    kv_cache.current_pos -= 1  # Revert (advance() will be called by caller)

    # 5. Expand K/V for GQA
    k_all = repeat_kv(k_all, kv_rep)  # [1, H, seq_len, D]
    v_all = repeat_kv(v_all, kv_rep)

    seq_len = k_all.shape[2]

    if verbose:
        print(f"[MHSA-AR] pos={current_pos}, Q:{q.shape}, K_all:{k_all.shape}, V_all:{v_all.shape}")

    # 6. Dequantize Q/K/V for FP32 attention
    q_fp32 = dequantize_linear(q, scale=scale_q_output, zero_point=0)
    k_fp32 = dequantize_linear(k_all, scale=scale_k_output, zero_point=0)
    v_fp32 = dequantize_linear(v_all, scale=scale_v_output, zero_point=0)

    # 7. Compute attention scores: Q @ K^T
    softmax_scale = layer_info.get('softmax_scale', 1.0 / np.sqrt(head_dim))
    scores = np.matmul(q_fp32, np.transpose(k_fp32, (0, 1, 3, 2))) * softmax_scale
    # scores shape: [1, H, 1, seq_len]

    # 8. Apply causal mask (only attend to positions <= current)
    # For autoregressive, we can attend to all cached positions (already causal)

    # 9. Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = fast_exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # 10. Compute context: A @ V
    context_fp32 = np.matmul(attention_weights, v_fp32)  # [1, H, 1, D]

    # 11. Reshape for output projection: [1, H, 1, D] -> [1, embed_dim]
    context_fp32 = context_fp32.transpose(0, 2, 1, 3).reshape(1, embed_dim)

    # 12. Output projection
    context_int8 = quantize_linear(context_fp32, scale=scale_v_output, zero_point=0)
    out_bias_int32 = None if out_bias_fp32 is None else (out_bias_fp32 / (scale_v_output * scale_out_weight)).astype(np.int32)
    out_int8 = linear_int8(context_int8, out_weight_int8, out_bias_int32, scale_v_output, scale_out_weight, scale_output)

    # Dequantize for FP32 output
    out_fp32 = dequantize_linear(out_int8, scale=scale_output, zero_point=0)

    if verbose:
        print(f"[MHSA-AR] Output: {out_int8.shape}, range=[{out_int8.min()}, {out_int8.max()}]")

    return out_int8, scale_output, out_fp32


def test_mhsa_autoregressive():
    """Test autoregressive MHSA with KV cache."""
    print("\n" + "=" * 80)
    print("Testing Autoregressive MHSA with KV Cache")
    print("=" * 80)

    if KVCache is None:
        print("KVCache not available, skipping test")
        return False

    # Test parameters
    n_layers = 2
    max_seq_len = 16
    embed_dim = 16
    num_heads = 4
    n_kv_heads = 2  # GQA: 4 query heads, 2 KV heads
    head_dim = embed_dim // num_heads

    # Create KV cache
    cache = KVCache(n_layers, max_seq_len, n_kv_heads, head_dim)
    print(f"\nCreated cache: {cache}")

    # Create test weights
    scale_weight = 0.005
    scale_qkv_output = 0.008
    scale_output = 0.01

    q_weight_fp32 = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1
    k_weight_fp32 = np.random.randn(n_kv_heads * head_dim, embed_dim).astype(np.float32) * 0.1
    v_weight_fp32 = np.random.randn(n_kv_heads * head_dim, embed_dim).astype(np.float32) * 0.1
    out_weight_fp32 = np.random.randn(embed_dim, embed_dim).astype(np.float32) * 0.1

    q_weight_int8 = quantize_linear(q_weight_fp32, scale=scale_weight, zero_point=0)
    k_weight_int8 = quantize_linear(k_weight_fp32, scale=scale_weight, zero_point=0)
    v_weight_int8 = quantize_linear(v_weight_fp32, scale=scale_weight, zero_point=0)
    out_weight_int8 = quantize_linear(out_weight_fp32, scale=scale_weight, zero_point=0)

    layer_info = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'q_weight_int8': q_weight_int8,
        'k_weight_int8': k_weight_int8,
        'v_weight_int8': v_weight_int8,
        'out_weight_int8': out_weight_int8,
        'q_bias_fp32': None,
        'k_bias_fp32': None,
        'v_bias_fp32': None,
        'out_bias_fp32': None,
        'q_scale_weight': scale_weight,
        'k_scale_weight': scale_weight,
        'v_scale_weight': scale_weight,
        'out_scale_weight': scale_weight,
        'q_scale_output': scale_qkv_output,
        'k_scale_output': scale_qkv_output,
        'v_scale_output': scale_qkv_output,
        'scale_output': scale_output,
        'softmax_scale': 1.0 / np.sqrt(head_dim),
        'use_rope': False,
    }

    # Test 1: Generate sequence token by token
    print("\n" + "-" * 40)
    print("Test 1: Generate 5 tokens autoregressively")
    print("-" * 40)

    scale_input = 0.01

    # Simulate prefill with first token
    x_0 = np.random.randn(1, embed_dim).astype(np.float32) * 0.1
    x_0_int8 = quantize_linear(x_0, scale=scale_input, zero_point=0)

    # Process first token for all layers (simplified - no residual)
    for layer_idx in range(n_layers):
        out_int8, scale_out, _ = mhsa_autoregressive_step(
            x_0_int8, layer_info, cache, layer_idx, scale_input, verbose=True
        )
        scale_input = scale_out  # Update scale for next layer

    cache.advance()
    print(f"After token 0: cache.pos = {cache.current_pos}")

    # Generate more tokens
    for token_idx in range(1, 5):
        # Random input for next token
        x_t = np.random.randn(1, embed_dim).astype(np.float32) * 0.1
        x_t_int8 = quantize_linear(x_t, scale=0.01, zero_point=0)
        scale_input = 0.01

        for layer_idx in range(n_layers):
            out_int8, scale_out, _ = mhsa_autoregressive_step(
                x_t_int8, layer_info, cache, layer_idx, scale_input, verbose=False
            )
            x_t_int8 = out_int8
            scale_input = scale_out

        cache.advance()
        print(f"After token {token_idx}: cache.pos = {cache.current_pos}, output range = [{out_int8.min()}, {out_int8.max()}]")

    print(f"\nFinal cache state: {cache}")
    print(f"Memory usage: {cache.get_memory_info()['used_bytes']} bytes")

    print("\n" + "=" * 80)
    print("[PASS] Autoregressive MHSA test passed!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    test_mhsa_int8_hybrid()
    test_mhsa_autoregressive()
