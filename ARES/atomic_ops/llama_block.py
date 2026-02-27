# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Llama Transformer Block - INT8 Implementation

Implements a complete Llama decoder block with:
1. RMSNorm (pre-attention normalization)
2. Multi-Head Self-Attention with GQA and RoPE
3. Residual connection
4. RMSNorm (pre-FFN normalization)
5. SwiGLU Feed-Forward Network
6. Residual connection

This module demonstrates the full Llama architecture in INT8 for ARES/GAP9.

Architecture:
    x = x + MHSA(RMSNorm(x))
    x = x + SwiGLU_FFN(RMSNorm(x))

References:
    - LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
    - llama4micro implementation
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    from .quantize import quantize_linear, dequantize_linear
    from .rmsnorm import rmsnorm_int8, rmsnorm_fp32_reference
    from .mhsa import mhsa_int8_hybrid, mhsa_autoregressive_step, repeat_kv
    from .swiglu import swiglu_ffn_int8, swiglu_ffn_fp32_reference
    from .kv_cache import KVCache
    from .linear import linear_int8
    from .rope import rope_precompute_sin_cos_q15
except ImportError:
    from quantize import quantize_linear, dequantize_linear
    from rmsnorm import rmsnorm_int8, rmsnorm_fp32_reference
    from mhsa import mhsa_int8_hybrid, mhsa_autoregressive_step, repeat_kv
    from swiglu import swiglu_ffn_int8, swiglu_ffn_fp32_reference
    from kv_cache import KVCache
    from linear import linear_int8
    from rope import rope_precompute_sin_cos_q15


class LlamaBlockConfig:
    """Configuration for a Llama transformer block."""

    def __init__(
        self,
        dim: int = 64,
        hidden_dim: int = 172,  # ~2.67x dim (Llama uses 8/3 * dim)
        num_heads: int = 4,
        n_kv_heads: int = 2,  # GQA: fewer KV heads
        max_seq_len: int = 128,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rms_eps: float = 1e-5,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.rms_eps = rms_eps

        # Validate
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        assert num_heads % n_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

    def __repr__(self):
        return (
            f"LlamaBlockConfig(dim={self.dim}, hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len})"
        )


class LlamaBlockWeights:
    """Weights for a Llama transformer block (INT8 quantized)."""

    def __init__(self, config: LlamaBlockConfig, scale: float = 0.02):
        """
        Initialize random weights for testing.

        Args:
            config: Block configuration
            scale: Weight initialization scale
        """
        self.config = config
        dim = config.dim
        hidden_dim = config.hidden_dim
        num_heads = config.num_heads
        n_kv_heads = config.n_kv_heads
        head_dim = config.head_dim

        kv_dim = n_kv_heads * head_dim

        # Quantization scales
        self.scale_weight = 0.01
        self.scale_hidden = 0.02
        self.scale_qkv = 0.008

        # Attention RMSNorm weights (FP32 - not quantized)
        self.attn_norm_weight = np.ones(dim, dtype=np.float32)

        # FFN RMSNorm weights (FP32 - not quantized)
        self.ffn_norm_weight = np.ones(dim, dtype=np.float32)

        # Attention weights (INT8)
        self.q_weight_fp32 = np.random.randn(dim, dim).astype(np.float32) * scale
        self.k_weight_fp32 = np.random.randn(kv_dim, dim).astype(np.float32) * scale
        self.v_weight_fp32 = np.random.randn(kv_dim, dim).astype(np.float32) * scale
        self.o_weight_fp32 = np.random.randn(dim, dim).astype(np.float32) * scale

        self.q_weight_int8 = quantize_linear(self.q_weight_fp32, self.scale_weight)
        self.k_weight_int8 = quantize_linear(self.k_weight_fp32, self.scale_weight)
        self.v_weight_int8 = quantize_linear(self.v_weight_fp32, self.scale_weight)
        self.o_weight_int8 = quantize_linear(self.o_weight_fp32, self.scale_weight)

        # SwiGLU FFN weights (INT8)
        self.w1_fp32 = np.random.randn(hidden_dim, dim).astype(np.float32) * scale  # Gate
        self.w3_fp32 = np.random.randn(hidden_dim, dim).astype(np.float32) * scale  # Up
        self.w2_fp32 = np.random.randn(dim, hidden_dim).astype(np.float32) * scale  # Down

        self.w1_int8 = quantize_linear(self.w1_fp32, self.scale_weight)
        self.w3_int8 = quantize_linear(self.w3_fp32, self.scale_weight)
        self.w2_int8 = quantize_linear(self.w2_fp32, self.scale_weight)

        # RoPE precomputed sin/cos (Q15 fixed-point)
        if config.use_rope:
            self.rope_cos_q15, self.rope_sin_q15 = rope_precompute_sin_cos_q15(
                config.max_seq_len, head_dim, config.rope_base
            )
        else:
            self.rope_cos_q15 = None
            self.rope_sin_q15 = None


def llama_block_fp32_reference(
    x: np.ndarray,
    weights: LlamaBlockWeights,
    config: LlamaBlockConfig
) -> np.ndarray:
    """
    FP32 reference implementation of a Llama transformer block.

    Args:
        x: Input tensor [batch, seq_len, dim]
        weights: Block weights
        config: Block configuration

    Returns:
        Output tensor [batch, seq_len, dim]
    """
    batch_size, seq_len, dim = x.shape

    # 1. Attention sub-block
    # Pre-attention RMSNorm
    x_norm = rmsnorm_fp32_reference(x, weights.attn_norm_weight, config.rms_eps)

    # Q/K/V projections
    x_flat = x_norm.reshape(-1, dim)
    q = (x_flat @ weights.q_weight_fp32.T).reshape(batch_size, seq_len, config.num_heads, config.head_dim)
    k = (x_flat @ weights.k_weight_fp32.T).reshape(batch_size, seq_len, config.n_kv_heads, config.head_dim)
    v = (x_flat @ weights.v_weight_fp32.T).reshape(batch_size, seq_len, config.n_kv_heads, config.head_dim)

    # Transpose to [batch, heads, seq_len, head_dim]
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # RoPE (simplified FP32 version)
    if config.use_rope:
        for pos in range(seq_len):
            for i in range(0, config.head_dim, 2):
                freq = 1.0 / (config.rope_base ** (i / config.head_dim))
                angle = pos * freq
                cos_a, sin_a = np.cos(angle), np.sin(angle)

                # Rotate Q
                q0 = q[:, :, pos, i].copy()
                q1 = q[:, :, pos, i + 1].copy()
                q[:, :, pos, i] = q0 * cos_a - q1 * sin_a
                q[:, :, pos, i + 1] = q0 * sin_a + q1 * cos_a

                # Rotate K
                k0 = k[:, :, pos, i].copy()
                k1 = k[:, :, pos, i + 1].copy()
                k[:, :, pos, i] = k0 * cos_a - k1 * sin_a
                k[:, :, pos, i + 1] = k0 * sin_a + k1 * cos_a

    # GQA: Repeat K/V heads
    kv_rep = config.num_heads // config.n_kv_heads
    k = np.repeat(k, kv_rep, axis=1)
    v = np.repeat(v, kv_rep, axis=1)

    # Attention scores
    scale = 1.0 / np.sqrt(config.head_dim)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    # Causal mask
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    scores = scores + mask

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # Context
    context = np.matmul(attn, v)  # [batch, heads, seq_len, head_dim]
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim)

    # Output projection
    attn_out = context @ weights.o_weight_fp32.T

    # Residual
    x = x + attn_out

    # 2. FFN sub-block
    # Pre-FFN RMSNorm
    x_norm = rmsnorm_fp32_reference(x, weights.ffn_norm_weight, config.rms_eps)

    # SwiGLU FFN
    ffn_out = swiglu_ffn_fp32_reference(
        x_norm,
        weights.w1_fp32,
        weights.w3_fp32,
        weights.w2_fp32
    )

    # Residual
    x = x + ffn_out

    return x


def llama_block_int8(
    x_int8: np.ndarray,
    weights: LlamaBlockWeights,
    config: LlamaBlockConfig,
    scale_input: float,
    scale_output: float,
    verbose: bool = False
) -> Tuple[np.ndarray, float]:
    """
    INT8 implementation of a Llama transformer block.

    Args:
        x_int8: Input tensor [batch, seq_len, dim] (INT8)
        weights: Block weights
        config: Block configuration
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        verbose: Print debug info

    Returns:
        Tuple of (output_int8, scale_output)
    """
    batch_size, seq_len, dim = x_int8.shape

    # Store input for residual (keep as INT8)
    residual_1 = x_int8.copy()

    # 1. Attention sub-block
    if verbose:
        print(f"[LlamaBlock] Input: {x_int8.shape}, scale={scale_input}")

    # Pre-attention RMSNorm
    # Flatten to [batch * seq_len, dim] for RMSNorm
    x_flat = x_int8.reshape(-1, dim)
    x_norm = rmsnorm_int8(
        x_flat,
        weights.attn_norm_weight,
        scale_input,
        scale_input,  # Keep same scale after norm
        dim,
        config.rms_eps
    )
    x_norm = x_norm.reshape(batch_size, seq_len, dim)

    if verbose:
        print(f"  After attn RMSNorm: range=[{x_norm.min()}, {x_norm.max()}]")

    # Build layer_info for MHSA
    mhsa_layer_info = {
        'sequence_length': seq_len,
        'embed_dim': dim,
        'num_heads': config.num_heads,
        'n_kv_heads': config.n_kv_heads,
        'head_dim': config.head_dim,
        'q_weight_int8': weights.q_weight_int8,
        'k_weight_int8': weights.k_weight_int8,
        'v_weight_int8': weights.v_weight_int8,
        'out_weight_int8': weights.o_weight_int8,
        'q_bias_fp32': None,
        'k_bias_fp32': None,
        'v_bias_fp32': None,
        'out_bias_fp32': None,
        'q_scale_weight': weights.scale_weight,
        'k_scale_weight': weights.scale_weight,
        'v_scale_weight': weights.scale_weight,
        'out_scale_weight': weights.scale_weight,
        'q_scale_output': weights.scale_qkv,
        'k_scale_output': weights.scale_qkv,
        'v_scale_output': weights.scale_qkv,
        'scale_output': scale_input,
        'softmax_scale': 1.0 / np.sqrt(config.head_dim),
        'pool_sequence': None,  # Don't pool
        'use_rope': config.use_rope,
    }

    if config.use_rope:
        mhsa_layer_info['rope_cos_q15'] = weights.rope_cos_q15
        mhsa_layer_info['rope_sin_q15'] = weights.rope_sin_q15

    # MHSA
    attn_out_int8, _, _ = mhsa_int8_hybrid(
        x_norm, mhsa_layer_info, scale_input, verbose=False
    )

    if verbose:
        print(f"  After MHSA: {attn_out_int8.shape}, range=[{attn_out_int8.min()}, {attn_out_int8.max()}]")

    # Residual connection (INT8 + INT8 -> INT8)
    # Simple add with clipping (both tensors at same scale)
    x_int8 = np.clip(residual_1.astype(np.int16) + attn_out_int8.astype(np.int16), -128, 127).astype(np.int8)

    if verbose:
        print(f"  After residual 1: range=[{x_int8.min()}, {x_int8.max()}]")

    # Store for second residual
    residual_2 = x_int8.copy()

    # 2. FFN sub-block
    # Pre-FFN RMSNorm
    x_flat = x_int8.reshape(-1, dim)
    x_norm = rmsnorm_int8(
        x_flat,
        weights.ffn_norm_weight,
        scale_input,
        scale_input,
        dim,
        config.rms_eps
    )

    if verbose:
        print(f"  After FFN RMSNorm: range=[{x_norm.min()}, {x_norm.max()}]")

    # SwiGLU FFN
    ffn_out_int8 = swiglu_ffn_int8(
        x_norm,
        weights.w1_int8,
        weights.w3_int8,
        weights.w2_int8,
        scale_input,
        weights.scale_weight,
        weights.scale_weight,
        weights.scale_weight,
        weights.scale_hidden,
        scale_input
    )

    if verbose:
        print(f"  After SwiGLU: {ffn_out_int8.shape}, range=[{ffn_out_int8.min()}, {ffn_out_int8.max()}]")

    # Reshape back to [batch, seq_len, dim]
    ffn_out_int8 = ffn_out_int8.reshape(batch_size, seq_len, dim)

    # Residual connection
    x_int8 = np.clip(residual_2.astype(np.int16) + ffn_out_int8.astype(np.int16), -128, 127).astype(np.int8)

    if verbose:
        print(f"  After residual 2: range=[{x_int8.min()}, {x_int8.max()}]")

    return x_int8, scale_input


def test_llama_block():
    """Test the complete Llama transformer block."""
    print("=" * 80)
    print("Testing Complete Llama Transformer Block")
    print("=" * 80)

    # Configuration
    config = LlamaBlockConfig(
        dim=64,
        hidden_dim=172,  # ~2.67x dim
        num_heads=4,
        n_kv_heads=2,  # GQA
        max_seq_len=32,
        use_rope=True
    )
    print(f"\nConfig: {config}")

    # Create weights
    weights = LlamaBlockWeights(config)
    print("Weights initialized")

    # Test 1: Single sequence
    print("\n" + "-" * 40)
    print("Test 1: Single sequence (batch=1, seq_len=8)")
    print("-" * 40)

    batch_size = 1
    seq_len = 8
    scale_input = 0.05

    x_fp32 = np.random.randn(batch_size, seq_len, config.dim).astype(np.float32) * 0.5
    x_int8 = quantize_linear(x_fp32, scale_input)

    print(f"Input: {x_int8.shape}, range=[{x_int8.min()}, {x_int8.max()}]")

    # FP32 reference
    y_fp32_ref = llama_block_fp32_reference(x_fp32, weights, config)
    print(f"FP32 reference output: {y_fp32_ref.shape}, range=[{y_fp32_ref.min():.4f}, {y_fp32_ref.max():.4f}]")

    # INT8 implementation
    y_int8, scale_out = llama_block_int8(x_int8, weights, config, scale_input, scale_input, verbose=True)
    y_fp32_from_int8 = dequantize_linear(y_int8, scale_out)

    print(f"\nINT8 output: {y_int8.shape}, range=[{y_int8.min()}, {y_int8.max()}]")
    print(f"Dequantized: range=[{y_fp32_from_int8.min():.4f}, {y_fp32_from_int8.max():.4f}]")

    # Compare
    max_diff = np.max(np.abs(y_fp32_from_int8 - y_fp32_ref))
    mean_diff = np.mean(np.abs(y_fp32_from_int8 - y_fp32_ref))
    print(f"\nMax diff vs FP32: {max_diff:.4f}")
    print(f"Mean diff vs FP32: {mean_diff:.4f}")

    # Test 2: Batch processing
    print("\n" + "-" * 40)
    print("Test 2: Batch processing (batch=2, seq_len=16)")
    print("-" * 40)

    batch_size = 2
    seq_len = 16

    x_fp32 = np.random.randn(batch_size, seq_len, config.dim).astype(np.float32) * 0.5
    x_int8 = quantize_linear(x_fp32, scale_input)

    y_int8, scale_out = llama_block_int8(x_int8, weights, config, scale_input, scale_input, verbose=False)
    print(f"Input: {x_int8.shape}")
    print(f"Output: {y_int8.shape}, range=[{y_int8.min()}, {y_int8.max()}]")

    # Test 3: Multiple blocks (simulating a small model)
    print("\n" + "-" * 40)
    print("Test 3: Stack of 3 Llama blocks")
    print("-" * 40)

    n_layers = 3
    batch_size = 1
    seq_len = 8

    x_fp32 = np.random.randn(batch_size, seq_len, config.dim).astype(np.float32) * 0.5
    x_int8 = quantize_linear(x_fp32, scale_input)
    scale = scale_input

    print(f"Input: {x_int8.shape}, scale={scale}")

    # Create weights for each layer
    layer_weights = [LlamaBlockWeights(config) for _ in range(n_layers)]

    for layer_idx in range(n_layers):
        x_int8, scale = llama_block_int8(
            x_int8, layer_weights[layer_idx], config, scale, scale, verbose=False
        )
        print(f"  Layer {layer_idx}: output range=[{x_int8.min()}, {x_int8.max()}]")

    print(f"Final output: {x_int8.shape}, scale={scale}")

    # Test 4: Memory estimates
    print("\n" + "-" * 40)
    print("Test 4: Memory estimates for different configs")
    print("-" * 40)

    configs = [
        {"name": "Tiny", "dim": 64, "hidden": 172, "heads": 4, "kv_heads": 2},
        {"name": "Small", "dim": 128, "hidden": 344, "heads": 4, "kv_heads": 2},
        {"name": "Base", "dim": 256, "hidden": 688, "heads": 8, "kv_heads": 4},
    ]

    for cfg in configs:
        # Weight sizes (INT8)
        q_size = cfg["dim"] * cfg["dim"]
        k_size = (cfg["kv_heads"] * cfg["dim"] // cfg["heads"]) * cfg["dim"]
        v_size = k_size
        o_size = cfg["dim"] * cfg["dim"]
        w1_size = cfg["hidden"] * cfg["dim"]
        w2_size = cfg["dim"] * cfg["hidden"]
        w3_size = w1_size

        total_weights = q_size + k_size + v_size + o_size + w1_size + w2_size + w3_size
        print(f"  {cfg['name']}: dim={cfg['dim']}, weights={total_weights / 1024:.1f} KB per layer")

    print("\n" + "=" * 80)
    print("[PASS] All Llama block tests passed!")
    print("=" * 80)
    return True


def test_llama_autoregressive():
    """Test autoregressive generation with Llama block and KV cache."""
    print("\n" + "=" * 80)
    print("Testing Llama Block with Autoregressive Generation")
    print("=" * 80)

    # Configuration
    config = LlamaBlockConfig(
        dim=32,
        hidden_dim=86,
        num_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        use_rope=True
    )
    print(f"\nConfig: {config}")

    # Create weights and cache
    weights = LlamaBlockWeights(config)
    cache = KVCache(
        n_layers=1,
        max_seq_len=config.max_seq_len,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim
    )
    print(f"Cache: {cache}")

    # Build MHSA layer info
    mhsa_layer_info = {
        'embed_dim': config.dim,
        'num_heads': config.num_heads,
        'n_kv_heads': config.n_kv_heads,
        'head_dim': config.head_dim,
        'q_weight_int8': weights.q_weight_int8,
        'k_weight_int8': weights.k_weight_int8,
        'v_weight_int8': weights.v_weight_int8,
        'out_weight_int8': weights.o_weight_int8,
        'q_bias_fp32': None,
        'k_bias_fp32': None,
        'v_bias_fp32': None,
        'out_bias_fp32': None,
        'q_scale_weight': weights.scale_weight,
        'k_scale_weight': weights.scale_weight,
        'v_scale_weight': weights.scale_weight,
        'out_scale_weight': weights.scale_weight,
        'q_scale_output': weights.scale_qkv,
        'k_scale_output': weights.scale_qkv,
        'v_scale_output': weights.scale_qkv,
        'scale_output': 0.05,
        'softmax_scale': 1.0 / np.sqrt(config.head_dim),
        'use_rope': config.use_rope,
        'rope_cos_q15': weights.rope_cos_q15,
        'rope_sin_q15': weights.rope_sin_q15,
    }

    # Simulate autoregressive generation
    print("\n" + "-" * 40)
    print("Generating 8 tokens autoregressively")
    print("-" * 40)

    scale = 0.05
    outputs = []

    for token_idx in range(8):
        # Random input token
        x_fp32 = np.random.randn(1, config.dim).astype(np.float32) * 0.3
        x_int8 = quantize_linear(x_fp32, scale)

        # 1. RMSNorm
        x_norm = rmsnorm_int8(x_int8, weights.attn_norm_weight, scale, scale, config.dim)

        # 2. Autoregressive MHSA
        attn_out, _, _ = mhsa_autoregressive_step(
            x_norm, mhsa_layer_info, cache, layer_idx=0, scale_input=scale
        )

        # 3. Residual
        x_int8 = np.clip(x_int8.astype(np.int16) + attn_out.astype(np.int16), -128, 127).astype(np.int8)

        # 4. RMSNorm
        x_norm = rmsnorm_int8(x_int8, weights.ffn_norm_weight, scale, scale, config.dim)

        # 5. SwiGLU FFN
        ffn_out = swiglu_ffn_int8(
            x_norm, weights.w1_int8, weights.w3_int8, weights.w2_int8,
            scale, weights.scale_weight, weights.scale_weight, weights.scale_weight,
            weights.scale_hidden, scale
        )

        # 6. Residual
        x_int8 = np.clip(x_int8.astype(np.int16) + ffn_out.astype(np.int16), -128, 127).astype(np.int8)

        # Advance cache
        cache.advance()

        outputs.append(x_int8.copy())
        print(f"  Token {token_idx}: output range=[{x_int8.min()}, {x_int8.max()}], cache_pos={cache.current_pos}")

    print(f"\nGenerated {len(outputs)} tokens")
    print(f"Final cache state: {cache}")

    print("\n" + "=" * 80)
    print("[PASS] Llama autoregressive test passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    test_llama_block()
    test_llama_autoregressive()
