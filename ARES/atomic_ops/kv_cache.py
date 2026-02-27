# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache for Autoregressive LLM Inference - INT8

Implements Key-Value caching for efficient autoregressive generation.
The cache stores computed K/V projections to avoid redundant computation
when generating tokens one at a time.

Usage:
    cache = KVCache(n_layers=6, max_seq_len=256, n_kv_heads=4, head_dim=64)

    # Prefill phase (process prompt)
    for layer_idx in range(n_layers):
        k, v = compute_kv(prompt, layer_idx)
        cache.prefill(layer_idx, k, v, k_scale, v_scale)

    # Generation phase (one token at a time)
    for step in range(max_new_tokens):
        for layer_idx in range(n_layers):
            k_t, v_t = compute_kv(x_t, layer_idx)
            cache.update(layer_idx, k_t, v_t)
            k_all, v_all = cache.get(layer_idx)
            # ... attention with k_all, v_all
        cache.advance()
"""

import numpy as np
from typing import Tuple, Optional


class KVCache:
    """
    INT8 Key-Value cache for autoregressive LLM generation.

    Stores K/V projections for all layers, enabling efficient single-token
    generation by reusing previously computed values.

    Attributes:
        n_layers: Number of transformer layers
        max_seq_len: Maximum sequence length supported
        n_kv_heads: Number of KV heads (for GQA, may be < num_query_heads)
        head_dim: Dimension per attention head
        current_pos: Current position in the sequence (0-indexed)
    """

    def __init__(
        self,
        n_layers: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: np.dtype = np.int8
    ):
        """
        Initialize the KV cache.

        Args:
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length to cache
            n_kv_heads: Number of KV heads (supports GQA when < num_query_heads)
            head_dim: Dimension per attention head
            dtype: Data type for cache (default np.int8)
        """
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Pre-allocate cache: [n_layers, max_seq_len, n_kv_heads, head_dim]
        cache_shape = (n_layers, max_seq_len, n_kv_heads, head_dim)
        self.key_cache = np.zeros(cache_shape, dtype=dtype)
        self.value_cache = np.zeros(cache_shape, dtype=dtype)

        # Per-layer quantization scales
        self.k_scales = np.ones(n_layers, dtype=np.float32)
        self.v_scales = np.ones(n_layers, dtype=np.float32)

        # Current position (next position to write)
        self.current_pos = 0

        # Track if cache has been initialized
        self._initialized = False

    @property
    def seq_len(self) -> int:
        """Current sequence length (number of cached positions)."""
        return self.current_pos

    @property
    def kv_dim(self) -> int:
        """Total KV dimension (n_kv_heads * head_dim)."""
        return self.n_kv_heads * self.head_dim

    @property
    def cache_size_bytes(self) -> int:
        """Total cache size in bytes."""
        element_size = np.dtype(self.dtype).itemsize
        return 2 * self.n_layers * self.max_seq_len * self.kv_dim * element_size

    def reset(self) -> None:
        """Reset cache for a new sequence."""
        self.current_pos = 0
        self.key_cache.fill(0)
        self.value_cache.fill(0)
        self.k_scales.fill(1.0)
        self.v_scales.fill(1.0)
        self._initialized = False

    def prefill(
        self,
        layer_idx: int,
        k: np.ndarray,
        v: np.ndarray,
        k_scale: float = 1.0,
        v_scale: float = 1.0
    ) -> None:
        """
        Prefill cache with K/V from prompt processing (batch mode).

        This is called during the initial prompt processing phase where
        all prompt tokens are processed at once.

        Args:
            layer_idx: Layer index (0 to n_layers-1)
            k: Key tensor [batch=1, n_kv_heads, seq_len, head_dim]
            v: Value tensor [batch=1, n_kv_heads, seq_len, head_dim]
            k_scale: Quantization scale for K
            v_scale: Quantization scale for V
        """
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"K/V must be 4D [B, H, N, D], got K:{k.shape}, V:{v.shape}")

        batch_size, n_heads, seq_len, d = k.shape
        if batch_size != 1:
            raise ValueError(f"Batch size must be 1, got {batch_size}")
        if n_heads != self.n_kv_heads:
            raise ValueError(f"Expected {self.n_kv_heads} KV heads, got {n_heads}")
        if d != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got {d}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.max_seq_len}")

        # Store in cache: transpose from [1, H, N, D] to [N, H, D]
        # Cache layout: [layer, pos, n_kv_heads, head_dim]
        self.key_cache[layer_idx, :seq_len] = k[0].transpose(1, 0, 2)  # [N, H, D]
        self.value_cache[layer_idx, :seq_len] = v[0].transpose(1, 0, 2)

        self.k_scales[layer_idx] = k_scale
        self.v_scales[layer_idx] = v_scale

        # Update position only on first layer (all layers have same seq_len)
        if layer_idx == 0:
            self.current_pos = seq_len
            self._initialized = True

    def update(
        self,
        layer_idx: int,
        k: np.ndarray,
        v: np.ndarray,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None
    ) -> None:
        """
        Update cache with K/V for a single new token.

        This is called during autoregressive generation, once per layer
        for each new token.

        Args:
            layer_idx: Layer index (0 to n_layers-1)
            k: Key tensor [batch=1, n_kv_heads, seq_len=1, head_dim]
            v: Value tensor [batch=1, n_kv_heads, seq_len=1, head_dim]
            k_scale: Quantization scale for K (optional, uses existing if not provided)
            v_scale: Quantization scale for V (optional, uses existing if not provided)
        """
        if self.current_pos >= self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: pos={self.current_pos} >= max={self.max_seq_len}"
            )

        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"K/V must be 4D [B, H, N, D], got K:{k.shape}, V:{v.shape}")

        batch_size, n_heads, seq_len, d = k.shape
        if batch_size != 1:
            raise ValueError(f"Batch size must be 1, got {batch_size}")
        if seq_len != 1:
            raise ValueError(f"Expected seq_len=1 for update, got {seq_len}")
        if n_heads != self.n_kv_heads:
            raise ValueError(f"Expected {self.n_kv_heads} KV heads, got {n_heads}")
        if d != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got {d}")

        # Store at current position: [1, H, 1, D] -> [H, D]
        self.key_cache[layer_idx, self.current_pos] = k[0, :, 0, :]
        self.value_cache[layer_idx, self.current_pos] = v[0, :, 0, :]

        # Update scales if provided
        if k_scale is not None:
            self.k_scales[layer_idx] = k_scale
        if v_scale is not None:
            self.v_scales[layer_idx] = v_scale

    def get(
        self,
        layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Get all cached K/V up to current position.

        Returns all positions from 0 to current_pos-1 (i.e., current_pos positions).
        current_pos represents the number of filled positions after prefill/update+advance.

        Args:
            layer_idx: Layer index (0 to n_layers-1)

        Returns:
            Tuple of (k, v, k_scale, v_scale) where:
                k: Key tensor [1, n_kv_heads, seq_len, head_dim]
                v: Value tensor [1, n_kv_heads, seq_len, head_dim]
                k_scale: Quantization scale for K
                v_scale: Quantization scale for V
        """
        end_pos = self.current_pos
        if end_pos == 0:
            raise RuntimeError("Cache is empty, call prefill() or update() first")

        # Get cached values: [pos, H, D] -> [1, H, pos, D]
        k = self.key_cache[layer_idx, :end_pos]  # [pos, H, D]
        v = self.value_cache[layer_idx, :end_pos]

        # Transpose to attention format: [pos, H, D] -> [1, H, pos, D]
        k = k.transpose(1, 0, 2)[np.newaxis]  # [1, H, pos, D]
        v = v.transpose(1, 0, 2)[np.newaxis]

        return k, v, self.k_scales[layer_idx], self.v_scales[layer_idx]

    def advance(self) -> None:
        """
        Advance position after generating a token.

        Call this after processing all layers for the current token,
        before starting the next token.
        """
        self.current_pos += 1
        if self.current_pos > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow after advance: pos={self.current_pos} > max={self.max_seq_len}"
            )

    def get_memory_info(self) -> dict:
        """Get memory usage information."""
        element_size = np.dtype(self.dtype).itemsize
        total_elements = 2 * self.n_layers * self.max_seq_len * self.kv_dim
        used_elements = 2 * self.n_layers * self.current_pos * self.kv_dim

        return {
            'total_bytes': total_elements * element_size,
            'used_bytes': used_elements * element_size,
            'utilization': self.current_pos / self.max_seq_len if self.max_seq_len > 0 else 0,
            'dtype': str(self.dtype),
            'shape': {
                'n_layers': self.n_layers,
                'max_seq_len': self.max_seq_len,
                'n_kv_heads': self.n_kv_heads,
                'head_dim': self.head_dim,
            },
            'current_pos': self.current_pos,
        }

    def __repr__(self) -> str:
        return (
            f"KVCache(n_layers={self.n_layers}, max_seq_len={self.max_seq_len}, "
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"pos={self.current_pos}, size={self.cache_size_bytes // 1024}KB)"
        )


def test_kv_cache():
    """Unit tests for KVCache."""
    print("=" * 80)
    print("Testing KVCache")
    print("=" * 80)

    # Test parameters
    n_layers = 2
    max_seq_len = 16
    n_kv_heads = 4
    head_dim = 8

    # Create cache
    cache = KVCache(n_layers, max_seq_len, n_kv_heads, head_dim)
    print(f"\nCreated: {cache}")
    print(f"Memory info: {cache.get_memory_info()}")

    # Test 1: Prefill with prompt
    print("\n" + "-" * 40)
    print("Test 1: Prefill with prompt (seq_len=5)")
    print("-" * 40)

    prompt_len = 5
    k_prompt = np.random.randint(-128, 128, size=(1, n_kv_heads, prompt_len, head_dim), dtype=np.int8)
    v_prompt = np.random.randint(-128, 128, size=(1, n_kv_heads, prompt_len, head_dim), dtype=np.int8)

    for layer_idx in range(n_layers):
        cache.prefill(layer_idx, k_prompt, v_prompt, k_scale=0.01, v_scale=0.01)

    print(f"After prefill: pos={cache.current_pos}, seq_len={cache.seq_len}")
    assert cache.current_pos == prompt_len, f"Expected pos={prompt_len}, got {cache.current_pos}"

    # Verify prefill data
    k_retrieved, v_retrieved, k_scale, v_scale = cache.get(0)
    print(f"Retrieved K shape: {k_retrieved.shape}")
    print(f"Retrieved V shape: {v_retrieved.shape}")
    assert k_retrieved.shape == (1, n_kv_heads, prompt_len, head_dim), \
        f"Expected {(1, n_kv_heads, prompt_len, head_dim)}, got {k_retrieved.shape}"
    assert v_retrieved.shape == (1, n_kv_heads, prompt_len, head_dim)

    # Verify data matches (k_retrieved should match k_prompt)
    assert np.array_equal(k_retrieved, k_prompt), "Prefill K data mismatch"
    assert np.array_equal(v_retrieved, v_prompt), "Prefill V data mismatch"
    print("Prefill data verified!")

    # Test 2: Single token update
    print("\n" + "-" * 40)
    print("Test 2: Single token update")
    print("-" * 40)

    k_new = np.random.randint(-128, 128, size=(1, n_kv_heads, 1, head_dim), dtype=np.int8)
    v_new = np.random.randint(-128, 128, size=(1, n_kv_heads, 1, head_dim), dtype=np.int8)

    for layer_idx in range(n_layers):
        cache.update(layer_idx, k_new, v_new)

    cache.advance()
    print(f"After update+advance: pos={cache.current_pos}")
    assert cache.current_pos == prompt_len + 1

    # Retrieve and verify
    k_all, v_all, _, _ = cache.get(0)
    print(f"Retrieved K shape after update: {k_all.shape}")
    assert k_all.shape == (1, n_kv_heads, prompt_len + 1, head_dim)

    # Verify new token is at the right position
    assert np.array_equal(k_all[0, :, prompt_len, :], k_new[0, :, 0, :]), "New K not at correct position"
    print("Update data verified!")

    # Test 3: Multiple token generation
    print("\n" + "-" * 40)
    print("Test 3: Multiple token generation (5 tokens)")
    print("-" * 40)

    for i in range(5):
        k_gen = np.random.randint(-128, 128, size=(1, n_kv_heads, 1, head_dim), dtype=np.int8)
        v_gen = np.random.randint(-128, 128, size=(1, n_kv_heads, 1, head_dim), dtype=np.int8)

        for layer_idx in range(n_layers):
            cache.update(layer_idx, k_gen, v_gen)
        cache.advance()

        print(f"  Generated token {i+1}: pos={cache.current_pos}")

    expected_len = prompt_len + 1 + 5  # prefill + first update + 5 more
    assert cache.current_pos == expected_len
    print(f"Final sequence length: {cache.seq_len}")

    # Test 4: Reset
    print("\n" + "-" * 40)
    print("Test 4: Reset cache")
    print("-" * 40)

    cache.reset()
    print(f"After reset: pos={cache.current_pos}")
    assert cache.current_pos == 0
    assert not cache._initialized
    print("Reset verified!")

    # Test 5: Memory estimates
    print("\n" + "-" * 40)
    print("Test 5: Memory estimates for different configs")
    print("-" * 40)

    configs = [
        {"n_layers": 2, "max_seq_len": 256, "n_kv_heads": 4, "head_dim": 64, "name": "Llama-tiny"},
        {"n_layers": 6, "max_seq_len": 256, "n_kv_heads": 4, "head_dim": 64, "name": "Llama-small"},
        {"n_layers": 12, "max_seq_len": 512, "n_kv_heads": 8, "head_dim": 64, "name": "Llama-medium"},
    ]

    for cfg in configs:
        test_cache = KVCache(cfg["n_layers"], cfg["max_seq_len"], cfg["n_kv_heads"], cfg["head_dim"])
        size_kb = test_cache.cache_size_bytes / 1024
        print(f"  {cfg['name']}: {size_kb:.1f} KB")

    print("\n" + "=" * 80)
    print("[PASS] All KVCache tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    test_kv_cache()
