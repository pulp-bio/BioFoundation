# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Flip Operation - INT8

Implements sequence reversal for bidirectional Mamba models.
Used in FEMBA's MambaWrapper to process sequences in both directions.

Data Layout:
- Input: [B, L, D] or [L, D] (batch, sequence_length, features)
- Output: Same shape with sequence dimension reversed

The flip operation reverses the sequence dimension (dim=1 for 3D, dim=0 for 2D),
enabling bidirectional processing where:
1. Forward pass: mamba_fwd(x)
2. Reverse pass: mamba_rev(flip(x)) -> flip(output)
3. Combine: out_fwd + out_rev
"""

import numpy as np
from typing import Optional


def flip_sequence_int8(
    x_int8: np.ndarray,
    axis: int = 1
) -> np.ndarray:
    """
    Flip (reverse) a tensor along the sequence dimension.

    This is a pure data movement operation - no arithmetic, just reordering.
    Used in bidirectional Mamba to reverse the input sequence.

    Args:
        x_int8: Input tensor [B, L, D] or [L, D] in INT8
        axis: Axis to flip (default: 1 for sequence dim in [B, L, D])

    Returns:
        Flipped tensor with same shape and dtype

    Example:
        x = [[1, 2, 3],    # timestep 0
             [4, 5, 6],    # timestep 1
             [7, 8, 9]]    # timestep 2

        flip_sequence(x, axis=0) =
            [[7, 8, 9],    # timestep 2 -> 0
             [4, 5, 6],    # timestep 1 -> 1
             [1, 2, 3]]    # timestep 0 -> 2
    """
    return np.flip(x_int8, axis=axis).copy()


def flip_sequence_fp32(
    x: np.ndarray,
    axis: int = 1
) -> np.ndarray:
    """
    FP32 reference implementation of sequence flip.

    Args:
        x: Input tensor [B, L, D] or [L, D]
        axis: Axis to flip

    Returns:
        Flipped tensor
    """
    return np.flip(x, axis=axis).copy()


def test_flip_sequence():
    """Test flip_sequence_int8 implementation."""
    print("=" * 80)
    print("Testing Flip Sequence Operation")
    print("=" * 80)

    # Test 1: Basic 2D flip [L, D]
    print("\n--- Test 1: 2D Flip [L, D] ---")
    L, D = 4, 3
    x_2d = np.arange(L * D).reshape(L, D).astype(np.int8)
    print(f"Input shape: {x_2d.shape}")
    print(f"Input:\n{x_2d}")

    flipped_2d = flip_sequence_int8(x_2d, axis=0)
    print(f"Flipped (axis=0):\n{flipped_2d}")

    # Verify: first row should become last
    assert np.array_equal(flipped_2d[0], x_2d[-1]), "First row should be last row of input"
    assert np.array_equal(flipped_2d[-1], x_2d[0]), "Last row should be first row of input"
    print("Test 1 PASSED!")

    # Test 2: 3D flip [B, L, D]
    print("\n--- Test 2: 3D Flip [B, L, D] ---")
    B, L, D = 2, 4, 3
    x_3d = np.arange(B * L * D).reshape(B, L, D).astype(np.int8)
    print(f"Input shape: {x_3d.shape}")
    print(f"Input batch 0:\n{x_3d[0]}")

    flipped_3d = flip_sequence_int8(x_3d, axis=1)
    print(f"Flipped batch 0 (axis=1):\n{flipped_3d[0]}")

    # Verify: sequence reversed within each batch
    for b in range(B):
        assert np.array_equal(flipped_3d[b, 0], x_3d[b, -1]), f"Batch {b}: first should be last"
        assert np.array_equal(flipped_3d[b, -1], x_3d[b, 0]), f"Batch {b}: last should be first"
    print("Test 2 PASSED!")

    # Test 3: Double flip should restore original
    print("\n--- Test 3: Double Flip = Original ---")
    double_flipped = flip_sequence_int8(flipped_3d, axis=1)
    assert np.array_equal(double_flipped, x_3d), "Double flip should restore original"
    print("Test 3 PASSED!")

    # Test 4: Flip preserves INT8 range
    print("\n--- Test 4: INT8 Range Preservation ---")
    x_extreme = np.array([[-128, 0, 127], [1, -1, 100]], dtype=np.int8)
    flipped_extreme = flip_sequence_int8(x_extreme, axis=0)
    assert flipped_extreme.dtype == np.int8, "Output should be INT8"
    assert np.array_equal(flipped_extreme[0], x_extreme[1]), "Values should be preserved"
    print("Test 4 PASSED!")

    # Test 5: Memory layout (contiguous output)
    print("\n--- Test 5: Contiguous Output ---")
    x_test = np.random.randint(-128, 127, (4, 8, 16), dtype=np.int8)
    flipped_test = flip_sequence_int8(x_test, axis=1)
    assert flipped_test.flags['C_CONTIGUOUS'], "Output should be C-contiguous"
    print("Test 5 PASSED!")

    # Test 6: Simulate bidirectional Mamba pattern
    print("\n--- Test 6: Bidirectional Pattern Simulation ---")
    B, L, D = 1, 8, 4
    x = np.random.randint(-50, 50, (B, L, D), dtype=np.int8)

    # Forward "processing" (just identity for test)
    out_fwd = x.copy()

    # Reverse processing
    x_rev = flip_sequence_int8(x, axis=1)
    out_rev_internal = x_rev.copy()  # Would be mamba_rev(x_rev) in real code
    out_rev = flip_sequence_int8(out_rev_internal, axis=1)

    # Combine (in real code, would add with scale equalization)
    # For test, just verify shapes match
    assert out_fwd.shape == out_rev.shape, "Forward and reverse outputs should match shape"

    # Verify the reverse path correctly inverts
    # x_rev[t] should equal x[L-1-t]
    for t in range(L):
        assert np.array_equal(x_rev[0, t], x[0, L - 1 - t]), f"Timestep {t} mismatch in flip"

    print("Test 6 PASSED!")

    print("\n" + "=" * 80)
    print("All flip sequence tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_flip_sequence()
