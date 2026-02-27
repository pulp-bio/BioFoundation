#!/usr/bin/env python3
# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for integer-only softmax (i-Softmax)

Tests the i_softmax_int16() implementation against standard FP32 softmax
to validate accuracy and correctness.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from atomic_ops.mhsa import i_softmax_int16, load_softmax_lut
from tools.generate_softmax_lut import generate_softmax_lut


def fp32_softmax(x, axis=-1):
    """Standard FP32 softmax for reference."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def test_i_softmax_basic():
    """Test i-Softmax on simple inputs."""
    print("\n" + "="*60)
    print("TEST 1: Basic i-Softmax functionality")
    print("="*60)

    # Generate LUT
    lut, lut_metadata = generate_softmax_lut(
        input_range=(-8.0, 0.0),
        num_entries=256,
        output_scale=32767.0
    )

    # Test case 1: Simple scores
    scores = np.array([
        [-2.0, -1.0, -0.5, 0.0],
        [-4.0, -3.0, -2.0, -1.0]
    ], dtype=np.float32)

    print(f"\nInput scores:\n{scores}")

    # FP32 reference
    softmax_fp32 = fp32_softmax(scores, axis=-1)
    print(f"\nFP32 softmax:\n{softmax_fp32}")

    # i-Softmax
    softmax_i_fp32, softmax_i16 = i_softmax_int16(scores, lut, lut_metadata, axis=-1)
    print(f"\ni-Softmax (FP32):\n{softmax_i_fp32}")
    print(f"\ni-Softmax (INT16):\n{softmax_i16}")

    # Compare
    abs_error = np.abs(softmax_fp32 - softmax_i_fp32)
    max_abs_error = np.max(abs_error)
    mean_abs_error = np.mean(abs_error)

    print(f"\nAbsolute Error:")
    print(f"  Max:  {max_abs_error:.6f}")
    print(f"  Mean: {mean_abs_error:.6f}")

    # Validation
    # Relax thresholds for LUT-based approximation (0.3-0.7% mean error is acceptable)
    assert max_abs_error < 0.01, f"Max error too large: {max_abs_error}"
    assert mean_abs_error < 0.005, f"Mean error too large: {mean_abs_error}"

    print("[PASS] PASSED: Basic i-Softmax test")
    return True


def test_i_softmax_attention_scores():
    """Test i-Softmax on realistic attention scores."""
    print("\n" + "="*60)
    print("TEST 2: Realistic attention scores")
    print("="*60)

    # Generate LUT
    lut, lut_metadata = generate_softmax_lut(
        input_range=(-8.0, 0.0),
        num_entries=256,
        output_scale=32767.0
    )

    # Simulate attention scores from test_8_mhsa_basic
    # embed_dim=32, num_heads=4, seq_len=16, head_dim=8
    # Typical scores range: [-5, 0] after scaling
    batch_size = 1
    num_heads = 4
    seq_len = 16

    np.random.seed(42)
    scores = np.random.uniform(-5.0, 0.0, (batch_size, num_heads, seq_len, seq_len)).astype(np.float32)

    print(f"\nScores shape: {scores.shape}")
    print(f"Scores range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")

    # FP32 reference
    softmax_fp32 = fp32_softmax(scores, axis=-1)

    # i-Softmax
    softmax_i_fp32, softmax_i16 = i_softmax_int16(scores, lut, lut_metadata, axis=-1)

    # Validate sum to 1
    fp32_sums = np.sum(softmax_fp32, axis=-1)
    i16_sums = np.sum(softmax_i_fp32, axis=-1)

    print(f"\nFP32 softmax sums (should be 1.0):")
    print(f"  Min: {np.min(fp32_sums):.6f}, Max: {np.max(fp32_sums):.6f}")
    print(f"\ni-Softmax sums (should be ~1.0):")
    print(f"  Min: {np.min(i16_sums):.6f}, Max: {np.max(i16_sums):.6f}")

    # Compare
    abs_error = np.abs(softmax_fp32 - softmax_i_fp32)
    max_abs_error = np.max(abs_error)
    mean_abs_error = np.mean(abs_error)
    rel_error = np.abs((softmax_fp32 - softmax_i_fp32) / (softmax_fp32 + 1e-10)) * 100

    print(f"\nAbsolute Error:")
    print(f"  Max:  {max_abs_error:.6f}")
    print(f"  Mean: {mean_abs_error:.6f}")
    print(f"\nRelative Error:")
    print(f"  Max:  {np.max(rel_error):.3f}%")
    print(f"  Mean: {np.mean(rel_error):.3f}%")

    # Validation
    assert max_abs_error < 0.02, f"Max error too large: {max_abs_error}"
    assert mean_abs_error < 0.005, f"Mean error too large: {mean_abs_error}"
    assert np.allclose(i16_sums, 1.0, atol=0.01), f"Softmax doesn't sum to 1: {i16_sums}"

    print("[PASS] PASSED: Realistic attention scores test")
    return True


def test_i_softmax_edge_cases():
    """Test i-Softmax on edge cases."""
    print("\n" + "="*60)
    print("TEST 3: Edge cases")
    print("="*60)

    # Generate LUT
    lut, lut_metadata = generate_softmax_lut(
        input_range=(-8.0, 0.0),
        num_entries=256,
        output_scale=32767.0
    )

    # Edge case 1: All zeros
    print("\nEdge case 1: All zeros")
    scores_zeros = np.zeros((2, 4), dtype=np.float32)
    softmax_i_fp32, _ = i_softmax_int16(scores_zeros, lut, lut_metadata, axis=-1)
    expected = np.ones_like(scores_zeros) / 4.0  # Uniform distribution
    assert np.allclose(softmax_i_fp32, expected, atol=0.01), "All zeros failed"
    print("  [PASS] Passed")

    # Edge case 2: Large negative values (should clip to LUT min)
    print("\nEdge case 2: Large negative values")
    scores_large_neg = np.array([[-50.0, -40.0, -30.0, -0.1]], dtype=np.float32)
    softmax_i_fp32, _ = i_softmax_int16(scores_large_neg, lut, lut_metadata, axis=-1)
    # Last value should dominate (exp(-0.1) >> exp(-30))
    assert softmax_i_fp32[0, -1] > 0.9, f"Large negative failed: {softmax_i_fp32}"
    print(f"  Result: {softmax_i_fp32}")
    print("  [PASS] Passed")

    # Edge case 3: Uniform distribution
    print("\nEdge case 3: Uniform distribution")
    scores_uniform = np.full((2, 8), -2.0, dtype=np.float32)
    softmax_i_fp32, _ = i_softmax_int16(scores_uniform, lut, lut_metadata, axis=-1)
    expected = np.ones_like(scores_uniform) / 8.0
    assert np.allclose(softmax_i_fp32, expected, atol=0.01), "Uniform failed"
    print(f"  Result: {softmax_i_fp32[0]}")
    print("  [PASS] Passed")

    # Edge case 4: One hot (one value dominates)
    print("\nEdge case 4: One-hot distribution")
    scores_onehot = np.array([[-10.0, -10.0, 0.0, -10.0]], dtype=np.float32)
    softmax_i_fp32, _ = i_softmax_int16(scores_onehot, lut, lut_metadata, axis=-1)
    # Third value should be ~1.0
    assert softmax_i_fp32[0, 2] > 0.99, f"One-hot failed: {softmax_i_fp32}"
    print(f"  Result: {softmax_i_fp32}")
    print("  [PASS] Passed")

    print("\n[PASS] PASSED: All edge cases")
    return True


def test_i_softmax_vs_fast_exp():
    """Compare i-Softmax with fast_exp() softmax."""
    print("\n" + "="*60)
    print("TEST 4: i-Softmax vs fast_exp() softmax")
    print("="*60)

    from atomic_ops.mhsa import fast_exp

    # Generate LUT
    lut, lut_metadata = generate_softmax_lut(
        input_range=(-8.0, 0.0),
        num_entries=256,
        output_scale=32767.0
    )

    # Test scores
    np.random.seed(123)
    scores = np.random.uniform(-5.0, 0.0, (1, 4, 16, 16)).astype(np.float32)

    # fast_exp softmax (current GAP9 implementation)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = fast_exp(scores - scores_max)
    softmax_fast_exp = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # i-Softmax
    softmax_i_fp32, _ = i_softmax_int16(scores, lut, lut_metadata, axis=-1)

    # Compare
    abs_error = np.abs(softmax_fast_exp - softmax_i_fp32)
    max_abs_error = np.max(abs_error)
    mean_abs_error = np.mean(abs_error)

    print(f"\nAbsolute Error (i-Softmax vs fast_exp):")
    print(f"  Max:  {max_abs_error:.6f}")
    print(f"  Mean: {mean_abs_error:.6f}")

    # This comparison shows how much i-Softmax differs from fast_exp
    # They should be close but not identical (different approximations)
    assert max_abs_error < 0.05, f"Max difference too large: {max_abs_error}"
    print("[PASS] PASSED: i-Softmax and fast_exp() are reasonably close")
    return True


def test_i_softmax_multidimensional():
    """Test i-Softmax on multidimensional tensors."""
    print("\n" + "="*60)
    print("TEST 5: Multidimensional tensors")
    print("="*60)

    # Generate LUT
    lut, lut_metadata = generate_softmax_lut(
        input_range=(-8.0, 0.0),
        num_entries=256,
        output_scale=32767.0
    )

    # Test on various shapes
    shapes = [
        (4,),           # 1D
        (3, 4),         # 2D
        (2, 3, 4),      # 3D
        (2, 3, 4, 5),   # 4D
    ]

    for shape in shapes:
        print(f"\nTesting shape: {shape}")
        scores = np.random.uniform(-5.0, 0.0, shape).astype(np.float32)

        # FP32 reference
        softmax_fp32 = fp32_softmax(scores, axis=-1)

        # i-Softmax
        softmax_i_fp32, _ = i_softmax_int16(scores, lut, lut_metadata, axis=-1)

        # Validate
        abs_error = np.abs(softmax_fp32 - softmax_i_fp32)
        max_abs_error = np.max(abs_error)

        assert max_abs_error < 0.02, f"Shape {shape} failed: max_error={max_abs_error}"
        print(f"  Max error: {max_abs_error:.6f} [PASS]")

    print("\n[PASS] PASSED: Multidimensional tensors")
    return True


def run_all_tests():
    """Run all i-Softmax tests."""
    print("\n" + "="*60)
    print("I-SOFTMAX VALIDATION TEST SUITE")
    print("="*60)

    tests = [
        test_i_softmax_basic,
        test_i_softmax_attention_scores,
        test_i_softmax_edge_cases,
        test_i_softmax_vs_fast_exp,
        test_i_softmax_multidimensional,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[FAIL] ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[PASS] ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n[FAIL] {failed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
