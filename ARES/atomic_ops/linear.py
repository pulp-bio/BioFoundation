# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Atomic Linear (Fully Connected) Operation - INT8

Implements matrix multiplication with TRUE INT8 arithmetic.

Similar to Conv2D but uses matrix multiply instead of convolution.
INT8 x INT8 → INT32 accumulation → Rescale → INT8 output
"""

import numpy as np

# Handle both module import and standalone execution
try:
    from .quantize import quantize_linear, dequantize_linear
except ImportError:
    from quantize import quantize_linear, dequantize_linear


def linear_int8(
    x_int8: np.ndarray,
    w_int8: np.ndarray,
    bias_int32: np.ndarray = None,
    scale_x: float = 1.0,
    scale_w: float = 1.0,
    scale_y: float = 1.0,
    zero_point_x: int = 0,
    zero_point_w: int = 0,
    zero_point_y: int = 0
) -> np.ndarray:
    """
    INT8 fully connected (linear) layer with TRUE INT8 arithmetic.

    Formula:
        Y = X @ W^T + bias

    Where:
        - X: [B, in_features] INT8 input
        - W: [out_features, in_features] INT8 weights
        - bias: [out_features] INT32 bias
        - Y: [B, out_features] INT8 output

    Quantization-aware computation:
        1. INT8 x INT8 → INT32 accumulation (matmul)
        2. Add bias (INT32)
        3. Rescale: scale_combined = (scale_x * scale_w) / scale_y
        4. Requantize to INT8

    Args:
        x_int8: Input INT8 tensor [B, in_features]
        w_int8: Weight INT8 tensor [out_features, in_features]
        bias_int32: Bias INT32 tensor [out_features] (optional)
        scale_x: Input quantization scale
        scale_w: Weight quantization scale
        scale_y: Output quantization scale
        zero_point_x: Input zero point (default 0 for symmetric)
        zero_point_w: Weight zero point (default 0 for symmetric)
        zero_point_y: Output zero point (default 0 for symmetric)

    Returns:
        Output INT8 tensor [B, out_features]

    Notes:
        - Uses INT32 accumulation to avoid overflow
        - Rescaling factor: (scale_x * scale_w) / scale_y
        - For symmetric quantization: all zero_points = 0
    """
    # Ensure input is 2D [B, in_features]
    if x_int8.ndim == 1:
        x_int8 = x_int8.reshape(1, -1)

    batch_size, in_features = x_int8.shape
    out_features = w_int8.shape[0]

    # Step 1: Matrix multiplication with INT32 accumulation
    # Y = X @ W^T
    # [B, in_features] @ [in_features, out_features] = [B, out_features]
    output_int32 = x_int8.astype(np.int32) @ w_int8.T.astype(np.int32)

    # Step 2: Add bias (if provided)
    if bias_int32 is not None:
        # Broadcast bias to [B, out_features]
        output_int32 += bias_int32.astype(np.int32)

    # Step 3: Rescale
    # Combined scale: (scale_x * scale_w) / scale_y
    scale_combined = (scale_x * scale_w) / scale_y
    output_fp32 = output_int32.astype(np.float32) * scale_combined

    # Step 4: Requantize to INT8
    output_int8 = quantize_linear(
        output_fp32,
        scale=1.0,  # Already rescaled
        zero_point=zero_point_y,
        qmin=-128,
        qmax=127,
        dtype=np.int8
    )

    return output_int8


def linear_fp32_reference(
    x: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray = None
) -> np.ndarray:
    """
    FP32 Linear layer reference implementation.

    Used for testing and verification.

    Args:
        x: Input FP32 tensor [B, in_features]
        w: Weight FP32 tensor [out_features, in_features]
        bias: Bias FP32 tensor [out_features] (optional)

    Returns:
        Output FP32 tensor [B, out_features]
    """
    # Ensure input is 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Y = X @ W^T
    output = x @ w.T

    # Add bias
    if bias is not None:
        output += bias

    return output


def test_linear():
    """Test INT8 Linear implementation."""
    print("="*80)
    print("Testing INT8 Linear (Fully Connected)")
    print("="*80)

    # Test 1: Basic linear layer
    print("\nTest 1: Basic Linear Layer")

    # Create simple inputs
    x_fp32 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)  # [1, 4]
    w_fp32 = np.array([
        [0.5, -0.5, 0.5, -0.5],  # out neuron 1
        [0.1, 0.2, 0.3, 0.4],    # out neuron 2
    ], dtype=np.float32)  # [2, 4]
    bias_fp32 = np.array([1.0, -1.0], dtype=np.float32)  # [2]

    # Quantize
    scale_x = 0.1
    scale_w = 0.01
    scale_y = 0.05

    x_int8 = quantize_linear(x_fp32, scale=scale_x)
    w_int8 = quantize_linear(w_fp32, scale=scale_w)

    # Bias quantization: scale_bias = scale_x * scale_w
    scale_bias = scale_x * scale_w
    bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)

    # Run INT8 linear
    output_int8 = linear_int8(
        x_int8, w_int8, bias_int32,
        scale_x, scale_w, scale_y
    )

    # Run FP32 reference
    output_fp32_ref = linear_fp32_reference(x_fp32, w_fp32, bias_fp32)
    output_int8_expected = quantize_linear(output_fp32_ref, scale=scale_y)

    print(f"Input FP32:  {x_fp32}")
    print(f"Input INT8:  {x_int8}")
    print(f"Weight INT8 shape: {w_int8.shape}")
    print(f"Bias INT32:  {bias_int32}")
    print(f"Output INT8: {output_int8}")
    print(f"Expected:    {output_int8_expected}")
    print(f"Output FP32 ref: {output_fp32_ref}")

    # Allow small difference due to quantization rounding
    diff = np.abs(output_int8.astype(np.int32) - output_int8_expected.astype(np.int32))
    max_diff = np.max(diff)
    assert max_diff <= 1, f"Difference too large: {max_diff}"
    print(f"[PASS] Test 1 passed! (max diff: {max_diff})")

    # Test 2: Batch processing
    print("\n" + "="*80)
    print("Test 2: Batch Processing")
    print("="*80)

    batch_size = 4
    in_features = 10
    out_features = 5

    # Random FP32 inputs
    x_fp32 = np.random.randn(batch_size, in_features).astype(np.float32)
    w_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
    bias_fp32 = np.random.randn(out_features).astype(np.float32)

    # Quantize
    scale_x = 0.1
    scale_w = 0.01
    scale_y = 0.05

    x_int8 = quantize_linear(x_fp32, scale=scale_x)
    w_int8 = quantize_linear(w_fp32, scale=scale_w)

    scale_bias = scale_x * scale_w
    bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)

    # Run INT8 linear
    output_int8 = linear_int8(
        x_int8, w_int8, bias_int32,
        scale_x, scale_w, scale_y
    )

    # Run FP32 reference
    output_fp32_ref = linear_fp32_reference(x_fp32, w_fp32, bias_fp32)
    output_int8_expected = quantize_linear(output_fp32_ref, scale=scale_y)

    print(f"Input shape:    {x_int8.shape}")
    print(f"Weight shape:   {w_int8.shape}")
    print(f"Output shape:   {output_int8.shape}")
    print(f"Expected shape: {output_int8_expected.shape}")

    # Check shapes
    assert output_int8.shape == (batch_size, out_features), f"Shape mismatch: {output_int8.shape}"

    # Check values
    diff = np.abs(output_int8.astype(np.int32) - output_int8_expected.astype(np.int32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference:  {max_diff}")
    print(f"Mean difference: {mean_diff:.2f}")

    assert max_diff <= 2, f"Difference too large: {max_diff}"
    print(f"[PASS] Test 2 passed!")

    # Test 3: Without bias
    print("\n" + "="*80)
    print("Test 3: Linear Without Bias")
    print("="*80)

    x_fp32 = np.random.randn(2, 8).astype(np.float32)
    w_fp32 = np.random.randn(4, 8).astype(np.float32) * 0.1

    scale_x = 0.1
    scale_w = 0.01
    scale_y = 0.05

    x_int8 = quantize_linear(x_fp32, scale=scale_x)
    w_int8 = quantize_linear(w_fp32, scale=scale_w)

    # Run without bias
    output_int8 = linear_int8(
        x_int8, w_int8, bias_int32=None,
        scale_x=scale_x, scale_w=scale_w, scale_y=scale_y
    )

    # Run FP32 reference
    output_fp32_ref = linear_fp32_reference(x_fp32, w_fp32, bias=None)
    output_int8_expected = quantize_linear(output_fp32_ref, scale=scale_y)

    print(f"Input shape:  {x_int8.shape}")
    print(f"Output shape: {output_int8.shape}")
    print(f"Output INT8 sample: {output_int8[0, :3]}")
    print(f"Expected sample:    {output_int8_expected[0, :3]}")

    diff = np.abs(output_int8.astype(np.int32) - output_int8_expected.astype(np.int32))
    max_diff = np.max(diff)

    assert max_diff <= 2, f"Difference too large: {max_diff}"
    print(f"[PASS] Test 3 passed! (max diff: {max_diff})")

    # Test 4: Large layer (similar to SimpleCNN classifier)
    print("\n" + "="*80)
    print("Test 4: Large Layer (1568 → 10)")
    print("="*80)

    # Simulate SimpleCNN classifier: flatten output (7x7x32=1568) → 10 classes
    batch_size = 2
    in_features = 1568
    out_features = 10

    x_fp32 = np.random.randn(batch_size, in_features).astype(np.float32) * 0.5
    w_fp32 = np.random.randn(out_features, in_features).astype(np.float32) * 0.01
    bias_fp32 = np.random.randn(out_features).astype(np.float32) * 0.1

    scale_x = 0.1
    scale_w = 0.005
    scale_y = 0.05

    x_int8 = quantize_linear(x_fp32, scale=scale_x)
    w_int8 = quantize_linear(w_fp32, scale=scale_w)

    scale_bias = scale_x * scale_w
    bias_int32 = np.round(bias_fp32 / scale_bias).astype(np.int32)

    # Run INT8 linear
    output_int8 = linear_int8(
        x_int8, w_int8, bias_int32,
        scale_x, scale_w, scale_y
    )

    print(f"Input shape:  {x_int8.shape}")
    print(f"Weight shape: {w_int8.shape}")
    print(f"Output shape: {output_int8.shape}")
    print(f"Output INT8 sample: {output_int8[0]}")

    assert output_int8.shape == (batch_size, out_features), f"Shape mismatch"
    print(f"[PASS] Test 4 passed!")

    # Test 5: Compare with Conv2D formula (should be similar)
    print("\n" + "="*80)
    print("Test 5: Verify Rescaling Formula")
    print("="*80)

    # Small example to verify math
    x_fp32 = np.array([[2.0, 3.0]], dtype=np.float32)  # [1, 2]
    w_fp32 = np.array([[1.0, 1.0]], dtype=np.float32)  # [1, 2]

    scale_x = 0.1
    scale_w = 0.1
    scale_y = 0.1

    x_int8 = quantize_linear(x_fp32, scale=scale_x)  # [20, 30]
    w_int8 = quantize_linear(w_fp32, scale=scale_w)  # [10, 10]

    print(f"x_fp32: {x_fp32} → x_int8: {x_int8}")
    print(f"w_fp32: {w_fp32} → w_int8: {w_int8}")

    # Manual calculation
    # matmul: 20*10 + 30*10 = 200 + 300 = 500 (INT32)
    # rescale: 500 * (0.1 * 0.1 / 0.1) = 500 * 0.1 = 50.0
    # quantize: round(50.0 / 1.0) = 50 → clip to 127 → 50
    expected_int32 = 500
    expected_rescaled = 50.0
    expected_int8 = 50

    output_int8 = linear_int8(
        x_int8, w_int8, bias_int32=None,
        scale_x=scale_x, scale_w=scale_w, scale_y=scale_y
    )

    print(f"Expected INT32 accumulation: {expected_int32}")
    print(f"Expected after rescale: {expected_rescaled}")
    print(f"Expected INT8 output: {expected_int8}")
    print(f"Actual INT8 output: {output_int8[0, 0]}")

    assert output_int8[0, 0] == expected_int8, f"Math mismatch: {output_int8[0, 0]} != {expected_int8}"
    print(f"[PASS] Test 5 passed! Rescaling formula verified!")

    print("\n" + "="*80)
    print("[PASS] All Linear tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_linear()
