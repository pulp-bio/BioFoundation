# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

import numpy as np

try:
    from .constants import INT16_MAX_Q15, LUT_SIZE
except ImportError:
    from constants import INT16_MAX_Q15, LUT_SIZE

# ---
# i-LayerNorm: Integer-only LayerNorm with LUT for Bit-Exact Matching
# ---
# For bit-exact LayerNorm, we need to eliminate FP32 variations in:
# 1. Mean computation (use INT64 accumulation + integer division)
# 2. Variance computation (use INT64 accumulation + integer division)
# 3. sqrt computation (use LUT-based inverse sqrt)
#
# This approach uses integer arithmetic for mean/variance and a LUT for 1/sqrt
# ---

# LUT for 1/sqrt(x) - stores inverted scaled sqrt for variance values
# Variance in INT8 domain typically ranges from 0 to ~16000 (for normalized_dim up to 256)
# We store 1/sqrt(x) * SCALE for lookup
I_LAYERNORM_ISQRT_ENTRIES = LUT_SIZE  # 12-bit index
I_LAYERNORM_ISQRT_SCALE = INT16_MAX_Q15   # INT16 output scale
I_LAYERNORM_VAR_MAX = 16384       # Max expected variance (covers typical ranges)


def get_builtin_layernorm_isqrt_lut() -> tuple:
    """
    Generate the builtin inverse sqrt lookup table for bit-exact LayerNorm.

    The LUT stores 1/sqrt(var+1) * SCALE for variance values from 0 to VAR_MAX.

    Returns:
        tuple: (lut, metadata) where:
            - lut: INT16 numpy array of LUT values
            - metadata: dict with LUT parameters
    """
    # Generate variance values from 0 to VAR_MAX
    var_indices = np.arange(I_LAYERNORM_ISQRT_ENTRIES)

    # Map index to variance: var = (index / ENTRIES) * VAR_MAX
    # index 0 -> var = 0, index ENTRIES-1 -> var = VAR_MAX
    var_values = (var_indices / (I_LAYERNORM_ISQRT_ENTRIES - 1)) * I_LAYERNORM_VAR_MAX

    # Add epsilon (1 in integer domain) and compute 1/sqrt
    # Note: var_values + 1 to avoid division by zero
    isqrt_values = 1.0 / np.sqrt(var_values + 1.0)

    # Scale and store as INT16
    lut = np.round(isqrt_values * I_LAYERNORM_ISQRT_SCALE).astype(np.int16)

    metadata = {
        'num_entries': I_LAYERNORM_ISQRT_ENTRIES,
        'var_max': I_LAYERNORM_VAR_MAX,
        'output_scale': float(I_LAYERNORM_ISQRT_SCALE),
    }

    return lut, metadata


def layernorm_int8_lut(input_int8, weight_fp32, bias_fp32, scale_input, scale_output,
                       normalized_shape, eps=1e-5, isqrt_lut=None, lut_metadata=None):
    """
    INT8 LayerNorm using LUT-based inverse sqrt for bit-exact matching with C.

    This version uses:
    - INT64 accumulation for mean and variance
    - Integer division for mean and variance
    - LUT-based 1/sqrt for the normalization step

    Args:
        input_int8: INT8 input tensor
        weight_fp32: FP32 weight (gamma)
        bias_fp32: FP32 bias (beta)
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        normalized_shape: Int, size of dimension to normalize
        eps: Epsilon (unused, integer epsilon = 1 is implicit)
        isqrt_lut: Optional INT16 inverse sqrt LUT
        lut_metadata: Optional metadata dict

    Returns:
        output_int8: INT8 output tensor
    """
    if isqrt_lut is None or lut_metadata is None:
        isqrt_lut, lut_metadata = get_builtin_layernorm_isqrt_lut()

    num_entries = lut_metadata['num_entries']
    var_max = lut_metadata['var_max']
    output_scale = lut_metadata['output_scale']

    # Reshape input for easier processing
    original_shape = input_int8.shape
    transpose_perm = None

    if len(original_shape) == 1:
        input_reshaped = input_int8.reshape(1, -1)
    else:
        if original_shape[-1] == normalized_shape:
            batch_size = int(np.prod(original_shape[:-1]))
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)
        else:
            # Find matching dimension
            matching_dims = [i for i, dim in enumerate(original_shape) if dim == normalized_shape]
            if len(matching_dims) == 0:
                raise ValueError(
                    f"LayerNorm: normalized_shape={normalized_shape} not found in input shape {original_shape}"
                )
            if len(matching_dims) == 1:
                axis = matching_dims[0]
                transpose_perm = list(range(len(original_shape)))
                transpose_perm.append(transpose_perm.pop(axis))
                input_int8_transposed = np.transpose(input_int8, transpose_perm)
                batch_size = int(np.prod(input_int8_transposed.shape[:-1]))
                input_reshaped = input_int8_transposed.reshape(batch_size, normalized_shape)
            else:
                batch_size = int(np.prod(original_shape[:-1]))
                input_reshaped = input_int8.reshape(batch_size, normalized_shape)

    # Process each vector
    num_vectors = input_reshaped.shape[0]
    output_fp32 = np.zeros((num_vectors, normalized_shape), dtype=np.float32)

    for v in range(num_vectors):
        input_vec = input_reshaped[v]

        # Step 1: Compute mean using INT64 accumulation and INTEGER DIVISION
        sum_val = np.int64(0)
        for i in range(normalized_shape):
            sum_val += np.int64(input_vec[i])
        # IMPORTANT: Match C integer division semantics (truncate toward zero).
        # Python's `//` is floor-division for negatives, which does NOT match C.
        if sum_val >= 0:
            mean = sum_val // np.int64(normalized_shape)
        else:
            mean = -((-sum_val) // np.int64(normalized_shape))

        # Step 2: Compute variance using INT64 accumulation and INTEGER DIVISION
        var_sum = np.int64(0)
        for i in range(normalized_shape):
            diff = np.int64(input_vec[i]) - mean
            var_sum += diff * diff
        variance = var_sum // np.int64(normalized_shape)

        # Step 3: Look up 1/sqrt(variance + 1) from LUT
        # Map variance to LUT index
        var_clamped = min(int(variance), var_max)
        idx = int(round(var_clamped * (num_entries - 1) / var_max))
        if idx < 0:
            idx = 0
        if idx >= num_entries:
            idx = num_entries - 1

        isqrt_int16 = int(isqrt_lut[idx])

        # Step 4: Normalize and apply affine transform
        for i in range(normalized_shape):
            # Centered value (integer)
            x_centered = np.int64(input_vec[i]) - mean

            # Normalize: x_norm = x_centered * isqrt / SCALE
            # First multiply, then divide for integer arithmetic
            x_norm_scaled = x_centered * isqrt_int16
            x_norm_fp32 = float(x_norm_scaled) / output_scale

            # Convert to original FP32 domain
            x_norm_fp32 *= scale_input

            # Apply affine transform
            output_fp32[v, i] = weight_fp32[i] * x_norm_fp32 + bias_fp32[i]

    # Reshape back to original shape
    if transpose_perm is not None:
        transposed_shape = tuple(original_shape[i] for i in transpose_perm)
        output_fp32 = output_fp32.reshape(transposed_shape)
        inverse_perm = [0] * len(transpose_perm)
        for i, p in enumerate(transpose_perm):
            inverse_perm[p] = i
        output_fp32 = np.transpose(output_fp32, inverse_perm)
    else:
        output_fp32 = output_fp32.reshape(original_shape)

    # Quantize to INT8
    output_int8 = np.clip(np.round(output_fp32 / scale_output), -128, 127).astype(np.int8)

    return output_int8


def i_sqrt_newton(n: np.ndarray) -> np.ndarray:
    """
    I-BERT Algorithm 4: Integer-only Square Root using Newton-Raphson.

    Computes floor(sqrt(n)) using Newton-Raphson iteration with only integer
    arithmetic. This matches the I-BERT paper exactly.

    Algorithm:
        1. Initialize x0 = 2^(ceil(bits(n)/2))
        2. Iterate: x_{i+1} = (x_i + n/x_i) / 2
        3. Converges within 5 iterations for INT32 inputs

    Reference:
        "I-BERT: Integer-only BERT Quantization" (Kim et al., 2021), Algorithm 4

    Args:
        n: Input integer array (non-negative INT32)

    Returns:
        floor(sqrt(n)) as INT32
    """
    n = np.atleast_1d(np.asarray(n, dtype=np.int32))
    res = np.zeros_like(n)

    mask = n > 0
    if not np.any(mask):
        return res if res.shape != () else int(res)

    n_pos = n[mask]

    # Initialization: x0 = 2^(ceil(bits(n)/2))
    msb_pos = np.floor(np.log2(n_pos.astype(np.float64))).astype(np.int32) + 1
    exponent = -(-msb_pos // 2)  # Ceiling division
    x = (1 << exponent).astype(np.int32)

    # Newton-Raphson iteration (5 iterations sufficient for INT32)
    for _ in range(5):
        div = n_pos // x
        x_new = (x + div) >> 1

        not_converged = x_new < x
        if not np.any(not_converged):
            break

        x = np.where(not_converged, x_new, x)

    res[mask] = x
    return res if res.shape != () else int(res[0])


def sqrt_q64(number, frac_bits=0):
    """
    Fixed-point square root using binary search (Q-format support).

    This function computes sqrt(x) using binary search on INT64 values.
    It's bit-exact reproducible and matches the C implementation exactly.

    Algorithm: Binary search for y such that (y*y) >> frac_bits == number

    Note: For pure integer sqrt, i_sqrt_newton() provides an alternative
    implementation matching I-BERT Algorithm 4 exactly. Both methods
    produce identical floor(sqrt(n)) results.

    Based on PULP-NN library (University of Bologna)
    Original author: Moritz Scherer

    Args:
        number: INT64 input value
        frac_bits: Number of fractional bits in Q-format (0 for integer)

    Returns:
        INT64 square root result
    """
    number = int(number)
    root = 0

    start = 0
    end = 46342  # smallest integer that is larger than sqrt(0x7FFFFFFF)

    if number > 0:
        while start <= end:
            mid = (start + end) >> 1

            mid_squared = (mid * mid) >> frac_bits

            if mid_squared == number:
                root = mid
                break

            if mid_squared < number:
                start = mid + 1
                root = mid
            else:
                end = mid - 1

        return root
    else:
        return 0

def sqrt_approx_python(x):
    """
    Python implementation matching C sqrt_approx EXACTLY.

    This uses the same bit manipulation and Newton-Raphson iterations as C
    to ensure IDENTICAL results in Python and C for LayerNorm computation.

    CRITICAL: This must stay synchronized with sqrt_approx() in network_kernels.c.mako
    """
    import struct
    import numpy as np

    x = np.float32(x)  # Force FP32 precision
    if x <= 0.0:
        return np.float32(0.0)
    if x == 1.0:
        return np.float32(1.0)

    # Choose initial guess (matching C implementation exactly)
    if x < np.float32(0.01):
        guess = x
    else:
        # Bit manipulation approximation (matching C exactly)
        # C code: conv.i = (1 << 29) + (conv.i >> 1) - (1 << 22)
        i_bytes = struct.pack('f', x)
        i = struct.unpack('I', i_bytes)[0]
        i = (1 << 29) + (i >> 1) - (1 << 22)
        guess_bytes = struct.pack('I', i)
        guess = np.float32(struct.unpack('f', guess_bytes)[0])

    # Newton-Raphson refinement (12 iterations to match C)
    # Using FP32 arithmetic throughout
    for _ in range(12):
        if guess > np.float32(0.0):
            guess = np.float32(0.5) * (guess + x / guess)

    return float(guess)  # Convert back to Python float for compatibility

def layernorm_int8_fixed_point(input_int8, weight_fp32, bias_fp32, scale_input, scale_output,
                                normalized_shape, eps=1e-5):
    """
    INT8 LayerNorm operation using fixed-point arithmetic (bit-exact with C)

    This version uses INT64 accumulation and binary search sqrt for bit-exact
    reproducibility between Python and C implementations.

    Formula: output = gamma * (x - mean) / sqrt(variance + eps) + beta

    Key differences from FP32 version:
    - INT64 accumulation for mean/variance (prevents overflow, no FP rounding)
    - Binary search sqrt (sqrt_q64) instead of np.sqrt()
    - Integer division for mean/variance
    - Integer epsilon (eps = 1 in INT8 domain)

    Args:
        input_int8: INT8 input tensor, shape (..., normalized_shape)
        weight_fp32: FP32 weight (gamma), shape (normalized_shape,)
        bias_fp32: FP32 bias (beta), shape (normalized_shape,)
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        normalized_shape: Int, size of the dimension to normalize
        eps: Small constant for numerical stability (converted to integer domain)

    Returns:
        output_int8: INT8 output tensor, same shape as input
    """
    # Step 1: Reshape input for easier processing
    original_shape = input_int8.shape
    transpose_perm = None

    # Verify that normalized_shape matches expected dimension
    if len(original_shape) == 1:
        input_reshaped = input_int8.reshape(1, -1)
    else:
        if original_shape[-1] == normalized_shape:
            batch_size = np.prod(original_shape[:-1])
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)
        else:
            matching_dims = [i for i, dim in enumerate(original_shape) if dim == normalized_shape]
            if len(matching_dims) == 0:
                raise ValueError(
                    f"LayerNorm: normalized_shape={normalized_shape} not found in input shape {original_shape}"
                )
            if len(matching_dims) == 1:
                axis = matching_dims[0]
                transpose_perm = list(range(len(original_shape)))
                transpose_perm.append(transpose_perm.pop(axis))
                input_int8_transposed = np.transpose(input_int8, transpose_perm)
                batch_size = np.prod(input_int8_transposed.shape[:-1])
                input_reshaped = input_int8_transposed.reshape(batch_size, normalized_shape)
            else:
                batch_size = np.prod(original_shape[:-1])
                input_reshaped = input_int8.reshape(batch_size, normalized_shape)

    # Step 2-5: Process each vector with fixed-point arithmetic
    num_vectors = input_reshaped.shape[0]
    output_fp32 = np.zeros((num_vectors, normalized_shape), dtype=np.float32)

    for v in range(num_vectors):
        input_vec_int8 = input_reshaped[v]  # INT8 values

        # Step 2a: Compute mean using INT64 accumulation
        sum_val = np.int64(0)
        for i in range(normalized_shape):
            sum_val += np.int64(input_vec_int8[i])
        # IMPORTANT: Match C integer division semantics (truncate toward zero).
        # Python's `//` is floor-division for negatives, which does NOT match C.
        if sum_val >= 0:
            mean = sum_val // np.int64(normalized_shape)
        else:
            mean = -((-sum_val) // np.int64(normalized_shape))

        # Step 2b: Compute variance using INT64 accumulation
        var_sum = np.int64(0)
        for i in range(normalized_shape):
            diff = np.int64(input_vec_int8[i]) - mean
            var_sum += diff * diff
        variance = var_sum // np.int64(normalized_shape)  # Integer division

        # Step 3: Add epsilon and compute std using binary search
        # eps in INT8 domain = 1 (matches C implementation)
        variance += 1  # Integer epsilon
        std = sqrt_q64(variance, frac_bits=0)

        # Step 4: Normalize and apply affine transform
        for i in range(normalized_shape):
            # Compute normalized value in integer domain
            x_centered = np.int64(input_vec_int8[i]) - mean

            # Convert to FP32 for affine transform
            x_normalized = float(x_centered) / float(std)

            # Apply scale_input to convert to original FP32 domain
            x_normalized *= scale_input

            # Apply affine transform (gamma * normalized + beta)
            output_fp32[v, i] = weight_fp32[i] * x_normalized + bias_fp32[i]

    # Step 5: Reshape back to original shape
    if transpose_perm is not None:
        transposed_shape = tuple(original_shape[i] for i in transpose_perm)
        output_fp32 = output_fp32.reshape(transposed_shape)
        inverse_perm = [0] * len(transpose_perm)
        for i, p in enumerate(transpose_perm):
            inverse_perm[p] = i
        output_fp32 = np.transpose(output_fp32, inverse_perm)
    else:
        output_fp32 = output_fp32.reshape(original_shape)

    # Step 6: Quantize to INT8
    output_int8 = np.clip(np.round(output_fp32 / scale_output), -128, 127).astype(np.int8)

    return output_int8


def layernorm_int8(input_int8, weight_fp32, bias_fp32, scale_input, scale_output,
                   normalized_shape, eps=1e-5):
    """
    INT8 LayerNorm operation.

    LayerNorm normalizes across the last dimension(s) specified by normalized_shape.
    Formula: output = gamma * (x - mean) / sqrt(variance + eps) + beta

    Args:
        input_int8: INT8 input tensor, shape (..., normalized_shape)
        weight_fp32: FP32 weight (gamma), shape (normalized_shape,)
        bias_fp32: FP32 bias (beta), shape (normalized_shape,)
        scale_input: Input quantization scale
        scale_output: Output quantization scale
        normalized_shape: Int, size of the dimension to normalize
        eps: Small constant for numerical stability

    Returns:
        output_int8: INT8 output tensor, same shape as input

    Implementation:
        1. Dequantize input to FP32
        2. Compute mean and variance along normalized dimension
        3. Apply normalization: (x - mean) / sqrt(var + eps)
        4. Apply affine transform: gamma * normalized + beta
        5. Quantize back to INT8
    """
    # Step 1: Reshape INT8 input for easier processing (DO NOT dequantize yet!)
    # We process INT8 values and dequantize in each loop to match C implementation exactly
    original_shape = input_int8.shape
    transpose_perm = None  # Track if we need to transpose back

    # Verify that normalized_shape matches expected dimension
    # LayerNorm normalizes over the last dimension
    if len(original_shape) == 1:
        # Already 1D
        input_reshaped = input_int8.reshape(1, -1)
    else:
        # Check if last dimension matches normalized_shape
        if original_shape[-1] == normalized_shape:
            # Standard case: last dimension is the normalized dimension
            batch_size = np.prod(original_shape[:-1])
            input_reshaped = input_int8.reshape(batch_size, normalized_shape)
        else:
            # Handle transposed case: normalized_shape might be elsewhere
            matching_dims = [i for i, dim in enumerate(original_shape) if dim == normalized_shape]

            if len(matching_dims) == 0:
                raise ValueError(
                    f"LayerNorm: normalized_shape={normalized_shape} not found in input shape {original_shape}. "
                    f"Input must have a dimension matching normalized_shape."
                )

            if len(matching_dims) == 1:
                # Exactly one dimension matches - transpose to make it last
                axis = matching_dims[0]
                transpose_perm = list(range(len(original_shape)))
                transpose_perm.append(transpose_perm.pop(axis))
                input_int8_transposed = np.transpose(input_int8, transpose_perm)
                batch_size = np.prod(input_int8_transposed.shape[:-1])
                input_reshaped = input_int8_transposed.reshape(batch_size, normalized_shape)
            else:
                # Multiple dimensions match - ambiguous, use last one
                batch_size = np.prod(original_shape[:-1])
                input_reshaped = input_int8.reshape(batch_size, normalized_shape)

    # Step 2-5: Process each vector with C-style sequential loops
    # This matches the EXACT computation order of the C implementation to ensure
    # bit-exact results. We dequantize INT8→FP32 in each loop iteration, just like C.

    num_vectors = input_reshaped.shape[0]
    output_fp32 = np.zeros((num_vectors, normalized_shape), dtype=np.float32)

    for v in range(num_vectors):
        input_vec_int8 = input_reshaped[v]  # INT8 values

        # Step 2a: Compute mean (dequantize INT8→FP32 in loop, matching C)
        # Force FP32 precision to match C implementation
        sum_val = np.float32(0.0)
        for i in range(normalized_shape):
            sum_val += np.float32(float(input_vec_int8[i]) * scale_input)
        mean = np.float32(sum_val / np.float32(normalized_shape))

        # Step 2b: Compute variance (dequantize INT8→FP32 in loop, matching C)
        var_sum = np.float32(0.0)
        for i in range(normalized_shape):
            x = np.float32(float(input_vec_int8[i]) * scale_input)
            diff = np.float32(x - mean)
            var_sum += np.float32(diff * diff)
        variance = np.float32(var_sum / np.float32(normalized_shape))

        # Step 3: Compute std with epsilon
        # Use sqrt_approx_python to match C implementation exactly
        std = sqrt_approx_python(variance + eps)

        # Step 4: Normalize and apply affine transform (dequantize INT8→FP32 in loop, matching C)
        for i in range(normalized_shape):
            x = float(input_vec_int8[i]) * scale_input
            normalized_val = (x - mean) / std
            output_fp32[v, i] = weight_fp32[i] * normalized_val + bias_fp32[i]

    # Step 5: Reshape back to original shape
    if transpose_perm is not None:
        # We transposed the input, so output_fp32 is in transposed shape
        # Reshape to transposed shape first
        transposed_shape = tuple(original_shape[i] for i in transpose_perm)
        output_fp32 = output_fp32.reshape(transposed_shape)
        # Then transpose back to original order
        inverse_perm = [0] * len(transpose_perm)
        for i, p in enumerate(transpose_perm):
            inverse_perm[p] = i
        output_fp32 = np.transpose(output_fp32, inverse_perm)
    else:
        output_fp32 = output_fp32.reshape(original_shape)

    # Step 6: Quantize to INT8 (matching C's quantize_fp32_value)
    output_int8 = np.clip(np.round(output_fp32 / scale_output), -128, 127).astype(np.int8)

    return output_int8


def test_layernorm_int8():
    """Unit test for INT8 LayerNorm"""
    print("="*80)
    print("Testing INT8 LayerNorm")
    print("="*80)

    # Test case 1: Simple 1D LayerNorm
    print("\nTest 1: 1D LayerNorm (8 features)")
    normalized_shape = 8

    # Create FP32 input
    input_fp32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    # Quantize to INT8
    scale_input = 0.1
    input_int8 = np.clip(np.round(input_fp32 / scale_input), -128, 127).astype(np.int8)

    # LayerNorm parameters (simple: weight=1, bias=0)
    weight_fp32 = np.ones(normalized_shape, dtype=np.float32)
    bias_fp32 = np.zeros(normalized_shape, dtype=np.float32)

    # Output scale (similar to input)
    scale_output = 0.1

    # Apply INT8 LayerNorm
    output_int8 = layernorm_int8(
        input_int8, weight_fp32, bias_fp32,
        scale_input, scale_output, normalized_shape
    )

    # Dequantize for comparison
    output_fp32 = output_int8.astype(np.float32) * scale_output

    # Compute reference FP32 LayerNorm
    mean = np.mean(input_fp32)
    variance = np.var(input_fp32)
    reference_fp32 = (input_fp32 - mean) / np.sqrt(variance + 1e-5)

    # Compare
    print(f"Input (FP32):     {input_fp32}")
    print(f"Input (INT8):     {input_int8}")
    print(f"Output (INT8):    {output_int8}")
    print(f"Output (FP32):    {output_fp32}")
    print(f"Reference (FP32): {reference_fp32}")
    print(f"Max diff:         {np.max(np.abs(output_fp32 - reference_fp32)):.6f}")

    # Test case 2: 2D LayerNorm (batch of 3, 8 features)
    print("\nTest 2: 2D LayerNorm (batch=3, features=8)")
    input_fp32 = np.random.randn(3, 8).astype(np.float32)
    input_int8 = np.clip(np.round(input_fp32 / scale_input), -128, 127).astype(np.int8)

    # Apply INT8 LayerNorm
    output_int8 = layernorm_int8(
        input_int8, weight_fp32, bias_fp32,
        scale_input, scale_output, normalized_shape
    )

    # Dequantize
    output_fp32 = output_int8.astype(np.float32) * scale_output

    # Reference FP32 LayerNorm (row-wise)
    mean = np.mean(input_fp32, axis=-1, keepdims=True)
    variance = np.var(input_fp32, axis=-1, keepdims=True)
    reference_fp32 = (input_fp32 - mean) / np.sqrt(variance + 1e-5)

    print(f"Input shape:      {input_fp32.shape}")
    print(f"Output shape:     {output_fp32.shape}")
    print(f"Max diff:         {np.max(np.abs(output_fp32 - reference_fp32)):.6f}")
    print(f"Mean diff:        {np.mean(np.abs(output_fp32 - reference_fp32)):.6f}")

    # Test case 3: With learned weight and bias
    print("\nTest 3: LayerNorm with learned weight and bias")
    weight_fp32 = np.random.randn(normalized_shape).astype(np.float32) * 0.5 + 1.0
    bias_fp32 = np.random.randn(normalized_shape).astype(np.float32) * 0.1

    # Apply INT8 LayerNorm
    output_int8 = layernorm_int8(
        input_int8, weight_fp32, bias_fp32,
        scale_input, scale_output, normalized_shape
    )

    # Dequantize
    output_fp32 = output_int8.astype(np.float32) * scale_output

    # Reference FP32 LayerNorm
    mean = np.mean(input_fp32, axis=-1, keepdims=True)
    variance = np.var(input_fp32, axis=-1, keepdims=True)
    normalized = (input_fp32 - mean) / np.sqrt(variance + 1e-5)
    reference_fp32 = weight_fp32 * normalized + bias_fp32

    print(f"Weight (gamma):   {weight_fp32}")
    print(f"Bias (beta):      {bias_fp32}")
    print(f"Max diff:         {np.max(np.abs(output_fp32 - reference_fp32)):.6f}")
    print(f"Mean diff:        {np.mean(np.abs(output_fp32 - reference_fp32)):.6f}")

    # Check if errors are within quantization tolerance
    max_diff = np.max(np.abs(output_fp32 - reference_fp32))
    tolerance = max(scale_input, scale_output) * 2  # 2x quantization step

    if max_diff <= tolerance:
        print(f"\n[OK] All tests passed! (max diff {max_diff:.6f} <= tolerance {tolerance:.6f})")
    else:
        print(f"\n[FAIL] Test failed! (max diff {max_diff:.6f} > tolerance {tolerance:.6f})")

    print("="*80)


def test_sqrt_q64():
    """Unit test for INT64 binary search sqrt"""
    print("="*80)
    print("Testing INT64 Binary Search Sqrt (sqrt_q64)")
    print("="*80)

    # Test case 1: Perfect squares
    print("\nTest 1: Perfect squares")
    test_values = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 225, 400, 625, 900, 1024]
    for val in test_values:
        result = sqrt_q64(val, frac_bits=0)
        expected = int(np.sqrt(val))
        print(f"sqrt({val:4d}) = {result:3d}, expected {expected:3d}, match: {result == expected}")
        assert result == expected, f"Failed for {val}: got {result}, expected {expected}"

    # Test case 2: Non-perfect squares
    print("\nTest 2: Non-perfect squares")
    test_values = [2, 3, 5, 7, 10, 15, 20, 50, 99, 200, 500, 1000, 2000]
    for val in test_values:
        result = sqrt_q64(val, frac_bits=0)
        expected = int(np.sqrt(val))
        error = abs(result - np.sqrt(val))
        print(f"sqrt({val:4d}) = {result:3d}, expected ~{expected:3d}, error: {error:.4f}")
        # Binary search finds floor(sqrt(x)), so result should be <= sqrt(val) < result+1
        assert result * result <= val < (result + 1) * (result + 1), f"Failed for {val}"

    # Test case 3: Large values
    print("\nTest 3: Large values")
    test_values = [10000, 50000, 100000, 500000, 1000000, 2000000000]
    for val in test_values:
        result = sqrt_q64(val, frac_bits=0)
        expected = int(np.sqrt(val))
        error = abs(result - np.sqrt(val))
        print(f"sqrt({val:10d}) = {result:6d}, expected ~{expected:6d}, error: {error:.4f}")
        assert result * result <= val < (result + 1) * (result + 1), f"Failed for {val}"

    # Test case 4: Edge cases
    print("\nTest 4: Edge cases")
    assert sqrt_q64(0, frac_bits=0) == 0, "Failed for 0"
    print(f"sqrt(0) = 0 [OK]")
    assert sqrt_q64(1, frac_bits=0) == 1, "Failed for 1"
    print(f"sqrt(1) = 1 [OK]")

    # Test case 5: Variance-like values (from LayerNorm)
    print("\nTest 5: Variance-like values (LayerNorm typical)")
    # Variance for INT8 values: mean of squared deviations
    # For example, if mean = 10, and values are [8, 9, 10, 11, 12]
    # Deviations: [-2, -1, 0, 1, 2], squared: [4, 1, 0, 1, 4], mean: 10/5 = 2
    test_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    for val in test_values:
        result = sqrt_q64(val, frac_bits=0)
        np_result = np.sqrt(val)
        error = abs(result - np_result)
        print(f"sqrt({val:3d}) = {result:3d}, np.sqrt = {np_result:.3f}, error: {error:.4f}")

    print("\n[OK] All sqrt_q64 tests passed!")
    print("="*80)

if __name__ == "__main__":
    test_sqrt_q64()
    test_layernorm_int8()
