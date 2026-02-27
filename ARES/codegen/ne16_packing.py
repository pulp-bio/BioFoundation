# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
NE16 Weight Packing Utilities for Offline Preprocessing

This module provides Python functions to pack INT8 weights into the bit-sliced
format required by the GAP9 NE16 neural accelerator. Packing weights offline
(during code generation) eliminates runtime packing overhead.

NE16 Data Format:
- NE16 requires unsigned INT8 (0-255) inputs
- ARES uses signed INT8 (-128 to 127)
- Solution: Add 128 to signed inputs (u8 = s8 + 128)

Weight Packing (1x1 conv / linear):
- Groups of 16 input channels
- Each group: bit-sliced format (16 bytes for 8-bit weights)
- Bias correction: bias_corr = original_bias - 128 * sum(weights)

Reference: shareable_demos/tinymyo/src/ne16_linear_kernels.c
"""

import numpy as np
from typing import Tuple, Optional


# NE16 configuration constants
NE16_INPUT_ZP = 128        # Input zero-point offset (signed -> unsigned)
NE16_WEIGHT_OFFSET = -128  # Weight offset for unsigned storage


def ne16_pack_linear_weights(
    weights_s8: np.ndarray,
    bias_s32: Optional[np.ndarray] = None,
    input_zp: int = NE16_INPUT_ZP,
    weight_offset: int = NE16_WEIGHT_OFFSET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack linear layer weights for NE16 1x1 convolution mode.

    NE16 treats linear layers as 1x1 convolutions with spatial dimensions = 1.
    Weights are packed into a bit-sliced format with groups of 16 input channels.

    Args:
        weights_s8: INT8 weights [out_features, in_features]
        bias_s32: INT32 bias [out_features] (optional, can be None)
        input_zp: Input zero-point offset (default 128)
        weight_offset: Weight offset for unsigned domain (default -128)

    Returns:
        packed_weights: Packed weights in NE16 bit-sliced format [out_features, packed_in_features]
        bias_corrected: Bias with input zero-point correction [out_features]

    The bias correction compensates for the +128 input offset:
        bias_corr = original_bias - input_zp * sum(weights_for_output_channel)
    """
    weights_s8 = np.asarray(weights_s8, dtype=np.int8)
    out_features, in_features = weights_s8.shape

    # Pad in_features to multiple of 16
    nb_ki = (in_features + 15) // 16
    padded_in = nb_ki * 16

    # Allocate output arrays
    # Packed size: out_features rows, each with nb_ki * 16 bytes
    packed_weights = np.zeros((out_features, padded_in), dtype=np.uint8)
    bias_corrected = np.zeros(out_features, dtype=np.int32)

    for ko in range(out_features):
        sum_w = 0
        row = weights_s8[ko, :]

        for kimaj in range(nb_ki):
            # Read one group of 16 input channels (pad with zeros for tail)
            w_u8 = np.zeros(16, dtype=np.uint8)
            for kimin in range(16):
                idx = kimaj * 16 + kimin
                if idx < in_features:
                    w_s8 = int(row[idx])
                else:
                    w_s8 = 0
                sum_w += w_s8

                # Convert to unsigned storage domain
                w_shifted = w_s8 - weight_offset
                w_u8[kimin] = np.uint8(w_shifted & 0xFF)

            # Pack to NE16 1x1 layout: [bit][byte] where each byte packs 8 weights
            # Output: 16 bytes per group (8 bits * 2 bytes per bit)
            base = kimaj * 16
            for bit in range(8):
                b0 = 0
                b1 = 0
                for i in range(8):
                    b0 |= ((int(w_u8[i]) >> bit) & 0x1) << i
                    b1 |= ((int(w_u8[i + 8]) >> bit) & 0x1) << i
                packed_weights[ko, base + bit * 2 + 0] = np.uint8(b0)
                packed_weights[ko, base + bit * 2 + 1] = np.uint8(b1)

        # Compute bias correction
        base_bias = int(bias_s32[ko]) if bias_s32 is not None else 0
        bias_corrected[ko] = base_bias - input_zp * sum_w

    return packed_weights, bias_corrected


def ne16_pack_conv1x1_weights(
    weights_s8: np.ndarray,
    bias_s32: Optional[np.ndarray] = None,
    input_zp: int = NE16_INPUT_ZP,
    weight_offset: int = NE16_WEIGHT_OFFSET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack 1x1 convolution weights for NE16.

    1x1 convolutions use the same packing as linear layers. This function
    handles the additional spatial dimensions in the weight tensor.

    Args:
        weights_s8: INT8 weights [out_channels, in_channels, 1, 1]
        bias_s32: INT32 bias [out_channels] (optional)
        input_zp: Input zero-point offset (default 128)
        weight_offset: Weight offset for unsigned domain (default -128)

    Returns:
        packed_weights: Packed weights [out_channels, packed_in_channels]
        bias_corrected: Corrected bias [out_channels]
    """
    weights_s8 = np.asarray(weights_s8, dtype=np.int8)

    # Handle 4D weight tensor [out_ch, in_ch, 1, 1]
    if weights_s8.ndim == 4:
        out_ch, in_ch, kh, kw = weights_s8.shape
        assert kh == 1 and kw == 1, f"Expected 1x1 kernel, got {kh}x{kw}"
        weights_2d = weights_s8.reshape(out_ch, in_ch)
    else:
        weights_2d = weights_s8

    return ne16_pack_linear_weights(weights_2d, bias_s32, input_zp, weight_offset)


def ne16_pack_conv3x3_weights(
    weights_s8: np.ndarray,
    bias_s32: Optional[np.ndarray] = None,
    input_zp: int = NE16_INPUT_ZP,
    weight_offset: int = NE16_WEIGHT_OFFSET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack 3x3 convolution weights for NE16.

    NE16's 3x3 convolution mode uses a different packing layout than 1x1.
    The kernel spatial elements are interleaved with the channel dimension.

    Args:
        weights_s8: INT8 weights [out_channels, in_channels, 3, 3]
        bias_s32: INT32 bias [out_channels] (optional)
        input_zp: Input zero-point offset (default 128)
        weight_offset: Weight offset for unsigned domain (default -128)

    Returns:
        packed_weights: Packed weights in NE16 3x3 format
        bias_corrected: Corrected bias [out_channels]
    """
    weights_s8 = np.asarray(weights_s8, dtype=np.int8)

    if weights_s8.ndim == 4:
        out_ch, in_ch, kh, kw = weights_s8.shape
        assert kh == 3 and kw == 3, f"Expected 3x3 kernel, got {kh}x{kw}"
    else:
        raise ValueError("3x3 conv weights must be 4D [out_ch, in_ch, 3, 3]")

    # For 3x3 mode, NE16 expects weights organized as:
    # [ko][ki_group][bit][height][width][2 bytes]
    # This matches the GAP SDK ne16_reorg_utils.py layout.
    #
    # Driver strides (from GAP SDK CNN_BasicKernels_NE16.c):
    # - W_D0 = 2*3*3 = 18 bytes (stride per bit plane within a ki_group)
    # - W_D1 = 2*3*3*8*nb_ki = 144*nb_ki bytes (stride per output channel)

    nb_ki = (in_ch + 15) // 16

    # Packed size per output channel: nb_ki * 8 bits * 3 * 3 spatial * 2 bytes = nb_ki * 144
    packed_size_per_ko = nb_ki * 8 * 3 * 3 * 2  # = nb_ki * 144
    total_packed_size = out_ch * packed_size_per_ko

    # Allocate output
    packed_weights = np.zeros(total_packed_size, dtype=np.uint8)
    bias_corrected = np.zeros(out_ch, dtype=np.int32)

    # For each output channel, compute bias correction (sum of all weights)
    for ko in range(out_ch):
        sum_w = np.sum(weights_s8[ko, :, :, :].astype(np.int32))
        base_bias = int(bias_s32[ko]) if bias_s32 is not None else 0
        bias_corrected[ko] = base_bias - input_zp * sum_w

    # Convert weights to unsigned domain
    weights_u8 = (weights_s8.astype(np.int16) - weight_offset).astype(np.uint8)

    # IMPORTANT: NE16 expects weights in OHWI format [out_ch, height, width, in_ch]
    # but PyTorch provides OIHW [out_ch, in_ch, height, width].
    # We need to transpose before packing to match GAP SDK ne16_conv_3x3_weight_layout.
    # OIHW -> OHWI: transpose axes (0, 1, 2, 3) -> (0, 2, 3, 1)
    weights_ohwi = weights_u8.transpose(0, 2, 3, 1)  # [out_ch, 3, 3, in_ch]

    # Pack weights in NE16 3x3 format
    # Layout: [ko][ki_group][bit][height][width][2 bytes]
    # This matches the reference NE16 3x3 weight layout used by GAP SDK tooling.

    for ko in range(out_ch):
        for ki_group in range(nb_ki):
            ki_start = ki_group * 16

            for bit in range(8):
                for h in range(3):
                    for w in range(3):
                        # Gather 16 input channel weights for this spatial position
                        # and extract the specified bit
                        b0 = 0
                        b1 = 0
                        for ki_local in range(16):
                            ki = ki_start + ki_local
                            if ki < in_ch:
                                # Access in OHWI format: [ko, h, w, ki]
                                w_val = weights_ohwi[ko, h, w, ki]
                                bit_val = (w_val >> bit) & 0x1
                            else:
                                bit_val = 0

                            if ki_local < 8:
                                b0 |= bit_val << ki_local
                            else:
                                b1 |= bit_val << (ki_local - 8)

                        # Calculate offset: [ko][ki_group][bit][h][w][2bytes]
                        offset = (ko * nb_ki * 8 * 3 * 3 * 2 +
                                  ki_group * 8 * 3 * 3 * 2 +
                                  bit * 3 * 3 * 2 +
                                  h * 3 * 2 +
                                  w * 2)
                        packed_weights[offset] = np.uint8(b0)
                        packed_weights[offset + 1] = np.uint8(b1)

    return packed_weights, bias_corrected


def ne16_pack_conv3x3_depthwise_weights(
    weights_s8: np.ndarray,
    bias_s32: Optional[np.ndarray] = None,
    input_zp: int = NE16_INPUT_ZP,
    weight_offset: int = NE16_WEIGHT_OFFSET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack depthwise 3x3 convolution weights for NE16.

    Depthwise convolution: each output channel convolves with exactly ONE input
    channel (the corresponding one). Weight shape is [channels, 1, 3, 3].

    This matches the reference packing used in pulp-nnx (Ne16Weight.encode with
    depthwise=True). In depthwise mode, NE16 uses symmetric channel tiling
    (Ko/Ki both 16-wide subtiles), and weights are packed as a single "output"
    channel with the channel dimension moved into the Cin axis.

    Args:
        weights_s8: INT8 weights [channels, 1, 3, 3]
        bias_s32: INT32 bias [channels] (optional)
        input_zp: Input zero-point offset (default 128)
        weight_offset: Weight offset for unsigned domain (default -128)

    Returns:
        packed_weights: Packed weights in NE16 depthwise format
        bias_corrected: Corrected bias [channels]
    """
    weights_s8 = np.asarray(weights_s8, dtype=np.int8)

    if weights_s8.ndim == 4:
        channels, in_ch_per_group, kh, kw = weights_s8.shape
        assert in_ch_per_group == 1, f"Depthwise expects in_ch=1 per group, got {in_ch_per_group}"
        assert kh == 3 and kw == 3, f"Expected 3x3 kernel, got {kh}x{kw}"
    else:
        raise ValueError("Depthwise weights must be 4D [channels, 1, 3, 3]")

    bias_corrected = np.zeros(channels, dtype=np.int32)

    # Compute bias correction (sum of 9 weights per channel)
    for c in range(channels):
        sum_w = np.sum(weights_s8[c, 0, :, :].astype(np.int32))
        base_bias = int(bias_s32[c]) if bias_s32 is not None else 0
        bias_corrected[c] = base_bias - input_zp * sum_w

    # Convert weights to unsigned domain
    weights_u8 = (weights_s8.astype(np.int16) - weight_offset).astype(np.uint8)

    # Reference layout (pulp-nnx Ne16Weight.encode with depthwise=True):
    # - Swap Cout/Cin, so weights become [1, channels, 3, 3]
    # - Pad Cin up to a multiple of 16
    # - Pack as [cout=1][cinMajor][bits][spatial=9][cinMinorBytes=2]
    #   where cinMinor=16 and bits=8.
    w = np.ascontiguousarray(weights_u8.transpose(1, 0, 2, 3))  # [1, C, 3, 3]

    cin = w.shape[1]
    cin_subtile = 16
    if cin % cin_subtile != 0:
        cin_pad = cin_subtile - (cin % cin_subtile)
        w = np.pad(w, ((0, 0), (0, cin_pad), (0, 0), (0, 0)), constant_values=0)
        cin = cin + cin_pad

    cin_major = cin // cin_subtile
    # Reshape into (cout, cinMajor, cinMinor, spatial, 1) for bit unpacking
    w = w.reshape(1, cin_major, cin_subtile, 9, 1)
    w_bits = np.unpackbits(w, axis=-1, count=8, bitorder="little")  # (1, cinMajor, 16, 9, 8)
    w_bits = w_bits.transpose(0, 1, 4, 3, 2)  # (1, cinMajor, 8, 9, 16)

    # Pack 16 bits -> 2 bytes (little order), then flatten.
    w_bits = w_bits.reshape(-1, 8)
    packed_weights = np.packbits(w_bits, axis=-1, bitorder="little").reshape(-1).astype(np.uint8)

    return packed_weights, bias_corrected


def ne16_pack_conv3x3_depthwise_weights_with_requant(
    weights_s8: np.ndarray,
    bias_s32: Optional[np.ndarray],
    scale_input: float,
    scale_weight: float,
    scale_output: float,
    input_zp: int = NE16_INPUT_ZP,
    weight_offset: int = NE16_WEIGHT_OFFSET
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack depthwise 3x3 weights for NE16 with hardware requantization parameters.

    This is a wrapper around ne16_pack_conv3x3_depthwise_weights that also
    computes the per-channel scale (qbias) and shift (qnorm) for NE16's
    hardware output quantization unit.

    NE16 OUTQUANT formula:
        output = clamp((acc * scale + scale_bias) >> shift, -128, 127)

    To match our SW requant formula:
        output = clamp((acc + bias_corrected) * combined_scale, -128, 127)

    We compute scale_bias = bias_corrected * scale, which gives:
        (acc * scale + bias_corrected * scale) >> shift
        = (acc + bias_corrected) * scale >> shift
        = (acc + bias_corrected) * combined_scale

    Args:
        weights_s8: INT8 weights [channels, 1, 3, 3]
        bias_s32: INT32 bias [channels] (optional)
        scale_input: Input quantization scale
        scale_weight: Weight quantization scale (per-tensor)
        scale_output: Output quantization scale
        input_zp: Input zero-point offset (default 128)
        weight_offset: Weight offset for unsigned domain (default -128)

    Returns:
        packed_weights: Packed weights in NE16 depthwise format
        bias_scaled: Scaled bias for NE16 OUTQUANT [channels] (= bias_corrected * hw_scale)
        hw_scale: Per-channel scale multiplier (qbias) [channels]
        hw_scale_shift: Per-channel right shift (qnorm) [channels]
    """
    # Pack weights and compute bias correction
    packed_weights, bias_corrected = ne16_pack_conv3x3_depthwise_weights(
        weights_s8, bias_s32, input_zp, weight_offset
    )

    # Compute hardware requant parameters
    channels = weights_s8.shape[0]
    hw_scale, hw_scale_shift = compute_ne16_requant_params(
        scale_input, scale_weight, scale_output, channels
    )

    # Scale the bias for NE16 OUTQUANT: bias_scaled = bias_corrected * hw_scale
    # This converts the additive bias to the scaled domain expected by NE16
    bias_scaled = (bias_corrected.astype(np.int64) * hw_scale.astype(np.int64)).astype(np.int32)

    return packed_weights, bias_scaled, hw_scale, hw_scale_shift


def is_ne16_eligible_linear(in_features: int, out_features: int, min_macs: int = 1024,
                            max_packed_size: int = 65536, allow_tiling: bool = False) -> bool:
    """
    Determine if a linear layer should use NE16.

    NE16 is most efficient when weights fit entirely in L1 without tiling.
    With double-buffered tiling (allow_tiling=True), larger layers can also
    benefit from NE16 by overlapping DMA with compute.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        min_macs: Minimum MACs for NE16 to be beneficial (default 1024)
        max_packed_size: Maximum packed weight size (default 64KB).
                         Layers larger than this require tiling.
        allow_tiling: If True, allow layers that require output channel tiling.
                      Double-buffered tiling overlaps DMA with NE16 compute.

    Returns:
        True if NE16 is recommended for this layer
    """
    total_macs = in_features * out_features
    if total_macs < min_macs:
        return False

    nb_ki = (in_features + 15) // 16
    packed_size = out_features * nb_ki * 16

    # Check if weights fit in L1 without tiling
    if packed_size <= max_packed_size:
        return True

    # If tiling is allowed, check if a reasonably-sized tile fits in L1
    if allow_tiling:
        # Try tile_ko values from 64 down to 16 to find one that fits
        # Double-buffering needs: 2 * tile_packed + 2 * tile_bias
        for tile_ko in [64, 32, 16]:
            tile_packed_size = tile_ko * nb_ki * 16
            double_buffer_size = 2 * tile_packed_size + 2 * tile_ko * 4
            if double_buffer_size <= max_packed_size:
                return True

    return False


def is_ne16_eligible_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    height: int,
    width: int,
    min_macs: int = 1024,
    max_packed_weight_size: int = 16384  # 16KB limit to leave room for bias + scratch
) -> bool:
    """
    Determine if a Conv2D layer should use NE16.

    NE16 supports 1x1 and 3x3 kernels. Other kernel sizes fall back to software.
    IMPORTANT: NE16 on gvsoc requires weights in L1. Layers with weights too large
    for L1 will produce incorrect results, so we filter them out here.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: (kernel_h, kernel_w)
        height: Input height
        width: Input width
        min_macs: Minimum MACs threshold
        max_packed_weight_size: Maximum packed weight size in bytes (L1 limit)

    Returns:
        True if NE16 is recommended for this layer
    """
    kh, kw = kernel_size

    # NE16 supports 1x1 and 3x3 kernels
    # The executor (network_executor.c) handles this automatically by DMA'ing
    # weight tiles to L1 and scattering INT32 results to the full output buffer.
    if (kh, kw) not in [(1, 1), (3, 3)]:
        return False

    # Calculate total MACs
    total_macs = in_channels * out_channels * kh * kw * height * width
    if total_macs < min_macs:
        return False

    # Check if packed weights fit in L1 (NE16 on gvsoc requires weights in L1)
    # For 1x1 conv: packed_size = out_channels * ceil(in_channels/16) * 16
    # For 3x3 conv: packed_size is larger but has Ko-tiling support
    if (kh, kw) == (1, 1):
        nb_ki = (in_channels + 15) // 16
        packed_size = out_channels * nb_ki * 16
        if packed_size > max_packed_weight_size:
            return False  # Weights too large for L1, fall back to software

    # For 3x3 conv, check if entire output fits in L1 (no spatial tiling support)
    # NE16 3x3 needs: padded_input (U8) + output (INT32)
    # If output is too large, fall back to software convolution
    if (kh, kw) == (3, 3):
        # Conservative L1 limit for NE16 3x3 (leave room for weights and other buffers)
        NE16_3X3_MAX_OUTPUT_S32 = 100000  # 100KB for INT32 output buffer
        output_s32_size = height * width * out_channels * 4
        if output_s32_size > NE16_3X3_MAX_OUTPUT_S32:
            return False  # Output too large for L1, fall back to software

    return True


def get_ne16_packed_weight_size(in_features: int, out_features: int) -> int:
    """
    Calculate the size in bytes of NE16-packed weights for a linear layer.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension

    Returns:
        Size in bytes of the packed weight tensor
    """
    nb_ki = (in_features + 15) // 16
    padded_in = nb_ki * 16
    return out_features * padded_in


def compute_ne16_requant_params(
    scale_input: float,
    scale_weight: float,
    scale_output: float,
    out_features: int,
    max_scale_bits: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel scale and shift for NE16 hardware requantization.

    NE16 OUTQUANT applies a fixed-point requantization of the accumulator, using:
        output = clamp((acc * Scale + ScaleBias) >> Shift, -128, 127)

    Where `Scale` and `Shift` come from this function, and `ScaleBias` must be
    derived from ARES' `bias_corr` to match the intended ordering.

    The combined floating-point scale is:
        combined_scale = scale_input * scale_weight / scale_output

    We convert this to fixed-point: Scale * 2^(-Shift) ≈ combined_scale
    using the same approach as the reference `compute_scales()` implementation:
      - represent `combined_scale` as mantissa * 2^exponent (frexp)
      - choose an integer `Scale` and `Shift` such that:
          Scale = round(mantissa * 2^bits)
          Shift = bits - exponent
      - reduce `bits` while Scale overflows, Shift is too large, or Scale is even
        (odd Scale is preferred for accuracy/stability)

    Args:
        scale_input: Input tensor quantization scale
        scale_weight: Weight tensor quantization scale (per-tensor or per-channel)
        scale_output: Output tensor quantization scale
        out_features: Number of output features/channels
        max_scale_bits: Maximum bits for scale (default 8 for uint8)

    Returns:
        scale: uint8 array [out_features] - per-channel scale multiplier
        scale_shift: uint8 array [out_features] - per-channel right shift amount
    """
    # Calculate combined floating-point scale (handle per-tensor and per-channel weight scales)
    if isinstance(scale_weight, np.ndarray) and scale_weight.size == out_features:
        # Per-channel weight scale
        combined_scale = scale_input * scale_weight / scale_output
    else:
        # Per-tensor weight scale - broadcast to all channels
        combined_scale = np.full(out_features, scale_input * float(scale_weight) / scale_output)

    combined_scale = np.asarray(combined_scale, dtype=np.float32)

    # Handle invalid scales defensively
    valid = combined_scale > 0.0
    if not np.all(valid):
        combined_scale = combined_scale.copy()
        combined_scale[~valid] = 0.0

    # Reference compute_scales logic for uint8 (no external dependency)
    max_bits = int(max_scale_bits)
    max_val = 1 << max_bits
    bits = np.full(combined_scale.shape, max_bits, dtype=np.int32)
    mant, exp = np.frexp(combined_scale)  # mantissa in [0.5, 1) and exponent s.t. x = mant * 2^exp

    while True:
        qbias = np.floor(mant * (2.0 ** bits) + 0.5).astype(np.int64)
        qnorm = (bits - exp).astype(np.int64)  # Shift

        max_exceeded = qbias >= max_val
        norms_too_high = qnorm > (32 - 8)
        bias_even = (qbias & 1) == 0
        should_move = max_exceeded | norms_too_high | bias_even
        can_still_move = (qnorm > 0) & (bits > 0)
        move = should_move & can_still_move
        if not np.any(move):
            break
        bits[move] -= 1

    # Clamp and cast outputs
    qbias = np.clip(qbias, 0, max_val - 1).astype(np.uint8)
    qnorm = np.clip(qnorm, 0, 255).astype(np.uint8)

    # For invalid scales, return a safe identity-ish mapping
    qbias[~valid] = 1
    qnorm[~valid] = 0

    return qbias, qnorm


def compute_ne16_requant_params_from_scales(
    scale_input: float,
    scale_weight: np.ndarray,
    scale_output: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper for per-channel weight scales.

    Args:
        scale_input: Input tensor quantization scale
        scale_weight: Per-channel weight scales [out_features]
        scale_output: Output tensor quantization scale

    Returns:
        scale: uint8 array [out_features]
        scale_shift: uint8 array [out_features]
    """
    return compute_ne16_requant_params(
        scale_input, scale_weight, scale_output, len(scale_weight)
    )


def validate_ne16_packing(
    weights_s8: np.ndarray,
    packed_weights: np.ndarray,
    weight_offset: int = NE16_WEIGHT_OFFSET
) -> bool:
    """
    Validate NE16 weight packing by unpacking and comparing.

    Useful for debugging to ensure packing is correct.

    Args:
        weights_s8: Original INT8 weights [out_features, in_features]
        packed_weights: Packed weights from ne16_pack_linear_weights
        weight_offset: Weight offset used during packing

    Returns:
        True if unpacked weights match original (within padding)
    """
    out_features, in_features = weights_s8.shape
    nb_ki = (in_features + 15) // 16

    for ko in range(out_features):
        for kimaj in range(nb_ki):
            base = kimaj * 16

            # Unpack 16 weights from bit-sliced format
            unpacked = np.zeros(16, dtype=np.uint8)
            for bit in range(8):
                b0 = packed_weights[ko, base + bit * 2]
                b1 = packed_weights[ko, base + bit * 2 + 1]
                for i in range(8):
                    unpacked[i] |= ((b0 >> i) & 0x1) << bit
                    unpacked[i + 8] |= ((b1 >> i) & 0x1) << bit

            # Compare with original (after offset)
            for kimin in range(16):
                idx = kimaj * 16 + kimin
                expected = 0
                if idx < in_features:
                    expected = (int(weights_s8[ko, idx]) - weight_offset) & 0xFF
                if unpacked[kimin] != expected:
                    return False

    return True
