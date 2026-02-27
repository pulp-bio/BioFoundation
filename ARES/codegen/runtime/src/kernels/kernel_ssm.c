/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * kernel_ssm.c - MAMBA SSM Kernels
 *
 * Integer-only State Space Model kernels for GAP9.
 *
 * Contents:
 *   - Conv1D depthwise (with SIMD optimization for kernel_size=4)
 *   - Fused Conv1D + SiLU + Transpose
 *   - SiLU LUT generation (Q13 format)
 *   - SSM exp/phi1 LUTs (Q15 format, 512 entries)
 *   - Softplus kernel
 *   - SSM discretization and scan
 *   - Full SSM layer (Mamba v1)
 *
 * Part of the ARES modular kernel system.
 */

// ---
// MAMBA-Specific Kernels (Integer-Only SSM)
// ---

// SIMD optimization for Conv1D depthwise kernels (kernel_size=4 path)
// Maps to GAP9 SumDotpSS intrinsic for 4-element dot products
#ifndef CONV1D_USE_SIMD
#define CONV1D_USE_SIMD 1  // 1 = enable SIMD path for kernel_size=4, 0 = scalar only
#endif

/**
 * Depthwise 1D Convolution (MAMBA conv1d)
 *
 * Each channel has its own filter (groups=C). For causal mode, uses left-only
 * padding to ensure output at position t only depends on inputs up to t.
 *
 * Data layout: [C, L] (channel-first)
 * Weight layout: [C, K] (one K-tap filter per channel)
 */
void network_conv1d_depthwise_int8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int8_t *output,
    int channels,
    int length,
    int kernel_size,
    int causal,
    float scale_input,
    float scale_weight,
    float scale_output
) {
    const int requant_shift = 24;
    float scale_combined = (scale_input * scale_weight) / scale_output;
    const int32_t requant_mul = qround(scale_combined * (float)(1 << requant_shift));

    // Determine padding
    int pad_left = causal ? (kernel_size - 1) : (kernel_size / 2);
    int pad_right = causal ? 0 : (kernel_size - 1 - pad_left);
    int out_len = length;  // Same length output with proper padding

    // Parallelize across channels
    int core_id = pi_core_id();
    int chunk = (channels + NUM_CORES - 1) / NUM_CORES;
    int start_ch = core_id * chunk;
    int end_ch = (start_ch + chunk < channels) ? (start_ch + chunk) : channels;

    for (int c = start_ch; c < end_ch; c++) {
        const int8_t *in_ch = input + c * length;
        const int8_t *w_ch = weights + c * kernel_size;
        int8_t *out_ch = output + c * out_len;
        int32_t bias_val = (bias != NULL) ? bias[c] : 0;

#if CONV1D_USE_SIMD && !defined(__EMUL__)
        // SIMD path for kernel_size=4 (common in Mamba)
        // Uses SumDotpSS for 4-element dot products
        if (kernel_size == 4 && causal) {
            // Load 4-tap filter weights as SIMD vector
            v4s w_vec = *((const v4s *)w_ch);

            // Boundary positions (0, 1, 2): partial overlap, use scalar
            // Position 0: only w[3]*in[0] contributes
            {
                int32_t acc = bias_val + (int32_t)w_ch[3] * (int32_t)in_ch[0];
                int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                if (out_int < -128) out_int = -128;
                if (out_int > 127) out_int = 127;
                out_ch[0] = (int8_t)out_int;
            }
            // Position 1: w[2]*in[0] + w[3]*in[1]
            if (out_len > 1) {
                int32_t acc = bias_val +
                    (int32_t)w_ch[2] * (int32_t)in_ch[0] +
                    (int32_t)w_ch[3] * (int32_t)in_ch[1];
                int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                if (out_int < -128) out_int = -128;
                if (out_int > 127) out_int = 127;
                out_ch[1] = (int8_t)out_int;
            }
            // Position 2: w[1]*in[0] + w[2]*in[1] + w[3]*in[2]
            if (out_len > 2) {
                int32_t acc = bias_val +
                    (int32_t)w_ch[1] * (int32_t)in_ch[0] +
                    (int32_t)w_ch[2] * (int32_t)in_ch[1] +
                    (int32_t)w_ch[3] * (int32_t)in_ch[2];
                int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                if (out_int < -128) out_int = -128;
                if (out_int > 127) out_int = 127;
                out_ch[2] = (int8_t)out_int;
            }

            // Main loop: positions 3+ have full overlap, use SIMD
            // Input indices are [ol-3, ol-2, ol-1, ol], all valid
            for (int ol = 3; ol < out_len; ol++) {
                // Load 4 contiguous input values
                v4s x_vec = *((const v4s *)(in_ch + ol - 3));
                // SIMD 4-element dot product
                int32_t acc = SumDotpSS(x_vec, w_vec, bias_val);
                int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                if (out_int < -128) out_int = -128;
                if (out_int > 127) out_int = 127;
                out_ch[ol] = (int8_t)out_int;
            }
        } else
#endif
        // Scalar fallback for general case
        {
            for (int ol = 0; ol < out_len; ol++) {
                int32_t acc = 0;

                for (int k = 0; k < kernel_size; k++) {
                    int il = ol + k - pad_left;

                    int8_t x_val = 0;
                    if (il >= 0 && il < length) {
                        x_val = in_ch[il];
                    }
                    // Zero padding for out-of-bounds

                    int8_t w_val = w_ch[k];
                    acc += (int32_t)x_val * (int32_t)w_val;
                }

                // Add bias
                acc += bias_val;

                // Rescale and clip to INT8
                float out_fp = (float)acc * scale_combined;
                int32_t out_int = qround(out_fp);
                if (out_int < -128) out_int = -128;
                if (out_int > 127) out_int = 127;
                out_ch[ol] = (int8_t)out_int;
            }
        }
    }

    pi_cl_team_barrier();
}

/**
 * FUSED: Conv1D Depthwise + SiLU + Transpose
 *
 * Fuses three operations into a single pass:
 * 1. Depthwise Conv1D: [B, d_inner, L] -> [B, d_inner, L]
 * 2. SiLU activation via LUT
 * 3. Transpose: [B, d_inner, L] -> [B, L, d_inner]
 *
 * This eliminates the intermediate conv_out buffer and reduces memory traffic
 * by ~2KB for typical MambaBlock configurations.
 *
 * Input layout:  [batch, channels, length] = [B, d_inner, L]
 * Output layout: [batch, length, channels] = [B, L, d_inner]
 */
void network_conv1d_silu_transpose_fused(
    const int8_t *input,       // [B, d_inner, L] layout
    const int8_t *weights,     // [d_inner, kernel_size] depthwise weights
    const int32_t *bias,       // [d_inner] bias (INT32)
    const int8_t *silu_lut,    // [256] SiLU lookup table
    int8_t *output,            // [B, L, d_inner] layout (transposed)
    int batch,
    int channels,              // d_inner
    int length,                // L
    int kernel_size,
    int causal,
    float scale_input,
    float scale_weight,
    float scale_output
) {
    const int requant_shift = 24;
    float scale_combined = (scale_input * scale_weight) / scale_output;
    const int32_t requant_mul = qround(scale_combined * (float)(1 << requant_shift));

    // Determine padding (causal = left padding only)
    int pad_left = causal ? (kernel_size - 1) : (kernel_size / 2);

    // Parallelize across channels
    int core_id = pi_core_id();
    int chunk = (channels + NUM_CORES - 1) / NUM_CORES;
    int start_ch = core_id * chunk;
    int end_ch = (start_ch + chunk < channels) ? (start_ch + chunk) : channels;

    for (int b = 0; b < batch; b++) {
        const int8_t *in_batch = input + b * channels * length;
        int8_t *out_batch = output + b * length * channels;

        for (int c = start_ch; c < end_ch; c++) {
            const int8_t *in_ch = in_batch + c * length;
            const int8_t *w_ch = weights + c * kernel_size;
            const int32_t bias_val = bias ? bias[c] : 0;

            if (causal && kernel_size == 4) {
                // Fast path for common Mamba kernel (K=4, causal padding)
                const int8_t w0 = w_ch[0];
                const int8_t w1 = w_ch[1];
                const int8_t w2 = w_ch[2];
                const int8_t w3 = w_ch[3];

                if (length > 0) {
                    int32_t acc = bias_val + (int32_t)w3 * (int32_t)in_ch[0];
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;
                    out_batch[0 * channels + c] = silu_lut[(int)conv_out + 128];
                }
                if (length > 1) {
                    int32_t acc = bias_val +
                        (int32_t)w2 * (int32_t)in_ch[0] +
                        (int32_t)w3 * (int32_t)in_ch[1];
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;
                    out_batch[1 * channels + c] = silu_lut[(int)conv_out + 128];
                }
                if (length > 2) {
                    int32_t acc = bias_val +
                        (int32_t)w1 * (int32_t)in_ch[0] +
                        (int32_t)w2 * (int32_t)in_ch[1] +
                        (int32_t)w3 * (int32_t)in_ch[2];
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;
                    out_batch[2 * channels + c] = silu_lut[(int)conv_out + 128];
                }
                for (int l = 3; l < length; l++) {
                    int32_t acc = bias_val +
                        (int32_t)w0 * (int32_t)in_ch[l - 3] +
                        (int32_t)w1 * (int32_t)in_ch[l - 2] +
                        (int32_t)w2 * (int32_t)in_ch[l - 1] +
                        (int32_t)w3 * (int32_t)in_ch[l];
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;
                    out_batch[l * channels + c] = silu_lut[(int)conv_out + 128];
                }
            } else {
                for (int l = 0; l < length; l++) {
                    int32_t acc = bias_val;

                    // Conv1D: accumulate weighted inputs
                    for (int k = 0; k < kernel_size; k++) {
                        int il = l + k - pad_left;
                        if ((unsigned)il < (unsigned)length) {
                            acc += (int32_t)in_ch[il] * (int32_t)w_ch[k];
                        }
                    }

                    // Rescale and clip to INT8
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;

                    // Apply SiLU via LUT and write to transposed output
                    out_batch[l * channels + c] = silu_lut[(int)conv_out + 128];
                }
            }
        }
    }

    pi_cl_team_barrier();
}

void network_conv1d_silu_transpose_fused_strided(
    const int8_t *input,       // [B, L, input_stride] layout
    int input_stride,          // Stride between timesteps (elements)
    const int8_t *weights,     // [d_inner, kernel_size] depthwise weights
    const int32_t *bias,       // [d_inner] bias (INT32)
    const int8_t *silu_lut,    // [256] SiLU lookup table
    int8_t *output,            // [B, L, d_inner] layout (transposed)
    int batch,
    int channels,              // d_inner
    int length,                // L
    int kernel_size,
    int causal,
    float scale_input,
    float scale_weight,
    float scale_output
) {
    const int requant_shift = 24;
    float scale_combined = (scale_input * scale_weight) / scale_output;
    const int32_t requant_mul = qround(scale_combined * (float)(1 << requant_shift));

    // Determine padding (causal = left padding only)
    int pad_left = causal ? (kernel_size - 1) : (kernel_size / 2);

    // Parallelize across channels
    int core_id = pi_core_id();
    int chunk = (channels + NUM_CORES - 1) / NUM_CORES;
    int start_ch = core_id * chunk;
    int end_ch = (start_ch + chunk < channels) ? (start_ch + chunk) : channels;

    for (int b = 0; b < batch; b++) {
        const int8_t *in_batch = input + b * length * input_stride;
        int8_t *out_batch = output + b * length * channels;

        for (int c = start_ch; c < end_ch; c++) {
            const int8_t *in_ch = in_batch + c;
            const int8_t *w_ch = weights + c * kernel_size;
            const int32_t bias_val = bias ? bias[c] : 0;

            if (causal && kernel_size == 4) {
                // Fast path for common Mamba kernel (K=4, causal padding)
                const int8_t w0 = w_ch[0];
                const int8_t w1 = w_ch[1];
                const int8_t w2 = w_ch[2];
                const int8_t w3 = w_ch[3];

                if (length > 0) {
                    int32_t acc = bias_val + (int32_t)w3 * (int32_t)in_ch[0];
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;
                    out_batch[0 * channels + c] = silu_lut[(int)conv_out + 128];
                }
                if (length > 1) {
                    int32_t acc = bias_val +
                        (int32_t)w2 * (int32_t)in_ch[0 * input_stride] +
                        (int32_t)w3 * (int32_t)in_ch[1 * input_stride];
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;
                    out_batch[1 * channels + c] = silu_lut[(int)conv_out + 128];
                }
                if (length > 2) {
                    int32_t acc = bias_val +
                        (int32_t)w1 * (int32_t)in_ch[0 * input_stride] +
                        (int32_t)w2 * (int32_t)in_ch[1 * input_stride] +
                        (int32_t)w3 * (int32_t)in_ch[2 * input_stride];
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;
                    out_batch[2 * channels + c] = silu_lut[(int)conv_out + 128];
                }
                for (int l = 3; l < length; l++) {
                    int32_t acc = bias_val +
                        (int32_t)w0 * (int32_t)in_ch[(l - 3) * input_stride] +
                        (int32_t)w1 * (int32_t)in_ch[(l - 2) * input_stride] +
                        (int32_t)w2 * (int32_t)in_ch[(l - 1) * input_stride] +
                        (int32_t)w3 * (int32_t)in_ch[l * input_stride];
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;
                    out_batch[l * channels + c] = silu_lut[(int)conv_out + 128];
                }
            } else {
                for (int l = 0; l < length; l++) {
                    int32_t acc = bias_val;

                    // Conv1D: accumulate weighted inputs
                    for (int k = 0; k < kernel_size; k++) {
                        int il = l + k - pad_left;
                        if ((unsigned)il < (unsigned)length) {
                            acc += (int32_t)in_ch[il * input_stride] * (int32_t)w_ch[k];
                        }
                    }

                    // Rescale and clip to INT8
                    int32_t out_int = mul_shift_round_nearest_even(acc, requant_mul, requant_shift);
                    if (out_int < -128) out_int = -128;
                    if (out_int > 127) out_int = 127;
                    int8_t conv_out = (int8_t)out_int;

                    // Apply SiLU via LUT and write to transposed output
                    out_batch[l * channels + c] = silu_lut[(int)conv_out + 128];
                }
            }
        }
    }

    pi_cl_team_barrier();
}

// SiLU functions (network_silu_int8_lut, network_silu_int8_lut_inplace, generate_silu_lut_int8)
// moved to ops/op_activation.c

/**
 * Generate SiLU Q2.13 LUT for gating operation
 * Output range: [-4, 4] with 13 fractional bits
 */
void generate_silu_lut_q13(
    int16_t *lut,
    float scale_in
) {
    for (int i = 0; i < 256; i++) {
        int8_t x_int8 = (int8_t)(i - 128);
        float x_fp32 = (float)x_int8 * scale_in;

        // SiLU output
        float sigmoid_x = 1.0f / (1.0f + fast_exp(-x_fp32));
        float y_fp32 = x_fp32 * sigmoid_x;

        // Convert to Q2.13
        int32_t y_q13 = qround(y_fp32 * 8192.0f);  // 2^13 = 8192
        if (y_q13 < -32768) y_q13 = -32768;
        if (y_q13 > 32767) y_q13 = 32767;
        lut[i] = (int16_t)y_q13;
    }
}

/**
 * SSM Gate: y_output = y_ssm * silu(z)
 *
 * Uses Q2.13 SiLU LUT for the gating multiplication.
 * y_output = (y_ssm * silu_q13(z)) >> 13
 */
void ssm_gate_silu_q13(
    const int8_t *y_ssm_int8,
    const int8_t *z_int8,
    int8_t *output_int8,
    const int16_t *silu_lut_q13,
    int num_elements
) {
    // Parallelize across elements
    int core_id = pi_core_id();
    int chunk = (num_elements + NUM_CORES - 1) / NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk < num_elements) ? (start + chunk) : num_elements;

    for (int i = start; i < end; i++) {
        // SiLU lookup on z
        int z_idx = (int)z_int8[i] + 128;
        int16_t silu_z = silu_lut_q13[z_idx];

        // Multiply y_ssm by silu(z) in Q13
        int32_t product = (int32_t)y_ssm_int8[i] * (int32_t)silu_z;

        // Round and shift: (product + 4096) >> 13
        int32_t result = (product + (1 << 12)) >> 13;

        // Clip to INT8
        if (result < -128) result = -128;
        if (result > 127) result = 127;
        output_int8[i] = (int8_t)result;
    }

    pi_cl_team_barrier();
}

// Pre-computed exp() LUT for SSM discretization
// Maps z in [-10, 0] to exp(z) in Q15 (512 entries)
const int16_t ssm_exp_lut_q15[SSM_EXP_LUT_SIZE] = {
    // exp(-10) * 32768 = 1, exp(-9.98) * 32768 = 1, ... exp(0) * 32768 = 32768
    // Generated by Python: [int(np.exp(z) * 32768) for z in np.linspace(-10, 0, 512)]
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11,
    11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 17,
    17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25, 26,
    26, 27, 28, 28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 53, 54, 56, 57, 59, 61,
    62, 64, 66, 68, 70, 72, 74, 76, 79, 81, 83, 86, 88, 91, 94, 96,
    99, 102, 105, 108, 111, 115, 118, 122, 126, 129, 133, 137, 142, 146, 150, 155,
    160, 165, 170, 175, 180, 186, 191, 197, 203, 210, 216, 223, 230, 237, 244, 252,
    260, 268, 276, 285, 294, 303, 312, 322, 332, 342, 353, 364, 375, 387, 399, 412,
    424, 438, 451, 465, 479, 494, 510, 525, 542, 559, 576, 594, 612, 631, 651, 671,
    692, 714, 736, 759, 783, 808, 833, 859, 886, 914, 943, 973, 1004, 1036, 1068, 1102,
    1137, 1173, 1210, 1248, 1288, 1329, 1371, 1414, 1459, 1505, 1553, 1602, 1653, 1705, 1759, 1815,
    1872, 1931, 1992, 2055, 2120, 2187, 2256, 2327, 2401, 2477, 2555, 2636, 2719, 2805, 2894, 2985,
    3080, 3177, 3278, 3381, 3488, 3598, 3712, 3829, 3950, 4074, 4203, 4336, 4473, 4614, 4760, 4910,
    5065, 5225, 5390, 5560, 5736, 5917, 6104, 6297, 6496, 6702, 6914, 7133, 7359, 7593, 7834, 8083,
    8340, 8605, 8879, 9162, 9455, 9757, 10069, 10392, 10725, 11070, 11426, 11795, 12176, 12570, 12978, 13400,
    13837, 14290, 14758, 15244, 15746, 16267, 16806, 17365, 17945, 18545, 19169, 19815, 20485, 21181, 21903, 22652,
    23430, 24237, 25076, 25947, 26853, 27795, 28773, 29792, 30851, 31952, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768
};

// Pre-computed phi1() LUT for SSM discretization
// phi1(x) = (exp(x) - 1) / x for stability when x -> 0
// Maps z in [-10, 0] to phi1(z) in Q15 (512 entries)
const int16_t ssm_phi1_lut_q15[SSM_PHI1_LUT_SIZE] = {
    // phi1(-10) * 32768, ... phi1(0) * 32768 = 32768 (since phi1(0) = 1)
    // Generated by Python with stable formula
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7,
    8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 15, 15,
    16, 17, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31,
    32, 34, 35, 37, 39, 40, 42, 44, 46, 48, 51, 53, 56, 58, 61, 64,
    67, 70, 74, 77, 81, 85, 89, 93, 98, 103, 108, 113, 119, 125, 131, 138,
    145, 152, 160, 168, 177, 186, 195, 205, 216, 227, 239, 251, 264, 278, 292, 307,
    323, 340, 358, 377, 396, 417, 439, 462, 486, 512, 539, 567, 597, 629, 662, 697,
    733, 772, 813, 856, 901, 949, 1000, 1053, 1109, 1169, 1232, 1298, 1368, 1442, 1520, 1602,
    1689, 1781, 1878, 1981, 2089, 2204, 2325, 2453, 2589, 2733, 2885, 3047, 3218, 3399, 3592, 3796,
    4013, 4243, 4489, 4750, 5028, 5325, 5641, 5978, 6338, 6723, 7134, 7573, 8043, 8546, 9084, 9661,
    10280, 10945, 11660, 12430, 13259, 14154, 15121, 16168, 17304, 18539, 19882, 21349, 22955, 24716, 26656, 28800,
    31175, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768
};

// ---
// Softplus Kernel (for MAMBA dt computation)
// ---

/**
 * Softplus via piecewise approximation
 * softplus(x) = log(1 + exp(x))
 *
 * For x > 10: softplus(x) ≈ x
 * For x < -10: softplus(x) ≈ exp(x) ≈ 0
 * For -10 <= x <= 10: use polynomial or LUT
 *
 * Input: INT32 accumulator from dt_proj linear layer
 * Output: Q16 fixed-point (for dt in SSM)
 */
void network_softplus_q16(
    const int32_t *input,
    int32_t *output_q16,
    int num_elements,
    float scale_in
) {
    int core_id = pi_core_id();
    int chunk = (num_elements + NUM_CORES - 1) / NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk < num_elements) ? (start + chunk) : num_elements;

    for (int i = start; i < end; i++) {
        // Convert INT32 accumulator to FP32 x value
        float x = (float)input[i] * scale_in;

        float result;
        if (x > 10.0f) {
            // Linear region: softplus(x) ≈ x for large x
            result = x;
        } else if (x < -10.0f) {
            // Exponential region: softplus(x) ≈ exp(x) for x << 0
            result = fast_exp(x);
        } else {
            // Core region: log(1 + exp(x))
            // Use stable formula: softplus(x) = x + log(1 + exp(-|x|)) for x > 0
            //                    softplus(x) = log(1 + exp(x)) for x <= 0
            if (x > 0) {
                result = x + logf(1.0f + fast_exp(-x));
            } else {
                result = logf(1.0f + fast_exp(x));
            }
        }

        // Convert to Q16
        output_q16[i] = (int32_t)(result * 65536.0f);
    }

    pi_cl_team_barrier();
}

// ---
// SSM Discretization
// ---

/**
 * SSM Discretization: Convert continuous A, B to discrete dA, dB'
 *
 * dA = exp(dt * A)           -- state decay factor
 * dB' = dt * B * phi1(dt*A)  -- input scaling with numerical stability
 *
 * All computation in Q15 fixed-point using pre-computed LUTs.
 */
void ssm_discretize_q15(
    const int32_t *dt_q16,    // Delta timesteps [seq_len, d_inner] in Q16
    const int16_t *A_q15,     // A matrix [d_state, d_inner] in Q15 (negative values)
    const int16_t *B_q15,     // B matrix [seq_len, d_state] in Q15
    int16_t *dA_q15,          // Output: discretized A [seq_len, d_inner, d_state] in Q15
    int16_t *dB_prime_q15,    // Output: discretized B' [seq_len, d_inner, d_state] in Q15
    int seq_len,
    int d_inner,
    int d_state,
    int16_t s_x_q15           // Input scale in Q15
) {
    // LUT parameters
    const float lut_min = -10.0f;
    const float lut_max = 0.0f;
    const int lut_size = SSM_EXP_LUT_SIZE;
    const float lut_step = (lut_max - lut_min) / (lut_size - 1);
    const float inv_step = 1.0f / lut_step;

    int core_id = pi_core_id();

    // Parallelize over sequence length
    int chunk = (seq_len + NUM_CORES - 1) / NUM_CORES;
    int t_start = core_id * chunk;
    int t_end = (t_start + chunk < seq_len) ? (t_start + chunk) : seq_len;

    for (int t = t_start; t < t_end; t++) {
        for (int m = 0; m < d_inner; m++) {
            // Get dt value (Q16) and convert to float
            int32_t dt_val = dt_q16[t * d_inner + m];
            float dt_f = (float)dt_val / 65536.0f;

            for (int d = 0; d < d_state; d++) {
                // Get A value (Q15, negative) and convert to float
                int16_t A_val = A_q15[d * d_inner + m];
                float A_f = (float)A_val / 32768.0f;

                // z = dt * A (should be negative)
                float z = dt_f * A_f;

                // LUT index for exp(z)
                float idx_f = (z - lut_min) * inv_step;
                int idx = (int)idx_f;
                if (idx < 0) idx = 0;
                if (idx >= lut_size) idx = lut_size - 1;

                // dA = exp(dt * A)
                dA_q15[(t * d_inner + m) * d_state + d] = ssm_exp_lut_q15[idx];

                // dB' = dt * B * phi1(dt * A) * s_x
                int16_t phi1_val = ssm_phi1_lut_q15[idx];
                int16_t B_val = B_q15[t * d_state + d];

                // Compute in INT32 to avoid overflow
                int64_t dB_temp = ((int64_t)dt_val * (int64_t)B_val * (int64_t)phi1_val * (int64_t)s_x_q15) >> 31;

                // Clip to Q15 range
                if (dB_temp > 32767) dB_temp = 32767;
                if (dB_temp < -32768) dB_temp = -32768;

                dB_prime_q15[(t * d_inner + m) * d_state + d] = (int16_t)dB_temp;
            }
        }
    }

    pi_cl_team_barrier();
}

// ---
// SSM Scan (State Space Model Recurrence)
// ---

/**
 * SSM Scan: State update loop
 *
 * For each timestep t:
 *   h[t] = dA[t] * h[t-1] + dB'[t] * x[t]
 *   y[t] = C[t] · h[t] + D * x[t]
 *
 * All computation in Q15 fixed-point (dyadic arithmetic).
 */
void ssm_scan_q15(
    const int8_t *x_i8,       // Input [seq_len, d_inner] in INT8
    const int16_t *dA_q15,    // Discretized A [seq_len, d_inner, d_state] in Q15
    const int16_t *dB_prime_q15, // Discretized B' [seq_len, d_inner, d_state] in Q15
    const int16_t *C_q15,     // C matrix [seq_len, d_state] in Q15
    const int16_t *D_q15,     // D skip connection [d_inner] in Q15 (can be NULL)
    int32_t *y_acc,           // Output accumulator [seq_len, d_inner] in INT32
    int16_t *h_q15,           // Hidden state buffer [d_inner, d_state] in Q15
    int seq_len,
    int d_inner,
    int d_state
) {
    int core_id = pi_core_id();

    // Parallelize over d_inner channels
    int chunk = (d_inner + NUM_CORES - 1) / NUM_CORES;
    int m_start = core_id * chunk;
    int m_end = (m_start + chunk < d_inner) ? (m_start + chunk) : d_inner;

    for (int m = m_start; m < m_end; m++) {
        // Initialize hidden state to zero for this channel
        for (int d = 0; d < d_state; d++) {
            h_q15[m * d_state + d] = 0;
        }

        // Sequential scan over time (cannot be parallelized due to recurrence)
        for (int t = 0; t < seq_len; t++) {
            int8_t x_val = x_i8[t * d_inner + m];
            int32_t y_sum = 0;

            for (int d = 0; d < d_state; d++) {
                int idx = (t * d_inner + m) * d_state + d;
                int16_t dA = dA_q15[idx];
                int16_t dB_prime = dB_prime_q15[idx];
                int16_t h_prev = h_q15[m * d_state + d];
                int16_t C_val = C_q15[t * d_state + d];

                // h[t] = dA * h[t-1] + dB' * x
                int32_t h_decay = ((int32_t)dA * (int32_t)h_prev) >> 15;
                int32_t h_input = ((int32_t)dB_prime * (int32_t)x_val);
                int32_t h_new = h_decay + h_input;

                // Clip to Q15 range
                if (h_new > 32767) h_new = 32767;
                if (h_new < -32768) h_new = -32768;
                h_q15[m * d_state + d] = (int16_t)h_new;

                // y[t] += C * h
                y_sum += ((int32_t)C_val * h_new) >> 15;
            }

            // Add D skip connection: y[t] += D * x[t]
            // D_q15 is in Q15 format, x_val is INT8
            // Result is scaled by 2^15 * scale_x, accumulated in y_acc
            if (D_q15 != NULL) {
                y_sum += ((int32_t)D_q15[m] * (int32_t)x_val);
            }

            y_acc[t * d_inner + m] = y_sum;
        }
    }

    pi_cl_team_barrier();
}

// ---
// Complete SSM Forward Pass
// ---

/**
 * SSM Forward: Complete integer-only state space model
 * I-Mamba step 2c: D parameter now in Q15 format
 */
void network_ssm_forward_int8(
    const int8_t *x_i8,          // Input [seq_len, d_inner] in INT8
    const int32_t *dt_q16,       // Delta timesteps [seq_len, d_inner] in Q16
    const int16_t *A_q15,        // A matrix [d_state, d_inner] in Q15
    const int16_t *B_q15,        // B matrix [seq_len, d_state] in Q15
    const int16_t *C_q15,        // C matrix [seq_len, d_state] in Q15
    const int16_t *D_q15,        // D skip connection [d_inner] in Q15 (can be NULL)
    const int8_t *z_i8,          // Gate input [seq_len, d_inner] in INT8
    int8_t *y_i8,                // Output [seq_len, d_inner] in INT8
    int16_t *dA_buf,             // Temp buffer [seq_len, d_inner, d_state]
    int16_t *dB_buf,             // Temp buffer [seq_len, d_inner, d_state]
    int32_t *y_acc_buf,          // Temp buffer [seq_len, d_inner]
    int16_t *h_buf,              // Hidden state buffer [d_inner, d_state]
    const int16_t *silu_lut_q13, // SiLU LUT for gating (256 entries)
    int seq_len,
    int d_inner,
    int d_state,
    float scale_x,
    float scale_y
) {
    int16_t s_x_q15 = (int16_t)(scale_x * 32768.0f);

    // Step 1: Discretization
    ssm_discretize_q15(
        dt_q16, A_q15, B_q15,
        dA_buf, dB_buf,
        seq_len, d_inner, d_state,
        s_x_q15
    );

    // Step 2: SSM Scan (with D skip connection)
    ssm_scan_q15(
        x_i8, dA_buf, dB_buf, C_q15, D_q15,
        y_acc_buf, h_buf,
        seq_len, d_inner, d_state
    );

    // Step 3: SiLU Gating and output
    int core_id = pi_core_id();
    int num_elements = seq_len * d_inner;
    int chunk = (num_elements + NUM_CORES - 1) / NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk < num_elements) ? (start + chunk) : num_elements;

    for (int i = start; i < end; i++) {
        // SiLU lookup on z
        int z_idx = (int)z_i8[i] + 128;
        int16_t silu_z = silu_lut_q13[z_idx];

        // Multiply y_acc by silu(z) in Q13
        int32_t product = y_acc_buf[i] * (int32_t)silu_z;

        // Scale and requantize to INT8
        float y_float = (float)product / 8192.0f;  // Remove Q13
        int32_t result = qround(y_float / scale_y);

        if (result > 127) result = 127;
        if (result < -128) result = -128;
        y_i8[i] = (int8_t)result;
    }

    pi_cl_team_barrier();
}

// ---
// Complete SSM Layer with Projections
// ---

/**
 * SSM Layer: Complete layer including x_proj, dt_proj, softplus, scan
 * Matches the Python ssm_layer_forward_int8 function
 *
 * This handles the full QuantSSM layer:
 * 1. x_proj: Project input to get dt_input, B, C
 * 2. dt_proj: Project dt_input to full dt and apply softplus
 * 3. SSM core: Discretize and scan
 * 4. Output: Quantize to INT8
 *
 * I-Mamba step 10: Fully integer SSM layer (zero FP32 operations)
 * All computations use fixed-point arithmetic with precomputed scale factors:
 * - dt_proj: INT8 weights + Q16.16 bias + precomputed dt_scale_q
 * - B/C: Computed to Q15 using precomputed bc_scale_factor
 * - Hidden state: Q15 fixed-point with Q15 exp LUT for discretization
 * - Output: Direct INT8 using precomputed output_scale_q
 */
void network_ssm_layer_int8(
    const int8_t *x_int8,           // Input [batch, seq_len, d_inner]
    int8_t *output_int8,            // Output [batch, seq_len, d_inner]
    const int8_t *x_proj_weight,    // x_proj weights [dt_rank + 2*d_state, d_inner]
    const int8_t *dt_proj_weight,   // dt_proj weights [d_inner, dt_rank] (INT8)
    const int32_t *dt_proj_bias_q16_16,  // dt_proj bias [d_inner] (Q16.16 fixed-point)
    const int16_t *A_q15,           // A matrix [d_state, d_inner] in Q15 (pre-computed -exp(A_log))
    const int16_t *D_q15,           // D skip connection [d_inner] in Q15
    const int16_t *softplus_lut,    // Q8.8 softplus LUT [256]
    const int16_t *exp_lut,         // Q15 exp LUT for discretization [256]
    void *scratch,                  // Scratch buffer for integer intermediates
    int batch,
    int seq_len,
    int d_inner,
    int d_state,
    int dt_rank,
    int32_t dt_scale_q,             // Precomputed: (scale_x * scale_x_proj * scale_dt_proj) * 65536 * 2^24
    int dt_scale_shift,             // Shift amount for dt_scale_q (typically 24)
    int32_t bc_scale_factor,        // Precomputed: scale_x * scale_x_proj * 32768 * 2^16
    int32_t output_scale_q          // Precomputed: scale_x / (32768 * scale_output) * 2^24
) {
    // I-Mamba step 10: Fully integer SSM (zero FP32)
    // Scratch buffer layout (all integer):
    // - dt_q8_8 [batch * seq_len * d_inner] (INT16, Q8.8 softplus output)
    // - h_state_q15 [batch * d_inner * d_state] (INT16, Q15 hidden state)
    int16_t *dt_q8_8 = (int16_t *)scratch;
    int16_t *h_state_q15 = dt_q8_8 + batch * seq_len * d_inner;

    // Initialize Q15 state to zero
    memset(h_state_q15, 0, batch * d_inner * d_state * sizeof(int16_t));

    // Fixed-point constants
    const int BC_SHIFT = 16;

    // SIMD constants for x_proj
    const int simd_count = d_inner >> 2;  // d_inner / 4
    const int simd_remainder = d_inner & 0x3;

    // Process each batch and timestep
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq_len; t++) {
            // --- Step 1: x_proj to get B, C (directly to Q15) and dt_input ---
            int proj_size = dt_rank + 2 * d_state;
            int32_t proj_acc[32];  // INT32 accumulators (max size)
            const int8_t *x_ptr = &x_int8[(b * seq_len + t) * d_inner];

            for (int j = 0; j < proj_size; j++) {
                int32_t acc = 0;
                const int8_t *w_ptr = &x_proj_weight[j * d_inner];

                // SIMD inner loop: 4 INT8 MACs per iteration
                const v4s *pA = (const v4s *)x_ptr;
                const v4s *pB = (const v4s *)w_ptr;
                for (int i = 0; i < simd_count; i++) {
                    acc = SumDotpSS(pA[i], pB[i], acc);
                }

                // Handle remainder
                for (int i = simd_count * 4; i < d_inner; i++) {
                    acc += (int32_t)x_ptr[i] * (int32_t)w_ptr[i];
                }
                proj_acc[j] = acc;
            }

            // Extract B (Q15), dt_input (keep as scaled int), C (Q15)
            int16_t B_q15[16];   // Max d_state = 16
            int16_t C_q15[16];
            int32_t dt_input_acc[32];  // Keep INT32 for dt_proj

            for (int d = 0; d < d_state; d++) {
                // B: Convert to Q15 using precomputed bc_scale_factor
                int64_t b_scaled = ((int64_t)proj_acc[dt_rank + d] * bc_scale_factor) >> BC_SHIFT;
                if (b_scaled > 32767) b_scaled = 32767;
                if (b_scaled < -32768) b_scaled = -32768;
                B_q15[d] = (int16_t)b_scaled;

                // C: Convert to Q15 using precomputed bc_scale_factor
                int64_t c_scaled = ((int64_t)proj_acc[dt_rank + d_state + d] * bc_scale_factor) >> BC_SHIFT;
                if (c_scaled > 32767) c_scaled = 32767;
                if (c_scaled < -32768) c_scaled = -32768;
                C_q15[d] = (int16_t)c_scaled;
            }

            // dt_input: Keep as INT32 accumulator for dt_proj
            for (int r = 0; r < dt_rank; r++) {
                dt_input_acc[r] = proj_acc[r];
            }

            // --- Step 2: dt_proj + softplus (full integer path) ---
            for (int m = 0; m < d_inner; m++) {
                // dt_proj matmul: dt_input (scaled) x dt_proj_weight (INT8)
                int32_t dt_acc = 0;
                for (int r = 0; r < dt_rank; r++) {
                    dt_acc += (dt_input_acc[r] >> 8) * (int32_t)dt_proj_weight[m * dt_rank + r];
                }

                // Convert dt_acc to Q16.16 using precomputed scale factor
                int64_t dt_scaled = (int64_t)dt_acc * (int64_t)dt_scale_q;
                int32_t dt_val_q16_16 = (int32_t)(dt_scaled >> dt_scale_shift);

                // Add Q16.16 bias
                dt_val_q16_16 += dt_proj_bias_q16_16[m];

                // Integer LUT index: idx = (dt_val_q16_16 * 10) >> 16
                int32_t lut_idx = (dt_val_q16_16 * 10) >> 16;
                if (lut_idx > 127) lut_idx = 127;
                if (lut_idx < -128) lut_idx = -128;
                dt_q8_8[(b * seq_len + t) * d_inner + m] = softplus_lut[lut_idx + 128];
            }

            // --- Step 3: SSM scan (full integer) ---
            for (int m = 0; m < d_inner; m++) {
                int8_t x_i8 = x_int8[(b * seq_len + t) * d_inner + m];
                int16_t dt_val = dt_q8_8[(b * seq_len + t) * d_inner + m];  // Q8.8

                int32_t y_acc = 0;  // Accumulator for output

                for (int d = 0; d < d_state; d++) {
                    // Get A_q15 value (negative, in range [-32768, 0))
                    int16_t A_val = A_q15[d * d_inner + m];

                    // dt x A: Q8.8 x Q15 = Q23, shift to Q15
                    int32_t dt_A_q23 = (int32_t)dt_val * (int32_t)A_val;
                    int32_t dt_A_q15 = dt_A_q23 >> 8;

                    // LUT index: idx = (dt_A_q15 * 10) >> 15
                    int32_t exp_idx = (dt_A_q15 * 10) >> 15;
                    if (exp_idx > 127) exp_idx = 127;
                    if (exp_idx < -128) exp_idx = -128;
                    int16_t dA_q15 = exp_lut[exp_idx + 128];

                    // dB = dt x B: Q8.8 x Q15 = Q23, shift to Q15
                    int32_t dB_q23 = (int32_t)dt_val * (int32_t)B_q15[d];
                    int16_t dB_q15 = (int16_t)(dB_q23 >> 8);

                    // Get current hidden state (Q15)
                    int h_idx = b * d_inner * d_state + m * d_state + d;
                    int16_t h_prev = h_state_q15[h_idx];

                    // State update: h_new = dA x h_prev + dB x x
                    int32_t h_decay = ((int32_t)dA_q15 * (int32_t)h_prev) >> 15;
                    int32_t h_input = ((int32_t)dB_q15 * (int32_t)x_i8);
                    int32_t h_new = h_decay + (h_input >> 7);

                    // Clip to Q15 range
                    if (h_new > 32767) h_new = 32767;
                    if (h_new < -32768) h_new = -32768;
                    h_state_q15[h_idx] = (int16_t)h_new;

                    // Output: y += C x h (Q15 x Q15 = Q30, shift to Q15)
                    y_acc += ((int32_t)C_q15[d] * h_new) >> 15;
                }

                // Add skip connection: y += D x x
                y_acc += ((int32_t)D_q15[m] * (int32_t)x_i8);

                // Direct integer conversion to INT8
                const int OUTPUT_SHIFT = 24;
                int64_t y_scaled = (int64_t)y_acc * (int64_t)output_scale_q;
                int32_t q = (int32_t)((y_scaled + (1LL << (OUTPUT_SHIFT - 1))) >> OUTPUT_SHIFT);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                output_int8[(b * seq_len + t) * d_inner + m] = (int8_t)q;
            }
        }
    }
}

// ---
// Element-wise Multiply (for gating operations)
// ---

/**
 * Element-wise multiply: out = a * b with requantization
 * Used for x * gate(z) pattern in MAMBA
 */
void network_elementwise_mul_int8(
    const int8_t *a,
    const int8_t *b,
    int8_t *output,
    int num_elements,
    float scale_a,
    float scale_b,
    float scale_out
) {
    float combined_scale = (scale_a * scale_b) / scale_out;

    int core_id = pi_core_id();
    int chunk = (num_elements + NUM_CORES - 1) / NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk < num_elements) ? (start + chunk) : num_elements;

    for (int i = start; i < end; i++) {
        int32_t product = (int32_t)a[i] * (int32_t)b[i];
        int32_t result = qround((float)product * combined_scale);

        if (result > 127) result = 127;
        if (result < -128) result = -128;
        output[i] = (int8_t)result;
    }

    pi_cl_team_barrier();
}

// Simple inplace operations (relu_int8_inplace, requantize_int8_inplace)
// moved to ops/op_activation.c

// Conv2D tile workers (conv2d_tile_worker, conv2d_tile_worker_with_fusion) moved to ops/op_conv2d.c

// Linear tile workers (linear_tile_worker, linear_int8_l1_worker) moved to ops/op_linear.c

