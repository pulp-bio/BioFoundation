/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * kernel_linear.c - Linear Layer Kernels for GAP9
 *
 * Contains all Linear (fully-connected) layer variants:
 *   - network_linear_int8: Standard INT8->INT8 with SIMD
 *   - network_linear_int8_sequential: Batch/token processing
 *   - network_linear_int8_2bit_*: 2-bit packed weight variants
 *   - network_linear_int8_parallel_tokens: Multi-token parallel
 *   - network_linear_int8_parallel_tokens_strided: Strided output for tiling
 *   - network_linear_int8_to_fp32: Final classifier (INT8->FP32)
 *
 * Optimizations:
 *   - SIMD via SumDotpSS intrinsic (4 MACs per cycle)
 *   - 4x output feature unrolling
 *   - 8-wide loop unrolling for 2-bit weights
 *   - Fixed-point requantization option
 *
 * Part of the ARES modular kernel system.
 */

// ---
// LINEAR KERNEL (INT8 -> INT8) - SIMD Optimized with 4x Output Unrolling
// ---
// Uses PULP SIMD instructions for 4x throughput on inner loop
// PULP-NN style optimization: process 4 output features simultaneously
// to amortize input vector loads across multiple weight row dot products.

#ifndef LINEAR_INT8_OUTF_UNROLL
#define LINEAR_INT8_OUTF_UNROLL 4
#endif

void network_linear_int8(
    const int8_t *input, const int8_t *weights, const void *bias, int8_t *output,
    uint16_t in_features, uint16_t out_features,
    float scale_input, float scale_weight, float scale_output
) {
    int core_id = pi_core_id();
    int chunk = (out_features + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start_f = core_id * chunk;
    int end_f = (start_f + chunk > out_features) ? out_features : (start_f + chunk);

    // Precompute combined scale
    const float combined_scale = scale_input * scale_weight / scale_output;

    // SIMD loop count (process 4 elements at a time)
    const int simd_count = in_features >> 2;  // in_features / 4
    const int remainder = in_features & 0x3;  // in_features % 4

    // Precompute input pointers for SIMD and remainder
    const v4s *pA = (const v4s *)input;
    const int8_t *pA_rem = input + (simd_count << 2);

    int i = start_f;

#if LINEAR_INT8_OUTF_UNROLL >= 4
    // 4x output feature unrolling: process 4 outputs simultaneously
    // Same input vector multiplied by 4 different weight rows
    for (; i + 3 < end_f; i += 4) {
        int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

        const int8_t *w0 = weights + (i + 0) * in_features;
        const int8_t *w1 = weights + (i + 1) * in_features;
        const int8_t *w2 = weights + (i + 2) * in_features;
        const int8_t *w3 = weights + (i + 3) * in_features;

        const v4s *pB0 = (const v4s *)w0;
        const v4s *pB1 = (const v4s *)w1;
        const v4s *pB2 = (const v4s *)w2;
        const v4s *pB3 = (const v4s *)w3;

        // SIMD inner loop: 4 weight rows x 4 elements = 16 MACs per iteration
        for (int j = 0; j < simd_count; j++) {
            const v4s a = pA[j];
            acc0 = SumDotpSS(a, pB0[j], acc0);
            acc1 = SumDotpSS(a, pB1[j], acc1);
            acc2 = SumDotpSS(a, pB2[j], acc2);
            acc3 = SumDotpSS(a, pB3[j], acc3);
        }

        // Handle remainder elements
        if (remainder > 0) {
            const int8_t *pB0_rem = w0 + (simd_count << 2);
            const int8_t *pB1_rem = w1 + (simd_count << 2);
            const int8_t *pB2_rem = w2 + (simd_count << 2);
            const int8_t *pB3_rem = w3 + (simd_count << 2);
            for (int j = 0; j < remainder; j++) {
                int8_t a = pA_rem[j];
                acc0 += (int32_t)a * (int32_t)pB0_rem[j];
                acc1 += (int32_t)a * (int32_t)pB1_rem[j];
                acc2 += (int32_t)a * (int32_t)pB2_rem[j];
                acc3 += (int32_t)a * (int32_t)pB3_rem[j];
            }
        }

        // Add biases
        if (bias) {
            acc0 += ((int32_t*)bias)[i + 0];
            acc1 += ((int32_t*)bias)[i + 1];
            acc2 += ((int32_t*)bias)[i + 2];
            acc3 += ((int32_t*)bias)[i + 3];
        }

        // Requantize all 4 outputs
        float v0 = (float)acc0 * combined_scale;
        float v1 = (float)acc1 * combined_scale;
        float v2 = (float)acc2 * combined_scale;
        float v3 = (float)acc3 * combined_scale;

        int32_t q0 = qround(v0); if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
        int32_t q1 = qround(v1); if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
        int32_t q2 = qround(v2); if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
        int32_t q3 = qround(v3); if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

        output[i + 0] = (int8_t)q0;
        output[i + 1] = (int8_t)q1;
        output[i + 2] = (int8_t)q2;
        output[i + 3] = (int8_t)q3;
    }
#endif

    // Handle remaining output features (0-3)
    for (; i < end_f; i++) {
        int32_t acc = 0;
        const int8_t *w_row = weights + i * in_features;

        const v4s *pB = (const v4s *)w_row;
        for (int j = 0; j < simd_count; j++) {
            acc = SumDotpSS(pA[j], pB[j], acc);
        }

        const int8_t *pB_rem = w_row + (simd_count << 2);
        for (int j = 0; j < remainder; j++) {
            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
        }

        if (bias) {
            acc += ((int32_t*)bias)[i];
        }

        float val_fp32 = (float)acc * combined_scale;
        int32_t q = qround(val_fp32);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        output[i] = (int8_t)q;
    }
}

// ---
// LINEAR KERNEL (Sequential / Batch Tokens) - 4x Output Unrolling
// ---
// Used when batch_size > 1 (e.g. Transformer Tokens)
// PULP-NN style: process 4 output features simultaneously
void network_linear_int8_sequential(
    const int8_t *input, const int8_t *weights, const void *bias, int8_t *output,
    uint16_t in_features, uint16_t out_features,
    float scale_input, float scale_weight, float scale_output
) {
    // This function processes ONE token (input vector) across ALL output features.
    // It usually runs on a single core as part of a larger parallel loop over tokens.
    // SIMD optimized: 4 INT8 MACs per cycle via SumDotpSS
    // 4x output unrolling: load input once, multiply by 4 weight rows

    const float combined_scale = scale_input * scale_weight / scale_output;
    const int simd_count = in_features >> 2;  // in_features / 4
    const int remainder = in_features & 0x3;  // in_features % 4

    // Precompute input pointers
    const v4s *pA = (const v4s *)input;
    const int8_t *pA_rem = input + (simd_count << 2);

    int i = 0;

#if LINEAR_INT8_OUTF_UNROLL >= 4
    // 4x output feature unrolling
    for (; i + 3 < out_features; i += 4) {
        int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

        const int8_t *w0 = weights + (i + 0) * in_features;
        const int8_t *w1 = weights + (i + 1) * in_features;
        const int8_t *w2 = weights + (i + 2) * in_features;
        const int8_t *w3 = weights + (i + 3) * in_features;

        const v4s *pB0 = (const v4s *)w0;
        const v4s *pB1 = (const v4s *)w1;
        const v4s *pB2 = (const v4s *)w2;
        const v4s *pB3 = (const v4s *)w3;

        // SIMD inner loop: 4 weight rows x 4 elements = 16 MACs per iteration
        for (int j = 0; j < simd_count; j++) {
            const v4s a = pA[j];
            acc0 = SumDotpSS(a, pB0[j], acc0);
            acc1 = SumDotpSS(a, pB1[j], acc1);
            acc2 = SumDotpSS(a, pB2[j], acc2);
            acc3 = SumDotpSS(a, pB3[j], acc3);
        }

        // Handle remainder elements
        if (remainder > 0) {
            const int8_t *pB0_rem = w0 + (simd_count << 2);
            const int8_t *pB1_rem = w1 + (simd_count << 2);
            const int8_t *pB2_rem = w2 + (simd_count << 2);
            const int8_t *pB3_rem = w3 + (simd_count << 2);
            for (int j = 0; j < remainder; j++) {
                int8_t a = pA_rem[j];
                acc0 += (int32_t)a * (int32_t)pB0_rem[j];
                acc1 += (int32_t)a * (int32_t)pB1_rem[j];
                acc2 += (int32_t)a * (int32_t)pB2_rem[j];
                acc3 += (int32_t)a * (int32_t)pB3_rem[j];
            }
        }

        // Add biases
        if (bias) {
            acc0 += ((int32_t*)bias)[i + 0];
            acc1 += ((int32_t*)bias)[i + 1];
            acc2 += ((int32_t*)bias)[i + 2];
            acc3 += ((int32_t*)bias)[i + 3];
        }

        // Requantize all 4 outputs
        float v0 = (float)acc0 * combined_scale;
        float v1 = (float)acc1 * combined_scale;
        float v2 = (float)acc2 * combined_scale;
        float v3 = (float)acc3 * combined_scale;

        int32_t q0 = qround(v0); if (q0 > 127) q0 = 127; if (q0 < -128) q0 = -128;
        int32_t q1 = qround(v1); if (q1 > 127) q1 = 127; if (q1 < -128) q1 = -128;
        int32_t q2 = qround(v2); if (q2 > 127) q2 = 127; if (q2 < -128) q2 = -128;
        int32_t q3 = qround(v3); if (q3 > 127) q3 = 127; if (q3 < -128) q3 = -128;

        output[i + 0] = (int8_t)q0;
        output[i + 1] = (int8_t)q1;
        output[i + 2] = (int8_t)q2;
        output[i + 3] = (int8_t)q3;
    }
#endif

    // Handle remaining output features (0-3)
    for (; i < out_features; i++) {
        int32_t acc = 0;
        const int8_t *w_row = weights + i * in_features;

        const v4s *pB = (const v4s *)w_row;
        for (int j = 0; j < simd_count; j++) {
            acc = SumDotpSS(pA[j], pB[j], acc);
        }

        const int8_t *pB_rem = w_row + (simd_count << 2);
        for (int j = 0; j < remainder; j++) {
            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
        }

        if (bias) {
            acc += ((int32_t*)bias)[i];
        }

        float val_fp32 = (float)acc * combined_scale;
        int32_t q = qround(val_fp32);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        output[i] = (int8_t)q;
    }
}

// ---
// 2-BIT UNPACK LOOKUP TABLE: 256 entries, each unpacks 1 byte → 4 weights
// ---
// Eliminates 12 instructions (4 shifts + 4 ANDs + 4 subtracts) per byte
// Maps packed byte → v4s{w0, w1, w2, w3} where wi = ((byte >> 2i) & 3) - 1
static const int8_t unpack_2bit_lut[256][4] __attribute__((aligned(4))) = {
    {-1,-1,-1,-1}, { 0,-1,-1,-1}, { 1,-1,-1,-1}, { 2,-1,-1,-1},  // 0x00-0x03
    {-1, 0,-1,-1}, { 0, 0,-1,-1}, { 1, 0,-1,-1}, { 2, 0,-1,-1},  // 0x04-0x07
    {-1, 1,-1,-1}, { 0, 1,-1,-1}, { 1, 1,-1,-1}, { 2, 1,-1,-1},  // 0x08-0x0B
    {-1, 2,-1,-1}, { 0, 2,-1,-1}, { 1, 2,-1,-1}, { 2, 2,-1,-1},  // 0x0C-0x0F
    {-1,-1, 0,-1}, { 0,-1, 0,-1}, { 1,-1, 0,-1}, { 2,-1, 0,-1},  // 0x10-0x13
    {-1, 0, 0,-1}, { 0, 0, 0,-1}, { 1, 0, 0,-1}, { 2, 0, 0,-1},  // 0x14-0x17
    {-1, 1, 0,-1}, { 0, 1, 0,-1}, { 1, 1, 0,-1}, { 2, 1, 0,-1},  // 0x18-0x1B
    {-1, 2, 0,-1}, { 0, 2, 0,-1}, { 1, 2, 0,-1}, { 2, 2, 0,-1},  // 0x1C-0x1F
    {-1,-1, 1,-1}, { 0,-1, 1,-1}, { 1,-1, 1,-1}, { 2,-1, 1,-1},  // 0x20-0x23
    {-1, 0, 1,-1}, { 0, 0, 1,-1}, { 1, 0, 1,-1}, { 2, 0, 1,-1},  // 0x24-0x27
    {-1, 1, 1,-1}, { 0, 1, 1,-1}, { 1, 1, 1,-1}, { 2, 1, 1,-1},  // 0x28-0x2B
    {-1, 2, 1,-1}, { 0, 2, 1,-1}, { 1, 2, 1,-1}, { 2, 2, 1,-1},  // 0x2C-0x2F
    {-1,-1, 2,-1}, { 0,-1, 2,-1}, { 1,-1, 2,-1}, { 2,-1, 2,-1},  // 0x30-0x33
    {-1, 0, 2,-1}, { 0, 0, 2,-1}, { 1, 0, 2,-1}, { 2, 0, 2,-1},  // 0x34-0x37
    {-1, 1, 2,-1}, { 0, 1, 2,-1}, { 1, 1, 2,-1}, { 2, 1, 2,-1},  // 0x38-0x3B
    {-1, 2, 2,-1}, { 0, 2, 2,-1}, { 1, 2, 2,-1}, { 2, 2, 2,-1},  // 0x3C-0x3F
    {-1,-1,-1, 0}, { 0,-1,-1, 0}, { 1,-1,-1, 0}, { 2,-1,-1, 0},  // 0x40-0x43
    {-1, 0,-1, 0}, { 0, 0,-1, 0}, { 1, 0,-1, 0}, { 2, 0,-1, 0},  // 0x44-0x47
    {-1, 1,-1, 0}, { 0, 1,-1, 0}, { 1, 1,-1, 0}, { 2, 1,-1, 0},  // 0x48-0x4B
    {-1, 2,-1, 0}, { 0, 2,-1, 0}, { 1, 2,-1, 0}, { 2, 2,-1, 0},  // 0x4C-0x4F
    {-1,-1, 0, 0}, { 0,-1, 0, 0}, { 1,-1, 0, 0}, { 2,-1, 0, 0},  // 0x50-0x53
    {-1, 0, 0, 0}, { 0, 0, 0, 0}, { 1, 0, 0, 0}, { 2, 0, 0, 0},  // 0x54-0x57 (0x55 = all zeros)
    {-1, 1, 0, 0}, { 0, 1, 0, 0}, { 1, 1, 0, 0}, { 2, 1, 0, 0},  // 0x58-0x5B
    {-1, 2, 0, 0}, { 0, 2, 0, 0}, { 1, 2, 0, 0}, { 2, 2, 0, 0},  // 0x5C-0x5F
    {-1,-1, 1, 0}, { 0,-1, 1, 0}, { 1,-1, 1, 0}, { 2,-1, 1, 0},  // 0x60-0x63
    {-1, 0, 1, 0}, { 0, 0, 1, 0}, { 1, 0, 1, 0}, { 2, 0, 1, 0},  // 0x64-0x67
    {-1, 1, 1, 0}, { 0, 1, 1, 0}, { 1, 1, 1, 0}, { 2, 1, 1, 0},  // 0x68-0x6B
    {-1, 2, 1, 0}, { 0, 2, 1, 0}, { 1, 2, 1, 0}, { 2, 2, 1, 0},  // 0x6C-0x6F
    {-1,-1, 2, 0}, { 0,-1, 2, 0}, { 1,-1, 2, 0}, { 2,-1, 2, 0},  // 0x70-0x73
    {-1, 0, 2, 0}, { 0, 0, 2, 0}, { 1, 0, 2, 0}, { 2, 0, 2, 0},  // 0x74-0x77
    {-1, 1, 2, 0}, { 0, 1, 2, 0}, { 1, 1, 2, 0}, { 2, 1, 2, 0},  // 0x78-0x7B
    {-1, 2, 2, 0}, { 0, 2, 2, 0}, { 1, 2, 2, 0}, { 2, 2, 2, 0},  // 0x7C-0x7F
    {-1,-1,-1, 1}, { 0,-1,-1, 1}, { 1,-1,-1, 1}, { 2,-1,-1, 1},  // 0x80-0x83
    {-1, 0,-1, 1}, { 0, 0,-1, 1}, { 1, 0,-1, 1}, { 2, 0,-1, 1},  // 0x84-0x87
    {-1, 1,-1, 1}, { 0, 1,-1, 1}, { 1, 1,-1, 1}, { 2, 1,-1, 1},  // 0x88-0x8B
    {-1, 2,-1, 1}, { 0, 2,-1, 1}, { 1, 2,-1, 1}, { 2, 2,-1, 1},  // 0x8C-0x8F
    {-1,-1, 0, 1}, { 0,-1, 0, 1}, { 1,-1, 0, 1}, { 2,-1, 0, 1},  // 0x90-0x93
    {-1, 0, 0, 1}, { 0, 0, 0, 1}, { 1, 0, 0, 1}, { 2, 0, 0, 1},  // 0x94-0x97
    {-1, 1, 0, 1}, { 0, 1, 0, 1}, { 1, 1, 0, 1}, { 2, 1, 0, 1},  // 0x98-0x9B
    {-1, 2, 0, 1}, { 0, 2, 0, 1}, { 1, 2, 0, 1}, { 2, 2, 0, 1},  // 0x9C-0x9F
    {-1,-1, 1, 1}, { 0,-1, 1, 1}, { 1,-1, 1, 1}, { 2,-1, 1, 1},  // 0xA0-0xA3
    {-1, 0, 1, 1}, { 0, 0, 1, 1}, { 1, 0, 1, 1}, { 2, 0, 1, 1},  // 0xA4-0xA7
    {-1, 1, 1, 1}, { 0, 1, 1, 1}, { 1, 1, 1, 1}, { 2, 1, 1, 1},  // 0xA8-0xAB
    {-1, 2, 1, 1}, { 0, 2, 1, 1}, { 1, 2, 1, 1}, { 2, 2, 1, 1},  // 0xAC-0xAF
    {-1,-1, 2, 1}, { 0,-1, 2, 1}, { 1,-1, 2, 1}, { 2,-1, 2, 1},  // 0xB0-0xB3
    {-1, 0, 2, 1}, { 0, 0, 2, 1}, { 1, 0, 2, 1}, { 2, 0, 2, 1},  // 0xB4-0xB7
    {-1, 1, 2, 1}, { 0, 1, 2, 1}, { 1, 1, 2, 1}, { 2, 1, 2, 1},  // 0xB8-0xBB
    {-1, 2, 2, 1}, { 0, 2, 2, 1}, { 1, 2, 2, 1}, { 2, 2, 2, 1},  // 0xBC-0xBF
    {-1,-1,-1, 2}, { 0,-1,-1, 2}, { 1,-1,-1, 2}, { 2,-1,-1, 2},  // 0xC0-0xC3
    {-1, 0,-1, 2}, { 0, 0,-1, 2}, { 1, 0,-1, 2}, { 2, 0,-1, 2},  // 0xC4-0xC7
    {-1, 1,-1, 2}, { 0, 1,-1, 2}, { 1, 1,-1, 2}, { 2, 1,-1, 2},  // 0xC8-0xCB
    {-1, 2,-1, 2}, { 0, 2,-1, 2}, { 1, 2,-1, 2}, { 2, 2,-1, 2},  // 0xCC-0xCF
    {-1,-1, 0, 2}, { 0,-1, 0, 2}, { 1,-1, 0, 2}, { 2,-1, 0, 2},  // 0xD0-0xD3
    {-1, 0, 0, 2}, { 0, 0, 0, 2}, { 1, 0, 0, 2}, { 2, 0, 0, 2},  // 0xD4-0xD7
    {-1, 1, 0, 2}, { 0, 1, 0, 2}, { 1, 1, 0, 2}, { 2, 1, 0, 2},  // 0xD8-0xDB
    {-1, 2, 0, 2}, { 0, 2, 0, 2}, { 1, 2, 0, 2}, { 2, 2, 0, 2},  // 0xDC-0xDF
    {-1,-1, 1, 2}, { 0,-1, 1, 2}, { 1,-1, 1, 2}, { 2,-1, 1, 2},  // 0xE0-0xE3
    {-1, 0, 1, 2}, { 0, 0, 1, 2}, { 1, 0, 1, 2}, { 2, 0, 1, 2},  // 0xE4-0xE7
    {-1, 1, 1, 2}, { 0, 1, 1, 2}, { 1, 1, 1, 2}, { 2, 1, 1, 2},  // 0xE8-0xEB
    {-1, 2, 1, 2}, { 0, 2, 1, 2}, { 1, 2, 1, 2}, { 2, 2, 1, 2},  // 0xEC-0xEF
    {-1,-1, 2, 2}, { 0,-1, 2, 2}, { 1,-1, 2, 2}, { 2,-1, 2, 2},  // 0xF0-0xF3
    {-1, 0, 2, 2}, { 0, 0, 2, 2}, { 1, 0, 2, 2}, { 2, 0, 2, 2},  // 0xF4-0xF7
    {-1, 1, 2, 2}, { 0, 1, 2, 2}, { 1, 1, 2, 2}, { 2, 1, 2, 2},  // 0xF8-0xFB
    {-1, 2, 2, 2}, { 0, 2, 2, 2}, { 1, 2, 2, 2}, { 2, 2, 2, 2},  // 0xFC-0xFF
};

// ---
// LINEAR KERNEL (2-bit Packed Weights, Parallel) - 8-Wide Unrolled SIMD
// ---
// Processes 8 weights per iteration to reduce loop overhead
void network_linear_2bit_int8(
    const int8_t *input, const uint8_t *weights_packed, const void *bias, int8_t *output,
    uint16_t in_features, uint16_t out_features,
    float scale_input, float scale_weight, float scale_output
) {
    int core_id = pi_core_id();
    int chunk = (out_features + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start_f = core_id * chunk;
    int end_f = (start_f + chunk > out_features) ? out_features : (start_f + chunk);

    const float combined_scale = scale_input * scale_weight / scale_output;
    const int packed_row_size = (in_features + 3) >> 2;
    const int in_features_aligned8 = in_features & ~7;  // Align to 8
    const int in_features_aligned4 = in_features & ~3;  // Align to 4

    for (int i = start_f; i < end_f; i++) {
        int32_t acc = 0;
        const uint8_t *w_row = weights_packed + i * packed_row_size;

        // 8-wide unrolled: process 2 packed bytes (8 weights) per iteration
        int j = 0;
        int p = 0;
        for (; j < in_features_aligned8; p += 2, j += 8) {
            uint8_t packed0 = w_row[p];
            uint8_t packed1 = w_row[p + 1];

            // Unpack first 4 weights
            int8_t w0 = (int8_t)((packed0 & 0x03) - 1);
            int8_t w1 = (int8_t)(((packed0 >> 2) & 0x03) - 1);
            int8_t w2 = (int8_t)(((packed0 >> 4) & 0x03) - 1);
            int8_t w3 = (int8_t)(((packed0 >> 6) & 0x03) - 1);

            // Unpack second 4 weights
            int8_t w4 = (int8_t)((packed1 & 0x03) - 1);
            int8_t w5 = (int8_t)(((packed1 >> 2) & 0x03) - 1);
            int8_t w6 = (int8_t)(((packed1 >> 4) & 0x03) - 1);
            int8_t w7 = (int8_t)(((packed1 >> 6) & 0x03) - 1);

            // Two SIMD ops
            v4s weights0 = (v4s){w0, w1, w2, w3};
            v4s weights1 = (v4s){w4, w5, w6, w7};
            v4s inputs0 = *((v4s*)&input[j]);
            v4s inputs1 = *((v4s*)&input[j + 4]);
            acc = SumDotpSS(inputs0, weights0, acc);
            acc = SumDotpSS(inputs1, weights1, acc);
        }

        // Handle 4-7 remaining (one more packed byte)
        for (; j < in_features_aligned4; p++, j += 4) {
            uint8_t packed = w_row[p];
            int8_t w0 = (int8_t)((packed & 0x03) - 1);
            int8_t w1 = (int8_t)(((packed >> 2) & 0x03) - 1);
            int8_t w2 = (int8_t)(((packed >> 4) & 0x03) - 1);
            int8_t w3 = (int8_t)(((packed >> 6) & 0x03) - 1);
            v4s weights = (v4s){w0, w1, w2, w3};
            v4s inputs = *((v4s*)&input[j]);
            acc = SumDotpSS(inputs, weights, acc);
        }

        // Handle remaining elements (0-3) scalar
        for (; j < in_features; j++) {
            int p_idx = j >> 2;
            int bit_idx = (j & 3) * 2;
            uint8_t packed = w_row[p_idx];
            int8_t w = (int8_t)(((packed >> bit_idx) & 0x03) - 1);
            acc += (int32_t)input[j] * (int32_t)w;
        }

        if (bias) {
            acc += ((int32_t*)bias)[i];
        }

        float val_fp32 = (float)acc * combined_scale;
        int32_t q = qround(val_fp32);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        output[i] = (int8_t)q;
    }
}

// ---
// LINEAR KERNEL (2-bit Packed Weights, Sequential) - 8-Wide Unrolled SIMD
// ---
// Sequential version for L3 streaming chunks, with 8-wide loop unrolling
void network_linear_2bit_int8_sequential(
    const int8_t *input, const uint8_t *weights_packed, const void *bias, int8_t *output,
    uint16_t in_features, uint16_t out_features,
    float scale_input, float scale_weight, float scale_output
) {
    const float combined_scale = scale_input * scale_weight / scale_output;
    const int packed_row_size = (in_features + 3) >> 2;
    const int in_features_aligned8 = in_features & ~7;  // Align to 8
    const int in_features_aligned4 = in_features & ~3;  // Align to 4

    for (int i = 0; i < out_features; i++) {
        int32_t acc = 0;
        const uint8_t *w_row = weights_packed + i * packed_row_size;

        // 8-wide unrolled: process 2 packed bytes (8 weights) per iteration
        int j = 0;
        int p = 0;
        for (; j < in_features_aligned8; p += 2, j += 8) {
            uint8_t packed0 = w_row[p];
            uint8_t packed1 = w_row[p + 1];

            // Unpack first 4 weights
            int8_t w0 = (int8_t)((packed0 & 0x03) - 1);
            int8_t w1 = (int8_t)(((packed0 >> 2) & 0x03) - 1);
            int8_t w2 = (int8_t)(((packed0 >> 4) & 0x03) - 1);
            int8_t w3 = (int8_t)(((packed0 >> 6) & 0x03) - 1);

            // Unpack second 4 weights
            int8_t w4 = (int8_t)((packed1 & 0x03) - 1);
            int8_t w5 = (int8_t)(((packed1 >> 2) & 0x03) - 1);
            int8_t w6 = (int8_t)(((packed1 >> 4) & 0x03) - 1);
            int8_t w7 = (int8_t)(((packed1 >> 6) & 0x03) - 1);

            // Two SIMD ops
            v4s weights0 = (v4s){w0, w1, w2, w3};
            v4s weights1 = (v4s){w4, w5, w6, w7};
            v4s inputs0 = *((v4s*)&input[j]);
            v4s inputs1 = *((v4s*)&input[j + 4]);
            acc = SumDotpSS(inputs0, weights0, acc);
            acc = SumDotpSS(inputs1, weights1, acc);
        }

        // Handle 4-7 remaining (one more packed byte)
        for (; j < in_features_aligned4; p++, j += 4) {
            uint8_t packed = w_row[p];
            int8_t w0 = (int8_t)((packed & 0x03) - 1);
            int8_t w1 = (int8_t)(((packed >> 2) & 0x03) - 1);
            int8_t w2 = (int8_t)(((packed >> 4) & 0x03) - 1);
            int8_t w3 = (int8_t)(((packed >> 6) & 0x03) - 1);
            v4s weights = (v4s){w0, w1, w2, w3};
            v4s inputs = *((v4s*)&input[j]);
            acc = SumDotpSS(inputs, weights, acc);
        }

        // Handle remaining elements (0-3) scalar
        for (; j < in_features; j++) {
            int p_idx = j >> 2;
            int bit_idx = (j & 3) * 2;
            uint8_t packed = w_row[p_idx];
            int8_t w = (int8_t)(((packed >> bit_idx) & 0x03) - 1);
            acc += (int32_t)input[j] * (int32_t)w;
        }

        if (bias) {
            acc += ((int32_t*)bias)[i];
        }

        float val_fp32 = (float)acc * combined_scale;
        int32_t q = qround(val_fp32);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        output[i] = (int8_t)q;
    }
}

// ---
// LINEAR KERNEL (Multi-Token Parallel) - SIMD Optimized
// ---
// Process multiple tokens in parallel across cores.
// Each core handles a subset of tokens, all computing all output features.
// Weights should be in L1 for best performance.
typedef struct {
    const int8_t *input;       // [seq_len, in_features] in L2
    const int8_t *weights;     // [out_features, in_features] in L1
    const int32_t *bias;       // [out_features] in L2
    int8_t *output;            // [seq_len, out_features] in L2
    int seq_len;
    int in_features;
    int out_features;
    float scale_input;
    float scale_weight;
    float scale_output;
#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
    int32_t requant_mul;
    int requant_shift;
#endif
} mhsa_proj_args_t;

// ---

// LINEAR KERNEL (Multi-Token Parallel, Strided Output) - SIMD Optimized
// ---
// Writes into a larger output tensor with a configurable row stride.
// This enables weight-tiling: compute an output feature tile [out_features_tile]
// and write it into output rows of length [out_stride].
typedef struct {
    const int8_t *input;       // [seq_len, in_features] in L2
    const int8_t *weights;     // [out_features_tile, in_features] in L1
    const int32_t *bias;       // [out_features_tile] in L2 (tile start)
    int8_t *output;            // base pointer to output[0, out_offset] in L2
    int seq_len;
    int in_features;
    int out_features_tile;
    int out_stride;            // full output row stride (total out_features)
    float scale_input;
    float scale_weight;
    float scale_output;
#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
    int32_t requant_mul;
    int requant_shift;
#endif
} linear_strided_args_t;

static void linear_projection_strided_worker(void *args) {
    linear_strided_args_t *p = (linear_strided_args_t *)args;
    const int core_id = pi_core_id();

    // Distribute tokens across cores
    const int tokens_per_core = (p->seq_len + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int token_start = core_id * tokens_per_core;
    int token_end = token_start + tokens_per_core;
    if (token_end > p->seq_len) token_end = p->seq_len;

#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
    const int32_t requant_mul = p->requant_mul;
    const int requant_shift = p->requant_shift;
    const int64_t requant_round = (requant_shift > 0) ? (1LL << (requant_shift - 1)) : 0;
#else
    const float requant_scale = p->scale_input * p->scale_weight / p->scale_output;
#endif

    const int simd_count = p->in_features >> 2;
    const int in_features = p->in_features;
    const int out_features_tile = p->out_features_tile;
    const int out_stride = p->out_stride;

#if LINEAR_INT8_INPUT_L1_CACHE
    const int cache_input = (in_features <= LINEAR_INT8_INPUT_L1_CACHE_MAX_BYTES);
    v4s input_cache[cache_input ? simd_count : 1] __attribute__((aligned(4)));
#endif

    for (int t = token_start; t < token_end; t++) {
        const int8_t *input_row = p->input + t * in_features;
        int8_t *output_row = p->output + t * out_stride;

        const v4s *pA = (const v4s *)input_row;
#if LINEAR_INT8_INPUT_L1_CACHE
        if (cache_input) {
            const v4s *pA_src = (const v4s *)input_row;
            for (int k = 0; k < simd_count; k++) {
                input_cache[k] = pA_src[k];
            }
            pA = (const v4s *)input_cache;
        }
#endif

        int o = 0;
        for (; o + 3 < out_features_tile; o += 4) {
            const v4s *pW0 = (const v4s *)(p->weights + o * in_features);
            const v4s *pW1 = (const v4s *)(p->weights + (o + 1) * in_features);
            const v4s *pW2 = (const v4s *)(p->weights + (o + 2) * in_features);
            const v4s *pW3 = (const v4s *)(p->weights + (o + 3) * in_features);

            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            for (int k = 0; k < simd_count; k++) {
                v4s a = pA[k];
                acc0 = SumDotpSS(a, pW0[k], acc0);
                acc1 = SumDotpSS(a, pW1[k], acc1);
                acc2 = SumDotpSS(a, pW2[k], acc2);
                acc3 = SumDotpSS(a, pW3[k], acc3);
            }

            if (p->bias) {
                acc0 += p->bias[o];
                acc1 += p->bias[o + 1];
                acc2 += p->bias[o + 2];
                acc3 += p->bias[o + 3];
            }

#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
            int64_t prod0 = (int64_t)acc0 * (int64_t)requant_mul;
            int64_t prod1 = (int64_t)acc1 * (int64_t)requant_mul;
            int64_t prod2 = (int64_t)acc2 * (int64_t)requant_mul;
            int64_t prod3 = (int64_t)acc3 * (int64_t)requant_mul;
            int32_t q0 = (int32_t)((prod0 + (prod0 >= 0 ? requant_round : -requant_round)) >> requant_shift);
            int32_t q1 = (int32_t)((prod1 + (prod1 >= 0 ? requant_round : -requant_round)) >> requant_shift);
            int32_t q2 = (int32_t)((prod2 + (prod2 >= 0 ? requant_round : -requant_round)) >> requant_shift);
            int32_t q3 = (int32_t)((prod3 + (prod3 >= 0 ? requant_round : -requant_round)) >> requant_shift);
#else
            int32_t q0 = qround((float)acc0 * requant_scale);
            int32_t q1 = qround((float)acc1 * requant_scale);
            int32_t q2 = qround((float)acc2 * requant_scale);
            int32_t q3 = qround((float)acc3 * requant_scale);
#endif
            output_row[o] = (int8_t)(q0 > 127 ? 127 : (q0 < -128 ? -128 : q0));
            output_row[o + 1] = (int8_t)(q1 > 127 ? 127 : (q1 < -128 ? -128 : q1));
            output_row[o + 2] = (int8_t)(q2 > 127 ? 127 : (q2 < -128 ? -128 : q2));
            output_row[o + 3] = (int8_t)(q3 > 127 ? 127 : (q3 < -128 ? -128 : q3));
        }

        // Tail (should be rare; tile sizes are chosen aligned)
        for (; o < out_features_tile; o++) {
            const v4s *pW = (const v4s *)(p->weights + o * in_features);
            int32_t acc = 0;
            for (int k = 0; k < simd_count; k++) {
                acc = SumDotpSS(pA[k], pW[k], acc);
            }
            if (p->bias) acc += p->bias[o];
#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
            int64_t prod = (int64_t)acc * (int64_t)requant_mul;
            int32_t q = (int32_t)((prod + (prod >= 0 ? requant_round : -requant_round)) >> requant_shift);
#else
            int32_t q = qround((float)acc * requant_scale);
#endif
            output_row[o] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
    }
}

void network_linear_int8_parallel_tokens_strided_out(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int8_t *output,
    int seq_len,
    int in_features,
    int out_features,
    int out_stride,
    float scale_input,
    float scale_weight,
    float scale_output
) {
    linear_strided_args_t args = {
        .input = input,
        .weights = weights,
        .bias = bias,
        .output = output,
        .seq_len = seq_len,
        .in_features = in_features,
        .out_features_tile = out_features,
        .out_stride = out_stride,
        .scale_input = scale_input,
        .scale_weight = scale_weight,
        .scale_output = scale_output
    };
#ifdef LINEAR_INT8_FIXEDPOINT_REQUANT
    args.requant_shift = 24;
    float requant_scale = scale_input * scale_weight / scale_output;
    args.requant_mul = qround(requant_scale * (float)(1 << args.requant_shift));
#endif
    pi_cl_team_fork(CL_NUM_CORES, linear_projection_strided_worker, &args);
}

// ---
// LINEAR KERNEL (INT8 -> FP32) - SIMD Optimized
// ---
// Used for the final classifier
void network_linear_int8_to_fp32(
    const int8_t *input, const int8_t *weights, const void *bias, float *output,
    uint16_t in_features, uint16_t out_features,
    float scale_input, float scale_weight
) {
    int core_id = pi_core_id();
    int chunk = (out_features + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start_f = core_id * chunk;
    int end_f = (start_f + chunk > out_features) ? out_features : (start_f + chunk);

    // Precompute combined scale
    const float combined_scale = scale_input * scale_weight;
    const float *bias_f = (const float *)bias;

    // SIMD loop count
    const int simd_count = in_features >> 2;
    const int remainder = in_features & 0x3;

#if !defined(DISABLE_LINEAR_INT8_TO_FP32_OUTCH_UNROLL2)
    int i = start_f;
    for (; i + 1 < end_f; i += 2) {
        int32_t acc0 = 0;
        int32_t acc1 = 0;

        const int8_t *w_row0 = weights + i * in_features;
        const int8_t *w_row1 = w_row0 + in_features;

        const v4s *pA = (const v4s *)input;
        const v4s *pB0 = (const v4s *)w_row0;
        const v4s *pB1 = (const v4s *)w_row1;
#if !defined(DISABLE_LINEAR_INT8_TO_FP32_SIMD_UNROLL2)
        int j = 0;
        for (; j + 1 < simd_count; j += 2) {
            v4s a0 = pA[0];
            v4s a1 = pA[1];
            v4s b00 = pB0[0];
            v4s b01 = pB0[1];
            v4s b10 = pB1[0];
            v4s b11 = pB1[1];
            acc0 = SumDotpSS(a0, b00, acc0);
            acc1 = SumDotpSS(a0, b10, acc1);
            acc0 = SumDotpSS(a1, b01, acc0);
            acc1 = SumDotpSS(a1, b11, acc1);
            pA += 2;
            pB0 += 2;
            pB1 += 2;
        }
        for (; j < simd_count; j++) {
            v4s a = *pA++;
            acc0 = SumDotpSS(a, *pB0++, acc0);
            acc1 = SumDotpSS(a, *pB1++, acc1);
        }
#else
        for (int j = 0; j < simd_count; j++) {
            v4s a = *pA++;
            acc0 = SumDotpSS(a, *pB0++, acc0);
            acc1 = SumDotpSS(a, *pB1++, acc1);
        }
#endif

        const int8_t *pA_rem = input + (simd_count << 2);
        const int8_t *pB0_rem = w_row0 + (simd_count << 2);
        const int8_t *pB1_rem = w_row1 + (simd_count << 2);
        for (int j = 0; j < remainder; j++) {
            int32_t a = (int32_t)pA_rem[j];
            acc0 += a * (int32_t)pB0_rem[j];
            acc1 += a * (int32_t)pB1_rem[j];
        }

        float val0_fp32 = (float)acc0 * combined_scale;
        float val1_fp32 = (float)acc1 * combined_scale;
        if (bias_f) {
            val0_fp32 += bias_f[i];
            val1_fp32 += bias_f[i + 1];
        }
        output[i] = val0_fp32;
        output[i + 1] = val1_fp32;
    }

    for (; i < end_f; i++) {
        int32_t acc = 0;
        const int8_t *w_row = weights + i * in_features;

        const v4s *pA = (const v4s *)input;
        const v4s *pB = (const v4s *)w_row;
#if !defined(DISABLE_LINEAR_INT8_TO_FP32_SIMD_UNROLL2)
        int j = 0;
        for (; j + 1 < simd_count; j += 2) {
            v4s a0 = pA[0];
            v4s a1 = pA[1];
            v4s b0 = pB[0];
            v4s b1 = pB[1];
            acc = SumDotpSS(a0, b0, acc);
            acc = SumDotpSS(a1, b1, acc);
            pA += 2;
            pB += 2;
        }
        for (; j < simd_count; j++) {
            acc = SumDotpSS(*pA++, *pB++, acc);
        }
#else
        for (int j = 0; j < simd_count; j++) {
            acc = SumDotpSS(*pA++, *pB++, acc);
        }
#endif

        const int8_t *pA_rem = input + (simd_count << 2);
        const int8_t *pB_rem = w_row + (simd_count << 2);
        for (int j = 0; j < remainder; j++) {
            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
        }

        float val_fp32 = (float)acc * combined_scale;
        if (bias_f) val_fp32 += bias_f[i];
        output[i] = val_fp32;
    }
#else
    for (int i = start_f; i < end_f; i++) {
        int32_t acc = 0;
        const int8_t *w_row = weights + i * in_features;

        const v4s *pA = (const v4s *)input;
        const v4s *pB = (const v4s *)w_row;
#if !defined(DISABLE_LINEAR_INT8_TO_FP32_SIMD_UNROLL2)
        int j = 0;
        for (; j + 1 < simd_count; j += 2) {
            v4s a0 = pA[0];
            v4s a1 = pA[1];
            v4s b0 = pB[0];
            v4s b1 = pB[1];
            acc = SumDotpSS(a0, b0, acc);
            acc = SumDotpSS(a1, b1, acc);
            pA += 2;
            pB += 2;
        }
        for (; j < simd_count; j++) {
            acc = SumDotpSS(*pA++, *pB++, acc);
        }
#else
        for (int j = 0; j < simd_count; j++) {
            acc = SumDotpSS(*pA++, *pB++, acc);
        }
#endif

        const int8_t *pA_rem = input + (simd_count << 2);
        const int8_t *pB_rem = w_row + (simd_count << 2);
        for (int j = 0; j < remainder; j++) {
            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
        }

        float val_fp32 = (float)acc * combined_scale;
        if (bias_f) val_fp32 += bias_f[i];
        output[i] = val_fp32;
    }
#endif
}

