/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Linear Tile Workers for ARES Runtime
 *
 * L1-tiled and pipelined linear operations for efficient execution.
 * These workers are used by the network executor for tiled linear layers.
 */

#include "ops/op_linear.h"
#include "network_kernels.h"
#include "pmsis.h"


void linear_tile_worker(void *arg) {
    linear_tile_args_t *t = (linear_tile_args_t *)arg;
    network_linear_int8(t->input_l1, t->weights_l2, t->bias_l2, t->output_l1,
        t->dim_in, t->dim_out, t->scale_input, t->scale_weight, t->scale_output);
}

void linear_tile_batched_worker(void *arg) {
    linear_tile_batched_args_t *t = (linear_tile_batched_args_t *)arg;
    const int batch_tokens = t->batch_tokens;
    const int dim_in = t->dim_in;
    const int dim_out = t->dim_out;

    for (int tok = 0; tok < batch_tokens; tok++) {
        const int8_t *input_t = t->input_base + tok * dim_in;
        int8_t *output_t = t->output_base + tok * dim_out;
        network_linear_int8(input_t, t->weights_l2, t->bias_l2, output_t,
            dim_in, dim_out, t->scale_input, t->scale_weight, t->scale_output);
    }
}

void linear_to_fp32_tile_worker(void *arg) {
    linear_to_fp32_tile_args_t *t = (linear_to_fp32_tile_args_t *)arg;
    network_linear_int8_to_fp32(t->input_l1, t->weights_l2, t->bias_l2, t->output_l1,
        t->dim_in, t->dim_out, t->scale_input, t->scale_weight);
}

void linear_int8_sequential_strided(
    const int8_t *input, const int8_t *weights, const float *bias, int8_t *output,
    uint16_t in_features, uint16_t out_features, uint16_t out_stride,
    float scale_input, float scale_weight, float scale_output
) {
    const float combined_scale = scale_input * scale_weight / scale_output;
    const int simd_count = in_features >> 2;
    const int remainder = in_features & 0x3;

    for (int i = 0; i < out_features; i++) {
        int32_t acc = 0;
        const int8_t *w_row = weights + i * in_features;
        const v4s *pA = (const v4s *)input;
        const v4s *pB = (const v4s *)w_row;
        for (int j = 0; j < simd_count; j++) {
            acc = SumDotpSS(pA[j], pB[j], acc);
        }
        const int8_t *pA_rem = input + (simd_count << 2);
        const int8_t *pB_rem = w_row + (simd_count << 2);
        for (int j = 0; j < remainder; j++) {
            acc += (int32_t)pA_rem[j] * (int32_t)pB_rem[j];
        }
        if (bias) {
            acc += ((const int32_t *)bias)[i];
        }
        float val_fp32 = (float)acc * combined_scale;
        int32_t q = qround(val_fp32);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        output[i * out_stride] = (int8_t)q;
    }
}

/**
 * L1-tiled linear worker: OPTIMIZED VERSION with 4x output + 4x timestep unrolling
 * - Loads input vector ONCE and reuses across 4 output features
 * - Loads weight vector ONCE and reuses across 4 timesteps
 * - Achieves ~8-12 MACs/cycle vs ~0.05 in naive implementation
 */
void linear_int8_l1_worker(void *arg) {
    linear_int8_l1_args_t *a = (linear_int8_l1_args_t *)arg;
    int core_id = pi_core_id();

    int out_per_core = (a->out_features + NUM_CORES - 1) / NUM_CORES;
    out_per_core = (out_per_core + 3) & ~3;
    int out_start = core_id * out_per_core;
    int out_end = (out_start + out_per_core > a->out_features) ? a->out_features : (out_start + out_per_core);

    if (out_start >= a->out_features) {
        pi_cl_team_barrier();
        return;
    }

    const float combined_scale = a->scale_in * a->scale_w / a->scale_out;
    const int simd_count = a->in_features >> 2;
    const int in_features = a->in_features;
    const int out_features = a->out_features;
    const int batch_tokens = a->batch_tokens;

    int t = 0;

    for (; t + 4 <= batch_tokens; t += 4) {
        const v4s *pIn0 = (const v4s *)(a->input + (t + 0) * in_features);
        const v4s *pIn1 = (const v4s *)(a->input + (t + 1) * in_features);
        const v4s *pIn2 = (const v4s *)(a->input + (t + 2) * in_features);
        const v4s *pIn3 = (const v4s *)(a->input + (t + 3) * in_features);

        int8_t *pOut0 = a->output + (t + 0) * out_features;
        int8_t *pOut1 = a->output + (t + 1) * out_features;
        int8_t *pOut2 = a->output + (t + 2) * out_features;
        int8_t *pOut3 = a->output + (t + 3) * out_features;

        int o = out_start;

        for (; o + 4 <= out_end; o += 4) {
            const v4s *pW0 = (const v4s *)(a->weights_l1 + (o + 0) * in_features);
            const v4s *pW1 = (const v4s *)(a->weights_l1 + (o + 1) * in_features);
            const v4s *pW2 = (const v4s *)(a->weights_l1 + (o + 2) * in_features);
            const v4s *pW3 = (const v4s *)(a->weights_l1 + (o + 3) * in_features);

            int32_t acc00 = 0, acc01 = 0, acc02 = 0, acc03 = 0;
            int32_t acc10 = 0, acc11 = 0, acc12 = 0, acc13 = 0;
            int32_t acc20 = 0, acc21 = 0, acc22 = 0, acc23 = 0;
            int32_t acc30 = 0, acc31 = 0, acc32 = 0, acc33 = 0;

            int k = 0;
            int simd_count_even = simd_count & ~1;
            for (; k < simd_count_even; k += 2) {
                v4s w0_a = pW0[k], w1_a = pW1[k], w2_a = pW2[k], w3_a = pW3[k];
                v4s in0_a = pIn0[k], in1_a = pIn1[k], in2_a = pIn2[k], in3_a = pIn3[k];

                v4s w0_b = pW0[k+1], w1_b = pW1[k+1], w2_b = pW2[k+1], w3_b = pW3[k+1];
                v4s in0_b = pIn0[k+1], in1_b = pIn1[k+1], in2_b = pIn2[k+1], in3_b = pIn3[k+1];

                acc00 = SumDotpSS(in0_a, w0_a, acc00);
                acc01 = SumDotpSS(in0_a, w1_a, acc01);
                acc02 = SumDotpSS(in0_a, w2_a, acc02);
                acc03 = SumDotpSS(in0_a, w3_a, acc03);
                acc10 = SumDotpSS(in1_a, w0_a, acc10);
                acc11 = SumDotpSS(in1_a, w1_a, acc11);
                acc12 = SumDotpSS(in1_a, w2_a, acc12);
                acc13 = SumDotpSS(in1_a, w3_a, acc13);
                acc20 = SumDotpSS(in2_a, w0_a, acc20);
                acc21 = SumDotpSS(in2_a, w1_a, acc21);
                acc22 = SumDotpSS(in2_a, w2_a, acc22);
                acc23 = SumDotpSS(in2_a, w3_a, acc23);
                acc30 = SumDotpSS(in3_a, w0_a, acc30);
                acc31 = SumDotpSS(in3_a, w1_a, acc31);
                acc32 = SumDotpSS(in3_a, w2_a, acc32);
                acc33 = SumDotpSS(in3_a, w3_a, acc33);

                acc00 = SumDotpSS(in0_b, w0_b, acc00);
                acc01 = SumDotpSS(in0_b, w1_b, acc01);
                acc02 = SumDotpSS(in0_b, w2_b, acc02);
                acc03 = SumDotpSS(in0_b, w3_b, acc03);
                acc10 = SumDotpSS(in1_b, w0_b, acc10);
                acc11 = SumDotpSS(in1_b, w1_b, acc11);
                acc12 = SumDotpSS(in1_b, w2_b, acc12);
                acc13 = SumDotpSS(in1_b, w3_b, acc13);
                acc20 = SumDotpSS(in2_b, w0_b, acc20);
                acc21 = SumDotpSS(in2_b, w1_b, acc21);
                acc22 = SumDotpSS(in2_b, w2_b, acc22);
                acc23 = SumDotpSS(in2_b, w3_b, acc23);
                acc30 = SumDotpSS(in3_b, w0_b, acc30);
                acc31 = SumDotpSS(in3_b, w1_b, acc31);
                acc32 = SumDotpSS(in3_b, w2_b, acc32);
                acc33 = SumDotpSS(in3_b, w3_b, acc33);
            }

            if (k < simd_count) {
                v4s w0 = pW0[k], w1 = pW1[k], w2 = pW2[k], w3 = pW3[k];
                v4s in0 = pIn0[k], in1 = pIn1[k], in2 = pIn2[k], in3 = pIn3[k];
                acc00 = SumDotpSS(in0, w0, acc00);
                acc01 = SumDotpSS(in0, w1, acc01);
                acc02 = SumDotpSS(in0, w2, acc02);
                acc03 = SumDotpSS(in0, w3, acc03);
                acc10 = SumDotpSS(in1, w0, acc10);
                acc11 = SumDotpSS(in1, w1, acc11);
                acc12 = SumDotpSS(in1, w2, acc12);
                acc13 = SumDotpSS(in1, w3, acc13);
                acc20 = SumDotpSS(in2, w0, acc20);
                acc21 = SumDotpSS(in2, w1, acc21);
                acc22 = SumDotpSS(in2, w2, acc22);
                acc23 = SumDotpSS(in2, w3, acc23);
                acc30 = SumDotpSS(in3, w0, acc30);
                acc31 = SumDotpSS(in3, w1, acc31);
                acc32 = SumDotpSS(in3, w2, acc32);
                acc33 = SumDotpSS(in3, w3, acc33);
            }

            if (a->bias) {
                int32_t b0 = a->bias[o + 0];
                int32_t b1 = a->bias[o + 1];
                int32_t b2 = a->bias[o + 2];
                int32_t b3 = a->bias[o + 3];

                acc00 += b0; acc01 += b1; acc02 += b2; acc03 += b3;
                acc10 += b0; acc11 += b1; acc12 += b2; acc13 += b3;
                acc20 += b0; acc21 += b1; acc22 += b2; acc23 += b3;
                acc30 += b0; acc31 += b1; acc32 += b2; acc33 += b3;
            }

            #define REQUANT_STORE(acc, out_ptr, out_idx) do { \
                int32_t q = qround((float)(acc) * combined_scale); \
                (out_ptr)[(out_idx)] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q)); \
            } while(0)

            REQUANT_STORE(acc00, pOut0, o + 0);
            REQUANT_STORE(acc01, pOut0, o + 1);
            REQUANT_STORE(acc02, pOut0, o + 2);
            REQUANT_STORE(acc03, pOut0, o + 3);

            REQUANT_STORE(acc10, pOut1, o + 0);
            REQUANT_STORE(acc11, pOut1, o + 1);
            REQUANT_STORE(acc12, pOut1, o + 2);
            REQUANT_STORE(acc13, pOut1, o + 3);

            REQUANT_STORE(acc20, pOut2, o + 0);
            REQUANT_STORE(acc21, pOut2, o + 1);
            REQUANT_STORE(acc22, pOut2, o + 2);
            REQUANT_STORE(acc23, pOut2, o + 3);

            REQUANT_STORE(acc30, pOut3, o + 0);
            REQUANT_STORE(acc31, pOut3, o + 1);
            REQUANT_STORE(acc32, pOut3, o + 2);
            REQUANT_STORE(acc33, pOut3, o + 3);

            #undef REQUANT_STORE
        }

        for (; o < out_end; o++) {
            const v4s *pW = (const v4s *)(a->weights_l1 + o * in_features);
            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

            for (int k = 0; k < simd_count; k++) {
                v4s w = pW[k];
                acc0 = SumDotpSS(pIn0[k], w, acc0);
                acc1 = SumDotpSS(pIn1[k], w, acc1);
                acc2 = SumDotpSS(pIn2[k], w, acc2);
                acc3 = SumDotpSS(pIn3[k], w, acc3);
            }

            if (a->bias) {
                int32_t b = a->bias[o];
                acc0 += b; acc1 += b; acc2 += b; acc3 += b;
            }

            int32_t q0 = qround((float)acc0 * combined_scale);
            int32_t q1 = qround((float)acc1 * combined_scale);
            int32_t q2 = qround((float)acc2 * combined_scale);
            int32_t q3 = qround((float)acc3 * combined_scale);

            pOut0[o] = (int8_t)(q0 > 127 ? 127 : (q0 < -128 ? -128 : q0));
            pOut1[o] = (int8_t)(q1 > 127 ? 127 : (q1 < -128 ? -128 : q1));
            pOut2[o] = (int8_t)(q2 > 127 ? 127 : (q2 < -128 ? -128 : q2));
            pOut3[o] = (int8_t)(q3 > 127 ? 127 : (q3 < -128 ? -128 : q3));
        }
    }

    for (; t < batch_tokens; t++) {
        const v4s *pIn = (const v4s *)(a->input + t * in_features);
        int8_t *pOut = a->output + t * out_features;

        int o = out_start;
        for (; o + 4 <= out_end; o += 4) {
            const v4s *pW0 = (const v4s *)(a->weights_l1 + (o + 0) * in_features);
            const v4s *pW1 = (const v4s *)(a->weights_l1 + (o + 1) * in_features);
            const v4s *pW2 = (const v4s *)(a->weights_l1 + (o + 2) * in_features);
            const v4s *pW3 = (const v4s *)(a->weights_l1 + (o + 3) * in_features);

            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

            for (int k = 0; k < simd_count; k++) {
                v4s in = pIn[k];
                acc0 = SumDotpSS(in, pW0[k], acc0);
                acc1 = SumDotpSS(in, pW1[k], acc1);
                acc2 = SumDotpSS(in, pW2[k], acc2);
                acc3 = SumDotpSS(in, pW3[k], acc3);
            }

            if (a->bias) {
                acc0 += a->bias[o + 0];
                acc1 += a->bias[o + 1];
                acc2 += a->bias[o + 2];
                acc3 += a->bias[o + 3];
            }

            int32_t q0 = qround((float)acc0 * combined_scale);
            int32_t q1 = qround((float)acc1 * combined_scale);
            int32_t q2 = qround((float)acc2 * combined_scale);
            int32_t q3 = qround((float)acc3 * combined_scale);

            pOut[o + 0] = (int8_t)(q0 > 127 ? 127 : (q0 < -128 ? -128 : q0));
            pOut[o + 1] = (int8_t)(q1 > 127 ? 127 : (q1 < -128 ? -128 : q1));
            pOut[o + 2] = (int8_t)(q2 > 127 ? 127 : (q2 < -128 ? -128 : q2));
            pOut[o + 3] = (int8_t)(q3 > 127 ? 127 : (q3 < -128 ? -128 : q3));
        }

        for (; o < out_end; o++) {
            const v4s *pW = (const v4s *)(a->weights_l1 + o * in_features);
            int32_t acc = 0;

            for (int k = 0; k < simd_count; k++) {
                acc = SumDotpSS(pIn[k], pW[k], acc);
            }

            if (a->bias) acc += a->bias[o];

            int32_t q = qround((float)acc * combined_scale);
            pOut[o] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
    }

    pi_cl_team_barrier();
}
