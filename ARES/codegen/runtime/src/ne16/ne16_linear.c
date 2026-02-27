/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * NE16 Linear Layer Kernels
 *
 * High-level linear layer execution using the NE16 accelerator.
 * Provides parallel input conversion, NE16 execution, and postprocessing.
 *
 * Key optimization: Pre-packed weights eliminate runtime packing overhead.
 */

#include "ne16/ne16_driver.h"
#include "network_kernels.h"

#ifdef ARES_USE_NE16

#include <stddef.h>
#include <string.h>
#include "pmsis.h"

/* --- Worker Argument Structures --- */

typedef struct {
    const int8_t *in_s8;
    uint8_t *out_u8;
    int len_bytes;
    int input_zp;
} ne16_inconv_args_t;

typedef struct {
    const int32_t *acc_s32;    /* [num_tokens, out_features] */
    const int32_t *bias_corr;  /* [out_features] */
    int8_t *out_s8;            /* Base pointer for output */
    int out_stride;            /* Stride between tokens in output */
    int num_tokens;
    int out_features;
    float combined_scale;
} ne16_post_args_t;

typedef struct {
    const uint8_t *in_u8;
    const uint8_t *w_packed;
    int32_t *out_s32;
    int num_tokens;
    int in_features;
    int out_features;
    int8_t weight_offset;
} ne16_run_args_t;

typedef struct {
    const int8_t *src;     /* Contiguous [num_tokens, out_features] */
    int8_t *dst;           /* Strided [num_tokens, out_stride] */
    int out_features;
    int out_stride;
    int num_tokens;
} ne16_scatter_args_t;

/* --- Outquant Scale Generation (reference-compatible) --- */

void ne16_compute_outquant_scale(float scale, uint8_t *qbias_out, uint8_t *qnorm_out)
{
    if (!qbias_out || !qnorm_out) {
        return;
    }

    /* Reject non-positive, NaN, and Inf */
    union {
        float f;
        uint32_t u;
    } scale_bits = { .f = scale };
    const uint32_t exp_bits = (scale_bits.u >> 23) & 0xFFu;
    if (!(scale > 0.0f) || exp_bits == 0xFFu) {
        *qbias_out = 0;
        *qnorm_out = 0;
        return;
    }

    /* Minimal frexp(scale) replacement to avoid libm dependencies on this toolchain.
     * Returns mantissa in [0.5, 1) and exponent such that:
     *   scale = mantissa * 2^exponent
     */
    int exponent = 0;
    float mantissa = 0.0f;

    if (scale == 0.0f) {
        *qbias_out = 0;
        *qnorm_out = 0;
        return;
    }

    if (exp_bits == 0) {
        /* Subnormal: normalize by scaling into normal range. */
        float x = scale;
        exponent = 0;
        while (x < 0.5f) {
            x *= 2.0f;
            exponent--;
            if (exponent < -200) break; /* safety */
        }
        while (x >= 1.0f) {
            x *= 0.5f;
            exponent++;
            if (exponent > 200) break; /* safety */
        }
        mantissa = x;
    } else {
        /* Normal IEEE-754 float */
        const int e_unbiased = (int)exp_bits - 127;
        const uint32_t frac = scale_bits.u & 0x7FFFFFu;
        mantissa = (1.0f + ((float)frac / 8388608.0f)) * 0.5f; /* (1.frac)/2 */
        exponent = e_unbiased + 1;
    }

    /* Port of the reference compute_scales() flow (GAP SDK):
     * - qbias = floor(mantissa * 2^bits + 0.5)
     * - qnorm = bits - exponent
     * - Decrease bits while (qbias overflows) or (qnorm too large) or (qbias even),
     *   as long as (qnorm>0 && bits>0).
     */
    int bits = 8;               /* uint8 norm params */
    const int max_val = 256;    /* 2^8 */

    while (1) {
        const float scaled = mantissa * (float)(1u << (unsigned)bits);
        int32_t qbias = (int32_t)(scaled + 0.5f); /* truncation == floor for positive */
        int32_t qnorm = (int32_t)bits - (int32_t)exponent;

        const int max_exceeded = (qbias >= max_val);
        const int norms_too_high = (qnorm > (32 - 8));
        const int bias_pow2 = ((qbias & 0x1) == 0); /* even */
        const int should_move = max_exceeded || norms_too_high || bias_pow2;
        const int can_still_move = (qnorm > 0) && (bits > 0);

        if (!(should_move && can_still_move)) {
            if (qbias < 0) qbias = 0;
            if (qbias > 255) qbias = 255;
            if (qnorm < 0) qnorm = 0;
            if (qnorm > 255) qnorm = 255;
            *qbias_out = (uint8_t)qbias;
            *qnorm_out = (uint8_t)qnorm;
            return;
        }

        bits--;
    }
}

/* --- Parallel Worker Functions --- */

/**
 * Convert signed INT8 input to unsigned INT8 (parallel across cores).
 *
 * NE16 requires unsigned inputs, so we add input_zp (typically 128)
 * to convert from signed [-128, 127] to unsigned [0, 255] domain.
 */
static void ne16_input_s8_to_u8_worker(void *args)
{
    const ne16_inconv_args_t *a = (const ne16_inconv_args_t *)args;
    const int core_id = pi_core_id();
    const int chunk = (a->len_bytes + NUM_CORES - 1) / NUM_CORES;
    const int start = core_id * chunk;
    const int end = (start + chunk > a->len_bytes) ? a->len_bytes : (start + chunk);

    for (int i = start; i < end; i++) {
        a->out_u8[i] = (uint8_t)((int)a->in_s8[i] + a->input_zp);
    }
}

/**
 * Postprocess INT32 accumulators to INT8 output (parallel across cores).
 *
 * Applies bias correction and requantization:
 *   output = clamp(round((acc + bias_corr) * combined_scale), -128, 127)
 */
static void ne16_postprocess_s32_to_s8_worker(void *args)
{
    const ne16_post_args_t *a = (const ne16_post_args_t *)args;
    const int core_id = pi_core_id();
    const int chunk = (a->num_tokens + NUM_CORES - 1) / NUM_CORES;
    const int start_t = core_id * chunk;
    const int end_t = (start_t + chunk > a->num_tokens) ? a->num_tokens : (start_t + chunk);

    for (int t = start_t; t < end_t; t++) {
        const int32_t *acc_row = a->acc_s32 + (size_t)t * (size_t)a->out_features;
        int8_t *out_row = a->out_s8 + (size_t)t * (size_t)a->out_stride;
        for (int o = 0; o < a->out_features; o++) {
            int32_t acc = acc_row[o] + a->bias_corr[o];
            float val_fp32 = (float)acc * a->combined_scale;
            int32_t q = qround(val_fp32);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_row[o] = (int8_t)q;
        }
    }
}

/**
 * Scatter contiguous INT8 tile output into a strided output layout.
 *
 * NE16 OUTQUANT streamout writes contiguous `[num_tokens, out_features]` tiles.
 * Some ARES kernels use a larger `out_stride` for the final output layout.
 */
static void ne16_scatter_s8_worker(void *args)
{
    const ne16_scatter_args_t *a = (const ne16_scatter_args_t *)args;
    const int core_id = pi_core_id();
    const int chunk = (a->num_tokens + NUM_CORES - 1) / NUM_CORES;
    const int start_t = core_id * chunk;
    const int end_t = (start_t + chunk > a->num_tokens) ? a->num_tokens : (start_t + chunk);

    for (int t = start_t; t < end_t; t++) {
        memcpy(a->dst + (size_t)t * (size_t)a->out_stride,
               a->src + (size_t)t * (size_t)a->out_features,
               (size_t)a->out_features);
    }
}

/**
 * Run NE16 1x1 convolution from core 0 (single-threaded NE16 programming).
 *
 * NE16 should only be programmed from one core to avoid contention.
 */
static void ne16_run_1x1_worker(void *args)
{
    const ne16_run_args_t *a = (const ne16_run_args_t *)args;
    /* Only core 0 programs and runs NE16 */
    if (pi_core_id() == 0) {
        ne16_init();
        ne16_conv1x1_u8_u8_to_s32(
            a->in_u8,
            a->w_packed,
            a->out_s32,
            /*in_w=*/a->num_tokens,
            /*in_h=*/1,
            /*in_feat=*/a->in_features,
            /*out_feat=*/a->out_features,
            a->weight_offset
        );
    }
}

/* ---
 * DIAGNOSTIC: Runtime Weight Packing
 *
 * This mimics tinymyo's runtime packing to test if offline packing has issues.
 * --- */

typedef struct {
    const int8_t *w_s8;      /* Original S8 weights [out_features, in_features] */
    uint8_t *out_packed;     /* Output packed buffer */
    int in_features;
    int out_features;
} ne16_pack_args_t;

/**
 * Runtime weight packing worker - identical to tinymyo's algorithm
 */
static void ne16_pack_weights_runtime_worker(void *args)
{
    const ne16_pack_args_t *a = (const ne16_pack_args_t *)args;
    const int nb_ki = (a->in_features + 15) / 16;
    const int core_id = pi_core_id();
    const int chunk = (a->out_features + NUM_CORES - 1) / NUM_CORES;
    const int start_ko = core_id * chunk;
    const int end_ko = (start_ko + chunk > a->out_features) ? a->out_features : (start_ko + chunk);

    for (int ko = start_ko; ko < end_ko; ko++) {
        const int8_t *w_row = a->w_s8 + ko * a->in_features;
        uint8_t *dst_ko = a->out_packed + (size_t)ko * nb_ki * 16;

        for (int kimaj = 0; kimaj < nb_ki; kimaj++) {
            uint8_t dst_bits[16] = {0};  /* 8 bits * 2 bytes = 16 bytes */
            for (int kimin = 0; kimin < 16; kimin++) {
                int idx = kimaj * 16 + kimin;
                uint8_t w_u8_raw;
                if (idx < a->in_features) {
                    w_u8_raw = (uint8_t)(w_row[idx] + 128);  /* S8 -> U8 */
                } else {
                    w_u8_raw = 128;  /* Zero-pad */
                }
                /* Extract each bit into b0[7:0] (kimin=0..7) or b1[7:0] (kimin=8..15) */
                for (int bit = 0; bit < 8; bit++) {
                    int bit_val = (w_u8_raw >> bit) & 1;
                    if (kimin < 8) {
                        dst_bits[bit * 2] |= (bit_val << kimin);
                    } else {
                        dst_bits[bit * 2 + 1] |= (bit_val << (kimin - 8));
                    }
                }
            }
            memcpy(dst_ko + kimaj * 16, dst_bits, 16);
        }
    }
}

/**
 * Pack weights at runtime (diagnostic - compare with offline packing)
 */
void ne16_pack_weights_runtime(
    const int8_t *w_s8,
    uint8_t *out_packed,
    int in_features,
    int out_features
) {
    /* DEBUG: Print original S8 weights before packing */
#if defined(ARES_NE16_DEBUG) && !defined(MINIMAL_OUTPUT)
    printf("CL: NE16 PACK DEBUG: w_s8=%p in_features=%d out_features=%d\n",
           (void *)w_s8, in_features, out_features);
    printf("CL: NE16 PACK: Original S8 weights ch0[0..7]: %d %d %d %d %d %d %d %d\n",
           w_s8[0], w_s8[1], w_s8[2], w_s8[3], w_s8[4], w_s8[5], w_s8[6], w_s8[7]);
    printf("CL: NE16 PACK: Original S8 weights ch1[0..7]: %d %d %d %d %d %d %d %d\n",
           w_s8[in_features + 0], w_s8[in_features + 1], w_s8[in_features + 2], w_s8[in_features + 3],
           w_s8[in_features + 4], w_s8[in_features + 5], w_s8[in_features + 6], w_s8[in_features + 7]);
#endif

    ne16_pack_args_t args = {
        .w_s8 = w_s8,
        .out_packed = out_packed,
        .in_features = in_features,
        .out_features = out_features,
    };
    pi_cl_team_fork(NUM_CORES, ne16_pack_weights_runtime_worker, &args);
}

/* ---
 * TINYMYO-STYLE IN-PLACE PACKING
 *
 * Exact copy of tinymyo's algorithm - pack weights in-place, overwriting
 * the original S8 weights with the packed format.
 * --- */

typedef struct {
    int8_t *weights;     /* IN-PLACE: overwritten with packed format */
    int32_t *bias_corr;
    const int32_t *bias;
    int in_features;
    int out_features;
    int input_zp;        /* Typically 128 */
    int weight_offset;   /* Typically -128 */
} ne16_pack_inplace_args_t;

static void ne16_pack_weights_inplace_worker(void *args)
{
    const ne16_pack_inplace_args_t *a = (const ne16_pack_inplace_args_t *)args;
    const int core_id = pi_core_id();
    const int chunk = (a->out_features + NUM_CORES - 1) / NUM_CORES;
    const int start = core_id * chunk;
    const int end = (start + chunk > a->out_features) ? a->out_features : (start + chunk);

    const int in_feat = a->in_features;
    const int nb_ki = (in_feat + 15) / 16;

    for (int ko = start; ko < end; ko++) {
        int32_t sum_w = 0;
        uint8_t *row = (uint8_t *)a->weights + (size_t)ko * (size_t)in_feat;

        for (int kimaj = 0; kimaj < nb_ki; kimaj++) {
            uint8_t w_u8[16];
            /* Read one KiMin=16 group (pad with zeros for tail) */
            for (int kimin = 0; kimin < 16; kimin++) {
                const int idx = kimaj * 16 + kimin;
                int8_t w_s8 = 0;
                if (idx < in_feat) {
                    w_s8 = *((int8_t *)row + idx);
                }
                sum_w += (int32_t)w_s8;

                const int w_shifted = (int)w_s8 - a->weight_offset;  /* unsigned storage domain */
                w_u8[kimin] = (uint8_t)w_shifted;
            }

            /* Pack to NE16 1x1 layout: [w_bit][byte] where byte packs 8 weights */
            const int base = kimaj * 16;
            for (int bit = 0; bit < 8; bit++) {
                uint8_t b0 = 0;
                uint8_t b1 = 0;
                for (int i = 0; i < 8; i++) {
                    b0 |= (uint8_t)(((w_u8[i] >> bit) & 0x1u) << i);
                    b1 |= (uint8_t)(((w_u8[i + 8] >> bit) & 0x1u) << i);
                }
                row[base + bit * 2 + 0] = b0;
                row[base + bit * 2 + 1] = b1;
            }
        }

        const int32_t base_bias = a->bias ? a->bias[ko] : 0;
        a->bias_corr[ko] = base_bias - (int32_t)a->input_zp * sum_w;
    }
}

/**
 * Pack weights in-place (tinymyo style) and compute bias correction.
 */
static void ne16_pack_weights_inplace(
    int8_t *weights,
    int32_t *bias_corr,
    const int32_t *bias,
    int in_features,
    int out_features
) {
    ne16_pack_inplace_args_t args = {
        .weights = weights,
        .bias_corr = bias_corr,
        .bias = bias,
        .in_features = in_features,
        .out_features = out_features,
        .input_zp = NE16_INPUT_ZP,       /* 128 */
        .weight_offset = NE16_WEIGHT_OFFSET,  /* -128 */
    };
    pi_cl_team_fork(NUM_CORES, ne16_pack_weights_inplace_worker, &args);
}

/**
 * Public version of in-place packing for use by executor.
 */
void ne16_pack_weights_inplace_with_bias(
    int8_t *weights,
    int32_t *bias_corr,
    const int32_t *bias,
    int in_features,
    int out_features
) {
    ne16_pack_weights_inplace(weights, bias_corr, bias, in_features, out_features);
}

/* --- Public API --- */

void ne16_linear_int8_packed(
    const int8_t *input,
    const int8_t *weights_packed,
    const int32_t *bias_corr,
    int8_t *output,
    int num_tokens,
    int in_features,
    int out_features,
    int out_stride,
    float scale_input,
    float scale_weight,
    float scale_output,
    int tile_tokens,
    uint8_t *input_u8_scratch,
    int32_t *output_s32_scratch
) {
    if (!input || !weights_packed || !bias_corr || !output ||
        !input_u8_scratch || !output_s32_scratch) {
        return;
    }

    /* Combined requant scale (matches software kernels) */
    const float combined_scale = scale_input * scale_weight / scale_output;
    const int input_zp = NE16_INPUT_ZP;
    const int weight_offset = NE16_WEIGHT_OFFSET;

    /* Tile over tokens to keep scratch buffers small */
    for (int t0 = 0; t0 < num_tokens; t0 += tile_tokens) {
        int stripe_len = tile_tokens;
        if (t0 + stripe_len > num_tokens) stripe_len = num_tokens - t0;

        /* Step 1: Convert input stripe to unsigned domain (parallel) */
        ne16_inconv_args_t inconv = {
            .in_s8 = input + (size_t)t0 * (size_t)in_features,
            .out_u8 = input_u8_scratch,
            .len_bytes = stripe_len * in_features,
            .input_zp = input_zp,
        };
        pi_cl_team_fork(NUM_CORES, ne16_input_s8_to_u8_worker, &inconv);

        /* Step 2: Run NE16 on PE core 0 */
#if defined(ARES_NE16_DEBUG) && !defined(MINIMAL_OUTPUT)
        /* DEBUG: Clear output buffer to verify NE16 writes */
        memset(output_s32_scratch, 0xCD, (size_t)stripe_len * (size_t)out_features * sizeof(int32_t));

        /* DEBUG: Force CPU read of weight data before NE16 to test cache coherency */
        volatile uint8_t weight_touch = 0;
        const uint8_t *w_ptr = (const uint8_t *)weights_packed;
        int weight_bytes = (in_features / 16) * 16 * out_features;
        for (int i = 0; i < weight_bytes; i += 64) {
            weight_touch ^= w_ptr[i];
        }
        (void)weight_touch;
        asm volatile("fence" ::: "memory");
#endif

        ne16_run_args_t run_args = {
            .in_u8 = input_u8_scratch,
            .w_packed = (const uint8_t *)weights_packed,
            .out_s32 = output_s32_scratch,
            .num_tokens = stripe_len,
            .in_features = in_features,
            .out_features = out_features,
            .weight_offset = (int8_t)weight_offset,
        };
        pi_cl_team_fork(1, ne16_run_1x1_worker, &run_args);

        /* DEBUG: Print NE16 output buffer to understand layout */
#if defined(ARES_NE16_DEBUG) && !defined(MINIMAL_OUTPUT)
        printf("CL: NE16 DEBUG: in_feat=%d out_feat=%d stripe_len=%d\n",
               in_features, out_features, stripe_len);
        printf("CL: NE16 DEBUG: input_u8[0..4] = %u %u %u %u %u\n",
               input_u8_scratch[0], input_u8_scratch[1], input_u8_scratch[2],
               input_u8_scratch[3], input_u8_scratch[4]);

        /* Check if NE16 wrote anything (should differ from 0xCDCDCDCD pattern) */
        uint32_t check_val = *((uint32_t *)output_s32_scratch);
        printf("CL: NE16 DEBUG: first word = 0x%08x (0xcdcdcdcd means NE16 didn't write)\n",
               check_val);

        /* Print first few raw INT32 values */
        printf("CL: NE16 DEBUG: raw acc[0..7] = ");
        for (int i = 0; i < 8 && i < out_features; i++) {
            printf("%ld ", (long)output_s32_scratch[i]);
        }
        printf("\n");

        /* Check further into the buffer - different Ko subtile? */
        int buffer_size = stripe_len * out_features;
        printf("CL: NE16 DEBUG: acc at stride 32 (different spatial?) = ");
        for (int i = 0; i < 4 && i * 32 < buffer_size; i++) {
            printf("[%d]=%ld ", i * 32, (long)output_s32_scratch[i * 32]);
        }
        printf("\n");
        printf("CL: NE16 DEBUG: acc at stride 64 (different Ko subtile?) = ");
        for (int i = 0; i < 4 && i * 64 < buffer_size; i++) {
            printf("[%d]=%ld ", i * 64, (long)output_s32_scratch[i * 64]);
        }
        printf("\n");
        /* Check consecutive values at token boundary */
        printf("CL: NE16 DEBUG: acc[64..71] (second token row?) = ");
        for (int i = 64; i < 72 && i < buffer_size; i++) {
            printf("%ld ", (long)output_s32_scratch[i]);
        }
        printf("\n");

        /* Detailed output analysis: print Token 0's output channels 0, 1, 31, 32, 63 */
        /* With d1=256 bytes (64 int32s), Token 0 is indices 0-63 */
        printf("CL: NE16 DEBUG: Token0 layout check: ch0=%ld ch1=%ld ch31=%ld ch32=%ld ch63=%ld\n",
               (long)output_s32_scratch[0],
               (long)output_s32_scratch[1],
               (long)(31 < out_features ? output_s32_scratch[31] : -1),
               (long)(32 < out_features ? output_s32_scratch[32] : -1),
               (long)(63 < out_features ? output_s32_scratch[63] : -1));
        /* Token 1 starts at index 64 for 64-channel output */
        if (stripe_len > 1 && out_features == 64) {
            printf("CL: NE16 DEBUG: Token1 layout check: ch0=%ld ch1=%ld ch31=%ld ch32=%ld ch63=%ld\n",
                   (long)output_s32_scratch[64],
                   (long)output_s32_scratch[65],
                   (long)output_s32_scratch[64 + 31],
                   (long)output_s32_scratch[64 + 32],
                   (long)output_s32_scratch[64 + 63]);
        }

        printf("CL: NE16 DEBUG: bias_corr[0..4] = %ld %ld %ld %ld %ld\n",
               (long)bias_corr[0], (long)bias_corr[1],
               (long)bias_corr[2], (long)bias_corr[3], (long)bias_corr[4]);
        printf("CL: NE16 DEBUG: combined_scale = %f\n", combined_scale);
#endif

        /* Step 3: Requantize + store (parallel) */
        ne16_post_args_t post = {
            .acc_s32 = output_s32_scratch,
            .bias_corr = bias_corr,
            .out_s8 = output + (size_t)t0 * (size_t)out_stride,
            .out_stride = out_stride,
            .num_tokens = stripe_len,
            .out_features = out_features,
            .combined_scale = combined_scale,
        };
        pi_cl_team_fork(NUM_CORES, ne16_postprocess_s32_to_s8_worker, &post);
    }
}

void ne16_linear_int8_packed_hw_requant(
    const int8_t *input,
    const int8_t *weights_packed,
    const int32_t *bias_corr,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *output,
    int num_tokens,
    int in_features,
    int out_features,
    int out_stride,
    int tile_tokens,
    uint8_t *input_u8_ping,
    uint8_t *input_u8_pong,
    int8_t *output_s8_tile_scratch
) {
    if (!input || !weights_packed || !bias_corr || !scale || !scale_shift || !output || !input_u8_ping) {
        return;
    }

    if (!output_s8_tile_scratch && out_stride != out_features) {
        return;
    }

    const int input_zp = NE16_INPUT_ZP;
    const int weight_offset = NE16_WEIGHT_OFFSET;

    if (tile_tokens <= 0) {
        tile_tokens = num_tokens;
    }

    /* Pipelined execution overlaps S8->U8 conversion with NE16 compute when a pong buffer is provided. */
    const int use_pipelined = (input_u8_pong != NULL);

    /* Output pointer for the NE16 streamout: prefer L1 tile scratch when provided. */
    int8_t *tile_out_ptr = output_s8_tile_scratch ? output_s8_tile_scratch : output;

    if (!use_pipelined) {
        /* ===== Serial: convert -> run NE16 -> scatter, per tile ===== */
        for (int t0 = 0; t0 < num_tokens; t0 += tile_tokens) {
            int stripe_len = tile_tokens;
            if (t0 + stripe_len > num_tokens) stripe_len = num_tokens - t0;

            ne16_inconv_args_t inconv = {
                .in_s8 = input + (size_t)t0 * (size_t)in_features,
                .out_u8 = input_u8_ping,
                .len_bytes = stripe_len * in_features,
                .input_zp = input_zp,
            };
            pi_cl_team_fork(NUM_CORES, ne16_input_s8_to_u8_worker, &inconv);

            asm volatile("fence iorw, iorw" ::: "memory");

            ne16_conv1x1_u8_u8_to_s8(
                input_u8_ping,
                (const uint8_t *)weights_packed,
                bias_corr,
                scale,
                scale_shift,
                tile_out_ptr,
                /*in_w=*/stripe_len,
                /*in_h=*/1,
                in_features,
                out_features,
                (int8_t)weight_offset
            );

            if (output_s8_tile_scratch) {
                ne16_scatter_args_t scatter = {
                    .src = tile_out_ptr,
                    .dst = output + (size_t)t0 * (size_t)out_stride,
                    .out_features = out_features,
                    .out_stride = out_stride,
                    .num_tokens = stripe_len,
                };
                pi_cl_team_fork(NUM_CORES, ne16_scatter_s8_worker, &scatter);
            }
        }
        return;
    }

    /* ===== Pipelined: convert tile N+1 while NE16 runs tile N ===== */
    const int num_tiles = (num_tokens + tile_tokens - 1) / tile_tokens;
    if (num_tiles <= 0) {
        return;
    }

    uint8_t *active_buf = input_u8_ping;
    uint8_t *next_buf = input_u8_pong;

    /* Prologue: convert and submit tile 0 */
    int tile0_len = tile_tokens;
    if (tile0_len > num_tokens) tile0_len = num_tokens;

    ne16_inconv_args_t conv0 = {
        .in_s8 = input,
        .out_u8 = active_buf,
        .len_bytes = tile0_len * in_features,
        .input_zp = input_zp,
    };
    pi_cl_team_fork(NUM_CORES, ne16_input_s8_to_u8_worker, &conv0);

    asm volatile("fence iorw, iorw" ::: "memory");
    int job_id = ne16_conv1x1_submit_async(
        active_buf,
        (const uint8_t *)weights_packed,
        bias_corr,
        scale,
        scale_shift,
        tile_out_ptr,
        /*in_w=*/tile0_len,
        /*in_h=*/1,
        in_features,
        out_features,
        (int8_t)weight_offset
    );

    for (int t = 1; t < num_tiles; t++) {
        const int tile_start = t * tile_tokens;
        int tile_len = tile_tokens;
        if (tile_start + tile_len > num_tokens) tile_len = num_tokens - tile_start;

        /* Convert next tile while NE16 runs previous tile */
        ne16_inconv_args_t conv_next = {
            .in_s8 = input + (size_t)tile_start * (size_t)in_features,
            .out_u8 = next_buf,
            .len_bytes = tile_len * in_features,
            .input_zp = input_zp,
        };
        pi_cl_team_fork(NUM_CORES, ne16_input_s8_to_u8_worker, &conv_next);

        /* Wait for previous job */
        ne16_wait_job(job_id);

        /* Scatter/copy previous tile output */
        if (output_s8_tile_scratch) {
            const int prev_tile_start = (t - 1) * tile_tokens;
            int prev_tile_len = tile_tokens;
            if (prev_tile_start + prev_tile_len > num_tokens) prev_tile_len = num_tokens - prev_tile_start;

            ne16_scatter_args_t scatter = {
                .src = tile_out_ptr,
                .dst = output + (size_t)prev_tile_start * (size_t)out_stride,
                .out_features = out_features,
                .out_stride = out_stride,
                .num_tokens = prev_tile_len,
            };
            pi_cl_team_fork(NUM_CORES, ne16_scatter_s8_worker, &scatter);
        }

        /* Swap ping/pong buffers */
        uint8_t *tmp = active_buf;
        active_buf = next_buf;
        next_buf = tmp;

        /* Submit next tile */
        asm volatile("fence iorw, iorw" ::: "memory");
        job_id = ne16_conv1x1_submit_async(
            active_buf,
            (const uint8_t *)weights_packed,
            bias_corr,
            scale,
            scale_shift,
            tile_out_ptr,
            /*in_w=*/tile_len,
            /*in_h=*/1,
            in_features,
            out_features,
            (int8_t)weight_offset
        );
    }

    /* Epilogue: wait for last tile and scatter */
    ne16_wait_job(job_id);
    if (output_s8_tile_scratch) {
        const int final_tile_start = (num_tiles - 1) * tile_tokens;
        int final_tile_len = tile_tokens;
        if (final_tile_start + final_tile_len > num_tokens) final_tile_len = num_tokens - final_tile_start;

        ne16_scatter_args_t scatter = {
            .src = tile_out_ptr,
            .dst = output + (size_t)final_tile_start * (size_t)out_stride,
            .out_features = out_features,
            .out_stride = out_stride,
            .num_tokens = final_tile_len,
        };
        pi_cl_team_fork(NUM_CORES, ne16_scatter_s8_worker, &scatter);
    }
}

void ne16_linear_int8_batch_packed(
    const int8_t *input,
    const int8_t *weights_packed,
    const int32_t *bias_corr,
    int8_t *output,
    int batch,
    int num_tokens,
    int in_features,
    int out_features,
    float scale_input,
    float scale_weight,
    float scale_output,
    int tile_tokens,
    uint8_t *input_u8_scratch,
    int32_t *output_s32_scratch
) {
    /* Process each batch sequentially, tokens within batch use NE16 */
    const int in_batch_stride = num_tokens * in_features;
    const int out_batch_stride = num_tokens * out_features;

    for (int b = 0; b < batch; b++) {
        ne16_linear_int8_packed(
            input + b * in_batch_stride,
            weights_packed,
            bias_corr,
            output + b * out_batch_stride,
            num_tokens,
            in_features,
            out_features,
            out_features,  /* contiguous output */
            scale_input,
            scale_weight,
            scale_output,
            tile_tokens,
            input_u8_scratch,
            output_s32_scratch
        );
    }
}

/* ---
 * Pipelined NE16 Linear Execution
 *
 * Overlaps CPU input conversion (S8->U8) with NE16 compute:
 *
 *   Tile 0: [S8->U8 (8 cores)] -> [NE16 tile 0]
 *   Tile 1:           [S8->U8 (8 cores)] -> [NE16 tile 1]
 *   ...
 *
 * The key insight is that while NE16 is processing tile N, all 8 CPU cores
 * can convert tile N+1 in parallel. This eliminates CPU idle time.
 *
 * Requires double-buffered input scratch (ping/pong buffers).
 * --- */

/**
 * Worker function to convert a single tile of input from S8 to U8.
 * Called with pi_cl_team_fork to parallelize across all cores.
 */
typedef struct {
    const int8_t *in_s8;     /* Source: signed INT8 input */
    uint8_t *out_u8;         /* Dest: unsigned INT8 output */
    int tile_start;          /* First token index in this tile */
    int tile_len;            /* Number of tokens in this tile */
    int in_features;         /* Features per token */
    int input_zp;            /* Zero-point offset (typically 128) */
} ne16_tile_conv_args_t;

static void ne16_tile_convert_worker(void *args)
{
    const ne16_tile_conv_args_t *a = (const ne16_tile_conv_args_t *)args;
    const int core_id = pi_core_id();

    /* Compute total elements in this tile */
    const int total_elements = a->tile_len * a->in_features;

    /* Distribute work across cores */
    const int chunk = (total_elements + NUM_CORES - 1) / NUM_CORES;
    const int start = core_id * chunk;
    const int end = (start + chunk > total_elements) ? total_elements : (start + chunk);

    /* Source pointer accounts for tile offset */
    const int8_t *src = a->in_s8 + (size_t)a->tile_start * (size_t)a->in_features;

    for (int i = start; i < end; i++) {
        a->out_u8[i] = (uint8_t)((int)src[i] + a->input_zp);
    }
}

/**
 * Pipelined NE16 linear layer with double-buffered input conversion.
 *
 * This function overlaps NE16 compute with CPU input conversion by using
 * ping-pong buffers. While NE16 processes tile N, the CPU converts tile N+1.
 *
 * For hardware requantization mode (use_hw_requant=1):
 * - NE16 outputs INT8 directly using built-in requantization
 * - No output scratch buffer needed, no CPU postprocessing
 * - Requires scale/scale_shift/bias_corr parameters
 *
 * For software requantization mode (use_hw_requant=0):
 * - NE16 outputs INT32 accumulators
 * - CPU postprocessing applies bias and requantization
 * - Uses output_s32_scratch buffer
 *
 * @param input             Signed INT8 input [num_tokens, in_features]
 * @param weights_packed    Pre-packed NE16 weights
 * @param bias_corr         Bias correction [out_features] (for SW mode: includes input_zp adjustment)
 * @param scale             Per-channel scale [out_features] (for HW mode, else NULL)
 * @param scale_shift       Per-channel shift [out_features] (for HW mode, else NULL)
 * @param output            Signed INT8 output [num_tokens, out_stride]
 * @param num_tokens        Total number of tokens
 * @param in_features       Input feature dimension
 * @param out_features      Output feature dimension
 * @param out_stride        Stride between rows in output
 * @param scale_input       Input quantization scale (for SW mode)
 * @param scale_weight      Weight quantization scale (for SW mode)
 * @param scale_output      Output quantization scale (for SW mode)
 * @param tile_tokens       Number of tokens per tile
 * @param input_u8_ping     Ping buffer for U8 input [tile_tokens * in_features]
 * @param input_u8_pong     Pong buffer for U8 input [tile_tokens * in_features]
 * @param output_s32_scratch INT32 accumulator scratch [tile_tokens * out_features] (for SW mode)
 * @param use_hw_requant    1 to use NE16 hardware requantization, 0 for SW
 */
void ne16_linear_int8_pipelined(
    const int8_t *input,
    const int8_t *weights_packed,
    const int32_t *bias_corr,
    const uint8_t *scale,
    const uint8_t *scale_shift,
    int8_t *output,
    int num_tokens,
    int in_features,
    int out_features,
    int out_stride,
    float scale_input,
    float scale_weight,
    float scale_output,
    int tile_tokens,
    uint8_t *input_u8_ping,
    uint8_t *input_u8_pong,
    int32_t *output_s32_scratch,
    int use_hw_requant
) {
    if (!input || !weights_packed || !bias_corr || !output ||
        !input_u8_ping || !input_u8_pong) {
        return;
    }

    /* SW mode requires output scratch */
    if (!use_hw_requant && !output_s32_scratch) {
        return;
    }

    const int input_zp = NE16_INPUT_ZP;
    const int weight_offset = NE16_WEIGHT_OFFSET;
    const float combined_scale = scale_input * scale_weight / scale_output;

    /* Calculate number of tiles */
    const int num_tiles = (num_tokens + tile_tokens - 1) / tile_tokens;

    if (num_tiles == 0) return;

    /* Initialize active/next buffer pointers */
    uint8_t *active_buf = input_u8_ping;
    uint8_t *next_buf = input_u8_pong;

    /* ====== PROLOGUE: Convert and submit tile 0 ====== */
    int tile0_len = (tile_tokens > num_tokens) ? num_tokens : tile_tokens;

    ne16_tile_conv_args_t conv0 = {
        .in_s8 = input,
        .out_u8 = active_buf,
        .tile_start = 0,
        .tile_len = tile0_len,
        .in_features = in_features,
        .input_zp = input_zp,
    };
    pi_cl_team_fork(NUM_CORES, ne16_tile_convert_worker, &conv0);

    /* Submit tile 0 to NE16 */
    int job_id;
    if (use_hw_requant) {
        /* HW requant: NE16 outputs INT8 directly */
        job_id = ne16_conv1x1_submit_async(
            active_buf,
            (const uint8_t *)weights_packed,
            bias_corr,
            scale,
            scale_shift,
            output,
            /*in_w=*/tile0_len,
            /*in_h=*/1,
            in_features,
            out_features,
            (int8_t)weight_offset
        );
    } else {
        /* SW requant: NE16 outputs INT32 */
        job_id = ne16_conv1x1_s32_submit_async(
            active_buf,
            (const uint8_t *)weights_packed,
            output_s32_scratch,
            /*in_w=*/tile0_len,
            /*in_h=*/1,
            in_features,
            out_features,
            (int8_t)weight_offset
        );
    }

    /* ====== STEADY STATE: Pipeline tiles ====== */
    for (int t = 1; t < num_tiles; t++) {
        /* Calculate tile bounds */
        const int tile_start = t * tile_tokens;
        int tile_len = tile_tokens;
        if (tile_start + tile_len > num_tokens) {
            tile_len = num_tokens - tile_start;
        }

        /* PARALLEL: Convert next tile while NE16 runs */
        ne16_tile_conv_args_t conv_next = {
            .in_s8 = input,
            .out_u8 = next_buf,
            .tile_start = tile_start,
            .tile_len = tile_len,
            .in_features = in_features,
            .input_zp = input_zp,
        };
        pi_cl_team_fork(NUM_CORES, ne16_tile_convert_worker, &conv_next);

        /* Wait for previous NE16 job */
        ne16_wait_job(job_id);

        /* For SW requant mode, apply postprocessing for previous tile */
        if (!use_hw_requant) {
            const int prev_tile_start = (t - 1) * tile_tokens;
            int prev_tile_len = tile_tokens;
            if (prev_tile_start + prev_tile_len > num_tokens) {
                prev_tile_len = num_tokens - prev_tile_start;
            }

            ne16_post_args_t post = {
                .acc_s32 = output_s32_scratch,
                .bias_corr = bias_corr,
                .out_s8 = output + (size_t)prev_tile_start * (size_t)out_stride,
                .out_stride = out_stride,
                .num_tokens = prev_tile_len,
                .out_features = out_features,
                .combined_scale = combined_scale,
            };
            pi_cl_team_fork(NUM_CORES, ne16_postprocess_s32_to_s8_worker, &post);
        }

        /* Swap ping/pong buffers */
        uint8_t *tmp = active_buf;
        active_buf = next_buf;
        next_buf = tmp;

        /* Submit next NE16 job */
        int8_t *tile_output = use_hw_requant ? (output + (size_t)tile_start * (size_t)out_stride) : NULL;
        if (use_hw_requant) {
            job_id = ne16_conv1x1_submit_async(
                active_buf,
                (const uint8_t *)weights_packed,
                bias_corr,
                scale,
                scale_shift,
                tile_output,
                /*in_w=*/tile_len,
                /*in_h=*/1,
                in_features,
                out_features,
                (int8_t)weight_offset
            );
        } else {
            job_id = ne16_conv1x1_s32_submit_async(
                active_buf,
                (const uint8_t *)weights_packed,
                output_s32_scratch,
                /*in_w=*/tile_len,
                /*in_h=*/1,
                in_features,
                out_features,
                (int8_t)weight_offset
            );
        }
    }

    /* ====== EPILOGUE: Wait for last tile and postprocess ====== */
    ne16_wait_job(job_id);

    if (!use_hw_requant) {
        /* Postprocess final tile */
        const int final_tile_start = (num_tiles - 1) * tile_tokens;
        int final_tile_len = tile_tokens;
        if (final_tile_start + final_tile_len > num_tokens) {
            final_tile_len = num_tokens - final_tile_start;
        }

        ne16_post_args_t post = {
            .acc_s32 = output_s32_scratch,
            .bias_corr = bias_corr,
            .out_s8 = output + (size_t)final_tile_start * (size_t)out_stride,
            .out_stride = out_stride,
            .num_tokens = final_tile_len,
            .out_features = out_features,
            .combined_scale = combined_scale,
        };
        pi_cl_team_fork(NUM_CORES, ne16_postprocess_s32_to_s8_worker, &post);
    }
}

/**
 * Calculate scratch size for pipelined execution.
 *
 * Pipelined execution requires double-buffered input scratch (ping + pong).
 * For SW requantization mode, also needs output INT32 scratch.
 *
 * @param tile_tokens   Tokens per tile
 * @param in_features   Input features
 * @param out_features  Output features
 * @param use_hw_requant 1 if using HW requantization (no output scratch needed)
 * @return Total scratch size in bytes
 */
size_t ne16_linear_pipelined_scratch_size(int tile_tokens, int in_features, int out_features, int use_hw_requant)
{
    /* Double-buffered input: 2 * tile_tokens * in_features */
    size_t input_size = 2 * (size_t)tile_tokens * (size_t)in_features;

    /* Output scratch only needed for SW requantization */
    size_t output_size = 0;
    if (!use_hw_requant) {
        output_size = (size_t)tile_tokens * (size_t)out_features * sizeof(int32_t);
    }

    return input_size + output_size;
}

/* --- Scratch Buffer Size Calculation --- */

size_t ne16_linear_input_scratch_size(int tile_tokens, int in_features)
{
    /* Unsigned INT8 input tile */
    return (size_t)tile_tokens * (size_t)in_features;
}

size_t ne16_linear_output_scratch_size(int tile_tokens, int out_features)
{
    /* INT32 accumulator tile */
    return (size_t)tile_tokens * (size_t)out_features * sizeof(int32_t);
}

size_t ne16_linear_total_scratch_size(int tile_tokens, int in_features, int out_features)
{
    return ne16_linear_input_scratch_size(tile_tokens, in_features) +
           ne16_linear_output_scratch_size(tile_tokens, out_features);
}

/* --- NE16 Selftest (adapted from tinymyo) --- */

/**
 * Simple selftest to verify NE16 produces correct results.
 * Returns 0 on success, -1 on failure.
 *
 * @param l1_scratch    L1 buffer for scratch (input_u8 + acc_s32). If NULL, will allocate.
 * @param l1_size       Size of l1_scratch buffer in bytes
 */
int ne16_linear_selftest(void *l1_scratch, size_t l1_size)
{
    /* Test with EXACT fc1 dimensions: 49 tokens, 192 inputs, 64 outputs */
    const int num_tokens = 49;
    const int in_features = 192; /* 12 Ki subtiles - same as fc1 */
    const int out_features = 64; /* 2 Ko subtiles - same as fc1 */
    const int tile_tokens = 49;

#ifdef MINIMAL_OUTPUT
#define SELFTEST_LOG(...) ((void)0)
#else
#define SELFTEST_LOG(...) printf(__VA_ARGS__)
#endif

    SELFTEST_LOG("CL: === NE16 SELFTEST ===\n");
    SELFTEST_LOG("CL: Testing with tokens=%d in=%d out=%d tile=%d\n",
           num_tokens, in_features, out_features, tile_tokens);

    /* Allocate buffers */
    const size_t input_bytes = (size_t)num_tokens * (size_t)in_features * sizeof(int8_t);
    const size_t weight_bytes = (size_t)out_features * (size_t)in_features * sizeof(int8_t);
    const size_t bias_bytes = (size_t)out_features * sizeof(int32_t);
    const size_t out_bytes = (size_t)num_tokens * (size_t)out_features * sizeof(int8_t);

    int8_t *input = (int8_t *)pi_l2_malloc(input_bytes);
    int8_t *weights = (int8_t *)pi_l2_malloc(weight_bytes);
    /* CRITICAL: NE16 only reads weights correctly from L1 on gvsoc, NOT L2! */
    int8_t *weights_ne16 = (int8_t *)pi_cl_l1_malloc(NULL, weight_bytes);  /* Must be L1 for NE16 */
    int32_t *bias_corr = (int32_t *)pi_cl_l1_malloc(NULL, bias_bytes);     /* Also L1 for NE16 */
    int8_t *out_sw = (int8_t *)pi_l2_malloc(out_bytes);
    int8_t *out_ne16 = (int8_t *)pi_l2_malloc(out_bytes);

    /* CRITICAL: Use L1 for scratch buffers - NE16 only writes to L1, not L2! */
    /* Calculate required scratch sizes with 64-byte alignment */
    size_t input_u8_size = (size_t)tile_tokens * (size_t)in_features;
    size_t acc_s32_size = (size_t)tile_tokens * (size_t)out_features * sizeof(int32_t);
    input_u8_size = (input_u8_size + 63) & ~63;  /* Round up to 64-byte alignment */
    acc_s32_size = (acc_s32_size + 63) & ~63;
    const size_t total_scratch_needed = input_u8_size + acc_s32_size;

    uint8_t *input_u8;
    int32_t *acc_s32;
    int selftest_allocated_l1 = 0;

    if (l1_scratch && l1_size >= total_scratch_needed) {
        /* Use provided L1 buffer (same as fc1 uses) */
        input_u8 = (uint8_t *)l1_scratch;
        acc_s32 = (int32_t *)((uint8_t *)l1_scratch + input_u8_size);
        SELFTEST_LOG("CL: SELFTEST using provided L1 scratch at %p\n", l1_scratch);
    } else {
        /* Allocate our own L1 buffers */
        input_u8 = (uint8_t *)pi_cl_l1_malloc(NULL, input_u8_size);
        acc_s32 = (int32_t *)pi_cl_l1_malloc(NULL, acc_s32_size);
        selftest_allocated_l1 = 1;
        SELFTEST_LOG("CL: SELFTEST allocated own L1 scratch\n");
    }

    if (!input || !weights || !weights_ne16 || !bias_corr || !out_sw || !out_ne16 || !input_u8 || !acc_s32) {
        printf("CL: SELFTEST: alloc failed\n");
        return -1;
    }

    /* Clear L1 scratch buffers before use */
    memset(input_u8, 0, input_u8_size);
    memset(acc_s32, 0xCD, acc_s32_size);  /* Pattern to detect if NE16 writes */
    asm volatile("fence" ::: "memory");

    SELFTEST_LOG("CL: SELFTEST buffers: input(L2)=%p weights_ne16(L1)=%p bias_corr(L1)=%p\n",
           (void*)input, (void*)weights_ne16, (void*)bias_corr);
    SELFTEST_LOG("CL: SELFTEST scratch (L1): acc_s32=%p (size %zu) input_u8=%p (size %zu)\n",
           (void*)acc_s32, acc_s32_size, (void*)input_u8, input_u8_size);

    /* Initialize with simple pattern for easier debugging */
    /* All input values are 1 (signed), so dot product = sum of weights */
    /* Each output channel has weights = ko (constant), so dot product = ko * in_features */
    memset(input, 1, input_bytes);  /* All 1s */

    for (int ko = 0; ko < out_features; ko++) {
        int8_t w = (int8_t)(ko - 32);  /* Weights from -32 to 31 for channels 0-63 */
        int32_t sum_w = 0;
        for (int ki = 0; ki < in_features; ki++) {
            weights[ko * in_features + ki] = w;
            sum_w += w;
        }
        /* bias_corr = -128 * sum(weights) */
        bias_corr[ko] = -128 * sum_w;
    }
    /* Debug: print expected SW accumulators */
    /* For input=1 and weights[ko]=ko-32, sw_acc[ko] = (ko-32) * 192 */
    SELFTEST_LOG("CL: SELFTEST expected: ch0 acc=%d, ch1 acc=%d, ch32 acc=%d\n",
           (0-32)*192, (1-32)*192, (32-32)*192);
    memset(out_sw, 0, out_bytes);
    memset(out_ne16, 0, out_bytes);
    memcpy(weights_ne16, weights, weight_bytes);  /* Copy for in-place packing */

    /* Compute SW reference */
    const float scale = 1.0f / 65536.0f;  /* Very small scale to avoid saturation */
    SELFTEST_LOG("CL: SELFTEST SW raw accumulators (token 0, ch 0-7): ");
    for (int t = 0; t < num_tokens; t++) {
        for (int ko = 0; ko < out_features; ko++) {
            int32_t acc = 0;
            for (int ki = 0; ki < in_features; ki++) {
                acc += (int32_t)input[t * in_features + ki] * (int32_t)weights[ko * in_features + ki];
            }
            /* Debug: print raw accumulators for token 0 */
            if (t == 0 && ko < 8) {
                SELFTEST_LOG("%ld ", (long)acc);
            }
            /* Pure dot product - do NOT add bias_corr (that's only for NE16 compensation) */
            float val = (float)acc * scale;
            int32_t q = (int32_t)(val > 0 ? val + 0.5f : val - 0.5f);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_sw[t * out_features + ko] = (int8_t)q;
        }
    }
    SELFTEST_LOG("\n");

    /* Use IN-PLACE packing (same as fc1 uses in the executor) */
    /* This copies weights to weights_ne16 and packs in-place, computing bias_corr */
    memcpy(weights_ne16, weights, weight_bytes);
    asm volatile("fence" ::: "memory");

    /* Create a fake INT32 bias buffer of zeros for the in-place packing */
    int32_t *fake_bias = (int32_t *)pi_l2_malloc(bias_bytes);
    if (fake_bias) {
        memset(fake_bias, 0, bias_bytes);
        ne16_pack_weights_inplace_with_bias(weights_ne16, bias_corr, fake_bias, in_features, out_features);
        pi_l2_free(fake_bias, bias_bytes);
    } else {
        /* Fallback to runtime packing if allocation fails */
        ne16_pack_weights_runtime(weights, (uint8_t *)weights_ne16, in_features, out_features);
        for (int ko = 0; ko < out_features; ko++) {
            int32_t sum_w = 0;
            for (int ki = 0; ki < in_features; ki++) {
                sum_w += (int32_t)weights[ko * in_features + ki];
            }
            bias_corr[ko] = -128 * sum_w;
        }
    }
    asm volatile("fence" ::: "memory");

    /* Debug: verify packed weights (now in weights_ne16 buffer) */
    SELFTEST_LOG("CL: SELFTEST packed weights_ne16 at 0x%08X, first 16 bytes: ", (unsigned int)(uintptr_t)weights_ne16);
    for (int i = 0; i < 16; i++) {
        SELFTEST_LOG("%02x ", ((uint8_t*)weights_ne16)[i]);
    }
    SELFTEST_LOG("\n");
    SELFTEST_LOG("CL: SELFTEST packed row 1 (offset %d): ", in_features);
    for (int i = 0; i < 16; i++) {
        SELFTEST_LOG("%02x ", ((uint8_t*)weights_ne16)[in_features + i]);
    }
    SELFTEST_LOG("\n");

    /* Verify bias_corr was computed correctly */
    SELFTEST_LOG("CL: SELFTEST bias_corr[0..3] = %ld %ld %ld %ld\n",
           (long)bias_corr[0], (long)bias_corr[1], (long)bias_corr[2], (long)bias_corr[3]);

    /* Run NE16 */
    ne16_linear_int8_packed(
        input,
        (const int8_t *)weights_ne16,  /* Packed weights (in-place) */
        bias_corr,
        out_ne16,
        num_tokens,
        in_features,
        out_features,
        out_features,
        1.0f,  /* scale_input */
        1.0f,  /* scale_weight */
        65536.0f,  /* scale_output */
        tile_tokens,
        input_u8,
        acc_s32
    );

    /* Compare results */
    int errors = 0;
    for (int i = 0; i < num_tokens * out_features; i++) {
        if (out_sw[i] != out_ne16[i]) {
            if (errors < 8) {
                SELFTEST_LOG("CL: SELFTEST mismatch at %d: sw=%d ne16=%d\n", i, (int)out_sw[i], (int)out_ne16[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        SELFTEST_LOG("CL: SELFTEST PASS!\n");
    } else {
        SELFTEST_LOG("CL: SELFTEST FAIL: %d mismatches\n", errors);
    }

    /* Cleanup */
    pi_l2_free(input, input_bytes);
    pi_l2_free(weights, weight_bytes);
    /* weights_ne16 and bias_corr are in L1 */
    pi_cl_l1_free(NULL, weights_ne16, weight_bytes);
    pi_cl_l1_free(NULL, bias_corr, bias_bytes);
    pi_l2_free(out_sw, out_bytes);
    pi_l2_free(out_ne16, out_bytes);
    /* L1 scratch buffers - only free if we allocated them */
    if (selftest_allocated_l1) {
        pi_cl_l1_free(NULL, input_u8, input_u8_size);
        pi_cl_l1_free(NULL, acc_s32, acc_s32_size);
    }

    SELFTEST_LOG("CL: === END SELFTEST ===\n");
#undef SELFTEST_LOG
    return errors == 0 ? 0 : -1;
}

#endif /* ARES_USE_NE16 */
