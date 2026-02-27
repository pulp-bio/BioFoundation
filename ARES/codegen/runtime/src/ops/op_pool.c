/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Pooling Operations for ARES Runtime
 *
 * MaxPool, AvgPool, GlobalAvgPool, AdaptiveAvgPool1D.
 * All ops parallelize across the GAP9 cluster (8 cores).
 */

#include "ops/op_pool.h"
#include "network_kernels.h"
#include <math.h>
#include "pmsis.h"


/* SIMD-optimized contiguous INT8 sum (used by GlobalAvgPool and AdaptiveAvgPool) */
static inline int32_t sum_int8_contiguous(const int8_t *ptr, int count) {
    int32_t sum = 0;
    if (count <= 0) return 0;

    /* Check alignment for SIMD */
    if (((uintptr_t)ptr & 0x3) != 0) {
        for (int i = 0; i < count; i++) sum += (int32_t)ptr[i];
        return sum;
    }

    const v4s ones = (v4s){1, 1, 1, 1};
    const v4s *p = (const v4s *)ptr;
    const int simd_count = count >> 2;
    for (int i = 0; i < simd_count; i++) {
        sum = SumDotpSS(p[i], ones, sum);
    }

    const int8_t *tail = ptr + (simd_count << 2);
    const int remainder = count & 0x3;
    for (int i = 0; i < remainder; i++) sum += (int32_t)tail[i];
    return sum;
}

void network_maxpool_int8(
    const int8_t *input, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_h, uint16_t pad_w
) {
    int core_id = pi_core_id();
    int chunk = (in_ch + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start_ch = core_id * chunk;
    int end_ch = (start_ch + chunk > in_ch) ? in_ch : (start_ch + chunk);

    for (int c = start_ch; c < end_ch; c++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                int8_t max_val = -128;
                for (int ky = 0; ky < kernel_h; ky++) {
                    for (int kx = 0; kx < kernel_w; kx++) {
                        int in_y = y * stride_h + ky - pad_h;
                        int in_x = x * stride_w + kx - pad_w;
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            int8_t val = input[(c * in_h + in_y) * in_w + in_x];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                output[(c * out_h + y) * out_w + x] = max_val;
            }
        }
    }
}

void network_avgpool_int8(
    const int8_t *input, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t stride_h, uint16_t stride_w,
    float scale_in, float scale_out
) {
    int core_id = pi_core_id();
    int chunk = (in_ch + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start_ch = core_id * chunk;
    int end_ch = (start_ch + chunk > in_ch) ? in_ch : (start_ch + chunk);
    const int same_scale = (fabsf(scale_in - scale_out) < 1e-12f);
    const int shift = 24;
    const int expected_out_h = (in_h >= kernel_h && stride_h > 0) ? ((in_h - kernel_h) / stride_h + 1) : 0;
    const int expected_out_w = (in_w >= kernel_w && stride_w > 0) ? ((in_w - kernel_w) / stride_w + 1) : 0;
    const int count_constant = (expected_out_h == out_h && expected_out_w == out_w) ? (kernel_h * kernel_w) : 0;
    int32_t mul_const = 0;
    int32_t mul_base = 0;
    if (!same_scale) {
        const float scale_ratio = scale_in / scale_out;
        if (count_constant > 0) {
            mul_const = qround(scale_ratio / (float)count_constant * (float)(1 << shift));
        } else {
            mul_base = qround(scale_ratio * (float)(1 << shift));
        }
    }

    for (int c = start_ch; c < end_ch; c++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                int32_t sum = 0;
                int count = 0;
                for (int ky = 0; ky < kernel_h; ky++) {
                    for (int kx = 0; kx < kernel_w; kx++) {
                        int in_y = y * stride_h + ky;
                        int in_x = x * stride_w + kx;
                        if (in_y < in_h && in_x < in_w) {
                            sum += input[(c * in_h + in_y) * in_w + in_x];
                            count++;
                        }
                    }
                }
                int32_t val = 0;
                if (count > 0) {
                    if (same_scale) {
                        val = div_round_nearest_even_s64(sum, count);
                    } else if (count_constant > 0) {
                        val = mul_shift_round_nearest_even(sum, mul_const, shift);
                    } else {
                        int64_t denom = ((int64_t)count) << shift;
                        int64_t prod = (int64_t)sum * (int64_t)mul_base;
                        val = div_round_nearest_even_s64(prod, denom);
                    }
                }
                if (val > 127) val = 127;
                if (val < -128) val = -128;
                output[(c * out_h + y) * out_w + x] = (int8_t)val;
            }
        }
    }
}

void network_global_avgpool_int8(
    const int8_t *input, int8_t *output,
    uint16_t batch, uint16_t channels,
    uint16_t h, uint16_t w,
    float scale_in, float scale_out
) {
    int core_id = pi_core_id();
    int chunk = (channels + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start_ch = core_id * chunk;
    int end_ch = (start_ch + chunk > channels) ? channels : (start_ch + chunk);
    int total_pixels = h * w;
    const int same_scale = (fabsf(scale_in - scale_out) < 1e-12f);
    const int shift = 24;
    int32_t mul_const = 0;
    if (!same_scale && total_pixels > 0) {
        const float scale_ratio = scale_in / scale_out;
        mul_const = qround(scale_ratio / (float)total_pixels * (float)(1 << shift));
    }

    for (int c = start_ch; c < end_ch; c++) {
        int32_t sum = 0;
        const int8_t *plane = input + c * total_pixels;
        if (total_pixels > 0) {
            sum = sum_int8_contiguous(plane, total_pixels);
        }

        int32_t val = 0;
        if (total_pixels > 0) {
            if (same_scale) {
                /* Match atomic_ops/globalavgpool.py rounding */
                val = (sum >= 0)
                    ? (sum + (total_pixels >> 1)) / total_pixels
                    : (sum - (total_pixels >> 1)) / total_pixels;
            } else {
                const int64_t prod = (int64_t)sum * (int64_t)mul_const;
                const int64_t half = (int64_t)1 << (shift - 1);
                if (prod >= 0) {
                    val = (int32_t)((prod + half) >> shift);
                } else {
                    val = (int32_t)(-(((-prod) + half) >> shift));
                }
            }
        }
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[c] = (int8_t)val;
    }
}

/**
 * Global average pooling for HWC layout
 * Input: [H, W, C] -> Output: [1, 1, C] (just [C] values)
 */
void network_global_avgpool_int8_hwc(
    const int8_t *input, int8_t *output,
    uint16_t channels, uint16_t h, uint16_t w,
    float scale_in, float scale_out
) {
    const int core_id = pi_core_id();
    const int chunk = (channels + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start_ch = core_id * chunk;
    const int end_ch = (start_ch + chunk > channels) ? channels : (start_ch + chunk);
    const int total_pixels = h * w;
    const int same_scale = (fabsf(scale_in - scale_out) < 1e-12f);
    const int shift = 24;
    int32_t mul_const = 0;
    if (!same_scale && total_pixels > 0) {
        const float scale_ratio = scale_in / scale_out;
        mul_const = qround(scale_ratio / (float)total_pixels * (float)(1 << shift));
    }

    for (int c = start_ch; c < end_ch; c++) {
        int32_t sum = 0;

        // HWC layout: channels are interleaved at each spatial position
        // We need to iterate over all H*W positions and accumulate channel c
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                sum += (int32_t)input[(y * w + x) * channels + c];
            }
        }

        int32_t val = 0;
        if (total_pixels > 0) {
            if (same_scale) {
                val = (sum >= 0)
                    ? (sum + (total_pixels >> 1)) / total_pixels
                    : (sum - (total_pixels >> 1)) / total_pixels;
            } else {
                const int64_t prod = (int64_t)sum * (int64_t)mul_const;
                const int64_t half = (int64_t)1 << (shift - 1);
                if (prod >= 0) {
                    val = (int32_t)((prod + half) >> shift);
                } else {
                    val = (int32_t)(-(((-prod) + half) >> shift));
                }
            }
        }
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output[c] = (int8_t)val;
    }
}

// ---
// HWC Layout Pool Operations
// ---
// For networks using Height-Width-Channel layout, channels are contiguous
// at each spatial position enabling efficient memory access patterns.

void network_maxpool_int8_hwc(
    const int8_t *input, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t channels,
    uint16_t out_h, uint16_t out_w,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t stride_h, uint16_t stride_w,
    uint16_t pad_h, uint16_t pad_w
) {
    const int core_id = pi_core_id();
    const int total_out_pos = out_h * out_w;
    const int chunk = (total_out_pos + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start_pos = core_id * chunk;
    const int end_pos = (start_pos + chunk > total_out_pos) ? total_out_pos : start_pos + chunk;

    for (int pos = start_pos; pos < end_pos; pos++) {
        const int out_y = pos / out_w;
        const int out_x = pos - out_y * out_w;

        // HWC output pointer: all channels at (out_y, out_x) are contiguous
        int8_t *out_ptr = output + (out_y * out_w + out_x) * channels;

        // Initialize max values for all channels
        for (int c = 0; c < channels; c++) {
            out_ptr[c] = -128;
        }

        // Find max over the pooling window (non-square kernel support)
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                const int in_y = out_y * stride_h + ky - pad_h;
                const int in_x = out_x * stride_w + kx - pad_w;

                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    // HWC input: all channels at (in_y, in_x) are contiguous
                    const int8_t *in_ptr = input + (in_y * in_w + in_x) * channels;

                    for (int c = 0; c < channels; c++) {
                        if (in_ptr[c] > out_ptr[c]) {
                            out_ptr[c] = in_ptr[c];
                        }
                    }
                }
            }
        }
    }
}

void network_avgpool_int8_hwc(
    const int8_t *input, int8_t *output,
    uint16_t in_h, uint16_t in_w, uint16_t channels,
    uint16_t out_h, uint16_t out_w,
    uint16_t kernel_h, uint16_t kernel_w,
    uint16_t stride_h, uint16_t stride_w,
    float scale_in, float scale_out
) {
    const int core_id = pi_core_id();
    const int total_out_pos = out_h * out_w;
    const int chunk = (total_out_pos + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start_pos = core_id * chunk;
    const int end_pos = (start_pos + chunk > total_out_pos) ? total_out_pos : start_pos + chunk;

    const int same_scale = (fabsf(scale_in - scale_out) < 1e-12f);
    const int shift = 24;
    const int expected_out_h = (in_h >= kernel_h && stride_h > 0) ? ((in_h - kernel_h) / stride_h + 1) : 0;
    const int expected_out_w = (in_w >= kernel_w && stride_w > 0) ? ((in_w - kernel_w) / stride_w + 1) : 0;
    const int count_constant = (expected_out_h == out_h && expected_out_w == out_w) ? (kernel_h * kernel_w) : 0;

    int32_t mul_const = 0;
    int32_t mul_base = 0;
    if (!same_scale) {
        const float scale_ratio = scale_in / scale_out;
        if (count_constant > 0) {
            mul_const = qround(scale_ratio / (float)count_constant * (float)(1 << shift));
        } else {
            mul_base = qround(scale_ratio * (float)(1 << shift));
        }
    }

    // Stack sum buffer for channel accumulation (small for typical networks).
    int32_t sums[64];  // Support up to 64 channels on stack

    for (int pos = start_pos; pos < end_pos; pos++) {
        const int out_y = pos / out_w;
        const int out_x = pos - out_y * out_w;

        int8_t *out_ptr = output + (out_y * out_w + out_x) * channels;

        // Initialize sums
        for (int c = 0; c < channels; c++) {
            sums[c] = 0;
        }

        int count = 0;
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                const int in_y = out_y * stride_h + ky;
                const int in_x = out_x * stride_w + kx;

                if (in_y < in_h && in_x < in_w) {
                    const int8_t *in_ptr = input + (in_y * in_w + in_x) * channels;
                    for (int c = 0; c < channels; c++) {
                        sums[c] += (int32_t)in_ptr[c];
                    }
                    count++;
                }
            }
        }

        // Compute averages for all channels
        for (int c = 0; c < channels; c++) {
            int32_t val = 0;
            if (count > 0) {
                if (same_scale) {
                    val = div_round_nearest_even_s64(sums[c], count);
                } else if (count_constant > 0) {
                    val = mul_shift_round_nearest_even(sums[c], mul_const, shift);
                } else {
                    int64_t denom = ((int64_t)count) << shift;
                    int64_t prod = (int64_t)sums[c] * (int64_t)mul_base;
                    val = div_round_nearest_even_s64(prod, denom);
                }
            }
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            out_ptr[c] = (int8_t)val;
        }
    }
}

void network_adaptive_avgpool1d_int8(
    const int8_t *input, int8_t *output,
    uint16_t batch, uint16_t channels,
    uint16_t input_len, uint16_t output_size,
    uint16_t input_stride_ch, uint16_t input_stride_len,
    uint32_t input_batch_stride
) {
    const int core_id = pi_core_id();
    const int total = (int)batch * (int)channels;
    const int chunk = (total + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start = core_id * chunk;
    const int end = (start + chunk > total) ? total : (start + chunk);
    const uint32_t output_batch_stride = (uint32_t)channels * (uint32_t)output_size;

    for (int idx = start; idx < end; idx++) {
        const int b = idx / channels;
        const int c = idx - b * channels;
        const int8_t *base = input + (uint32_t)b * input_batch_stride + (uint32_t)c * input_stride_ch;
        int8_t *out_base = output + (uint32_t)b * output_batch_stride + (uint32_t)c * output_size;

        if (output_size == 1) {
            int32_t sum = 0;
            if (input_len > 0) {
                if (input_stride_len == 1) {
                    sum = sum_int8_contiguous(base, input_len);
                } else {
                    for (int i = 0; i < input_len; i++) {
                        sum += (int32_t)base[i * input_stride_len];
                    }
                }
                int32_t avg = (sum + (input_len >> 1)) / input_len;
                if (avg > 127) avg = 127;
                if (avg < -128) avg = -128;
                out_base[0] = (int8_t)avg;
            } else {
                out_base[0] = 0;
            }
            continue;
        }

        for (int o = 0; o < output_size; o++) {
            const int in_start = (o * input_len) / output_size;
            const int in_end = ((o + 1) * input_len) / output_size;
            const int count = in_end - in_start;
            int32_t sum = 0;
            if (count > 0) {
                const int8_t *ptr = base + in_start * input_stride_len;
                if (input_stride_len == 1) {
                    sum = sum_int8_contiguous(ptr, count);
                } else {
                    for (int i = 0; i < count; i++) {
                        sum += (int32_t)ptr[i * input_stride_len];
                    }
                }
                int32_t avg = (sum + (count >> 1)) / count;
                if (avg > 127) avg = 127;
                if (avg < -128) avg = -128;
                out_base[o] = (int8_t)avg;
            } else {
                out_base[o] = 0;
            }
        }
    }
}

/**
 * Mean pooling over sequence dimension: INT8 [B, seq_len, features] -> INT8 [B, features]
 *
 * Used for transformer classification heads (mean over sequence tokens).
 * Each core processes a subset of output features.
 *
 * The operation computes: output[b, f] = mean(input[b, :, f]) for each feature f
 * With scale conversion: output_int8 = round((sum_int32 / seq_len) * (scale_in / scale_out))
 */
void network_mean_pool_int8(
    const int8_t *input,
    int8_t *output,
    uint32_t batch,
    uint32_t seq_len,
    uint32_t features,
    float scale_input,
    float scale_output
) {
    int core_id = pi_core_id();

    // Parallelize across features (each core handles a subset of features)
    int chunk = (features + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start_f = core_id * chunk;
    int end_f = (start_f + chunk > features) ? features : (start_f + chunk);

    // Precompute scale factor
    const float scale_ratio = scale_input / scale_output;
    const int same_scale = (fabsf(scale_input - scale_output) < 1e-12f);

    // Fixed-point scale computation for efficiency
    const int shift = 24;
    int32_t mul = 0;
    if (!same_scale) {
        mul = qround(scale_ratio / (float)seq_len * (float)(1 << shift));
    }

    for (uint32_t b = 0; b < batch; b++) {
        const int8_t *batch_in = input + b * seq_len * features;
        int8_t *batch_out = output + b * features;

        for (int f = start_f; f < end_f; f++) {
            // Sum over sequence dimension
            int32_t sum = 0;
            for (uint32_t s = 0; s < seq_len; s++) {
                sum += (int32_t)batch_in[s * features + f];
            }

            // Compute mean with scale conversion
            int32_t result;
            if (same_scale) {
                // Simple integer division
                result = (sum + ((int32_t)seq_len >> 1)) / (int32_t)seq_len;
            } else {
                // Fixed-point multiply-shift
                result = mul_shift_round_nearest_even(sum, mul, shift);
            }

            // Clip to INT8 range
            if (result > 127) result = 127;
            if (result < -128) result = -128;
            batch_out[f] = (int8_t)result;
        }
    }
}
