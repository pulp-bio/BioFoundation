/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Normalization Operations for ARES Runtime
 *
 * LayerNorm and GroupNorm with integer-only variants for bit-exact matching.
 */

#include "ops/op_norm.h"
#include "network_kernels.h"
#include <math.h>
#include "pmsis.h"


/* Newton-Raphson sqrt approximation (matching Python sqrt_approx_python) */
static inline float sqrt_approx(float x) {
    if (x <= 0.0f) return 0.0f;
    if (x == 1.0f) return 1.0f;

    float guess;
    if (x < 0.01f) {
        guess = x;
    } else {
        union { float f; uint32_t i; } conv;
        conv.f = x;
        conv.i = (1 << 29) + (conv.i >> 1) - (1 << 22);
        guess = conv.f;
    }

    for (int iter = 0; iter < 12; iter++) {
        if (guess > 0.0f) {
            guess = 0.5f * (guess + x / guess);
        }
    }
    return guess;
}

/* Integer sqrt using binary search (bit-exact with Python) */
static inline int32_t sqrt_int32(int64_t number) {
    if (number <= 0) return 0;

    int32_t root = 0;
    int32_t start = 0;
    int32_t end = 46342;

    while (start <= end) {
        int32_t mid = (start + end) >> 1;
        int64_t mid_squared = (int64_t)mid * (int64_t)mid;

        if (mid_squared == number) {
            return mid;
        }

        if (mid_squared < number) {
            start = mid + 1;
            root = mid;
        } else {
            end = mid - 1;
        }
    }
    return root;
}

#ifdef ARES_LLAMA_SUPPORT
/* ---
 * RMSNorm Implementation
 * Simpler than LayerNorm - only computes root mean square, no mean centering.
 * --- */

void network_rmsnorm_int8_fp32(
    const int8_t *input, int8_t *output,
    const float *weight,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out, float eps
) {
    uint32_t num_vectors = total_elements / normalized_dim;

    for (uint32_t v = 0; v < num_vectors; v++) {
        const int8_t *input_vec = input + (v * normalized_dim);
        int8_t *output_vec = output + (v * normalized_dim);

        /* Compute sum of squares */
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < normalized_dim; i++) {
            float x = (float)input_vec[i] * scale_in;
            sum_sq += x * x;
        }

        /* Compute RMS */
        float mean_sq = sum_sq / (float)normalized_dim;
        float rms = sqrt_approx(mean_sq + eps);

        /* Normalize and apply weight */
        for (uint32_t i = 0; i < normalized_dim; i++) {
            float x = (float)input_vec[i] * scale_in;
            float normalized_val = x / rms;
            float output_fp32 = weight[i] * normalized_val;

            int32_t q = qround(output_fp32 / scale_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            output_vec[i] = (int8_t)q;
        }
    }
}

void network_rmsnorm_int8_integer(
    const int8_t *input, int8_t *output,
    const float *weight,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out
) {
    uint32_t num_vectors = total_elements / normalized_dim;

    for (uint32_t v = 0; v < num_vectors; v++) {
        const int8_t *input_vec = input + (v * normalized_dim);
        int8_t *output_vec = output + (v * normalized_dim);

        /* Compute sum of squares using SIMD if possible */
        int32_t sumsq_val = 0;
        if ((normalized_dim & 3) == 0) {
            const v4s *p = (const v4s *)input_vec;
            const uint32_t simd_count = normalized_dim >> 2;
            for (uint32_t i = 0; i < simd_count; i++) {
                const v4s x = p[i];
                sumsq_val = SumDotpSS(x, x, sumsq_val);
            }
        } else {
            for (uint32_t i = 0; i < normalized_dim; i++) {
                const int32_t x = (int32_t)input_vec[i];
                sumsq_val += x * x;
            }
        }

        /* Compute mean of squares and RMS */
        int64_t mean_sq = (int64_t)sumsq_val / (int64_t)normalized_dim;
        mean_sq += 1;  /* Integer epsilon */
        int32_t rms = sqrt_int32(mean_sq);
        if (rms <= 0) rms = 1;

        /* Normalize and apply weight.
         * scale_in cancels in normalization: (x*s) / (s*rms_int) = x/rms_int,
         * so we don't multiply by scale_in here. */
        for (uint32_t i = 0; i < normalized_dim; i++) {
            float x_normalized = (float)input_vec[i] / (float)rms;

            float output_fp32 = weight[i] * x_normalized;
            int32_t q = qround(output_fp32 / scale_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            output_vec[i] = (int8_t)q;
        }
    }
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    const float *weight;
    uint32_t num_vectors;
    uint32_t normalized_dim;
    float scale_in;
    float scale_out;
} rmsnorm_int8_integer_args_t;

static void rmsnorm_int8_integer_worker(void *arg) {
    rmsnorm_int8_integer_args_t *a = (rmsnorm_int8_integer_args_t *)arg;
    const int core_id = pi_core_id();

    for (uint32_t v = (uint32_t)core_id; v < a->num_vectors; v += (uint32_t)NUM_CORES) {
        const int8_t *input_vec = a->input + (v * a->normalized_dim);
        int8_t *output_vec = a->output + (v * a->normalized_dim);

        /* Compute sum of squares using SIMD if possible */
        int32_t sumsq_val = 0;
        if ((a->normalized_dim & 3) == 0) {
            const v4s *p = (const v4s *)input_vec;
            const uint32_t simd_count = a->normalized_dim >> 2;
            for (uint32_t i = 0; i < simd_count; i++) {
                const v4s x = p[i];
                sumsq_val = SumDotpSS(x, x, sumsq_val);
            }
        } else {
            for (uint32_t i = 0; i < a->normalized_dim; i++) {
                const int32_t x = (int32_t)input_vec[i];
                sumsq_val += x * x;
            }
        }

        /* Compute mean of squares and RMS */
        int64_t mean_sq = (int64_t)sumsq_val / (int64_t)a->normalized_dim;
        mean_sq += 1;  /* Integer epsilon */
        int32_t rms = sqrt_int32(mean_sq);
        if (rms <= 0) rms = 1;

        /* Normalize and apply weight.
         * scale_in cancels in normalization: (x*s) / (s*rms_int) = x/rms_int */
        for (uint32_t i = 0; i < a->normalized_dim; i++) {
            float x_normalized = (float)input_vec[i] / (float)rms;

            float output_fp32 = a->weight[i] * x_normalized;
            int32_t q = qround(output_fp32 / a->scale_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            output_vec[i] = (int8_t)q;
        }
    }
}

void network_rmsnorm_int8_integer_parallel(
    const int8_t *input, int8_t *output,
    const float *weight,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out
) {
    rmsnorm_int8_integer_args_t args = {
        .input = input,
        .output = output,
        .weight = weight,
        .num_vectors = total_elements / normalized_dim,
        .normalized_dim = normalized_dim,
        .scale_in = scale_in,
        .scale_out = scale_out,
    };
    pi_cl_team_fork(NUM_CORES, rmsnorm_int8_integer_worker, &args);
}
#endif // ARES_LLAMA_SUPPORT

/* --- LayerNorm Implementation --- */

void network_layernorm_int8_fixed_point(
    const int8_t *input, int8_t *output,
    const float *weight, const float *bias,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out, float eps
) {
    uint32_t num_vectors = total_elements / normalized_dim;

    for (uint32_t v = 0; v < num_vectors; v++) {
        const int8_t *input_vec = input + (v * normalized_dim);
        int8_t *output_vec = output + (v * normalized_dim);

        float sum_val = 0.0f;
        for (uint32_t i = 0; i < normalized_dim; i++) {
            sum_val += (float)input_vec[i] * scale_in;
        }
        float mean = sum_val / (float)normalized_dim;

        float var_sum = 0.0f;
        for (uint32_t i = 0; i < normalized_dim; i++) {
            float x = (float)input_vec[i] * scale_in;
            float diff = x - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / (float)normalized_dim;
        float std = sqrt_approx(variance + eps);

        for (uint32_t i = 0; i < normalized_dim; i++) {
            float x = (float)input_vec[i] * scale_in;
            float normalized_val = (x - mean) / std;
            float output_fp32 = weight[i] * normalized_val + bias[i];

            int32_t q = qround(output_fp32 / scale_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            output_vec[i] = (int8_t)q;
        }
    }
}

void network_layernorm_int8_integer(
    const int8_t *input, int8_t *output,
    const float *weight, const float *bias,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out
) {
    uint32_t num_vectors = total_elements / normalized_dim;

    for (uint32_t v = 0; v < num_vectors; v++) {
        const int8_t *input_vec = input + (v * normalized_dim);
        int8_t *output_vec = output + (v * normalized_dim);

        int32_t sum_val = 0;
        int32_t sumsq_val = 0;
        if ((normalized_dim & 3) == 0) {
            const v4s ones = (v4s){1, 1, 1, 1};
            const v4s *p = (const v4s *)input_vec;
            const uint32_t simd_count = normalized_dim >> 2;
            for (uint32_t i = 0; i < simd_count; i++) {
                const v4s x = p[i];
                sum_val = SumDotpSS(x, ones, sum_val);
                sumsq_val = SumDotpSS(x, x, sumsq_val);
            }
        } else {
            for (uint32_t i = 0; i < normalized_dim; i++) {
                const int32_t x = (int32_t)input_vec[i];
                sum_val += x;
                sumsq_val += x * x;
            }
        }

        const int64_t mean = (int64_t)sum_val / (int64_t)normalized_dim;
        const int64_t var_sum =
            (int64_t)sumsq_val -
            2 * mean * (int64_t)sum_val +
            (int64_t)normalized_dim * mean * mean;
        int64_t variance = var_sum / (int64_t)normalized_dim;
        variance += 1;
        int32_t std = sqrt_int32(variance);

        for (uint32_t i = 0; i < normalized_dim; i++) {
            int64_t x_centered = (int64_t)input_vec[i] - mean;
            float x_normalized = (float)x_centered / (float)std;
            x_normalized *= scale_in;

            float output_fp32 = weight[i] * x_normalized + bias[i];
            int32_t q = qround(output_fp32 / scale_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            output_vec[i] = (int8_t)q;
        }
    }
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    const float *weight;
    const float *bias;
    uint32_t num_vectors;
    uint32_t normalized_dim;
    float scale_in;
    float scale_out;
} layernorm_int8_integer_args_t;

static void layernorm_int8_integer_worker(void *arg) {
    layernorm_int8_integer_args_t *a = (layernorm_int8_integer_args_t *)arg;
    const int core_id = pi_core_id();

    for (uint32_t v = (uint32_t)core_id; v < a->num_vectors; v += (uint32_t)NUM_CORES) {
        const int8_t *input_vec = a->input + (v * a->normalized_dim);
        int8_t *output_vec = a->output + (v * a->normalized_dim);

        int32_t sum_val = 0;
        int32_t sumsq_val = 0;
        if ((a->normalized_dim & 3) == 0) {
            const v4s ones = (v4s){1, 1, 1, 1};
            const v4s *p = (const v4s *)input_vec;
            const uint32_t simd_count = a->normalized_dim >> 2;
            for (uint32_t i = 0; i < simd_count; i++) {
                const v4s x = p[i];
                sum_val = SumDotpSS(x, ones, sum_val);
                sumsq_val = SumDotpSS(x, x, sumsq_val);
            }
        } else {
            for (uint32_t i = 0; i < a->normalized_dim; i++) {
                const int32_t x = (int32_t)input_vec[i];
                sum_val += x;
                sumsq_val += x * x;
            }
        }

        const int64_t mean = (int64_t)sum_val / (int64_t)a->normalized_dim;
        const int64_t var_sum =
            (int64_t)sumsq_val -
            2 * mean * (int64_t)sum_val +
            (int64_t)a->normalized_dim * mean * mean;
        int64_t variance = var_sum / (int64_t)a->normalized_dim;
        variance += 1;
        int32_t std = sqrt_int32(variance);

        for (uint32_t i = 0; i < a->normalized_dim; i++) {
            int64_t x_centered = (int64_t)input_vec[i] - mean;
            float x_normalized = (float)x_centered / (float)std;
            x_normalized *= a->scale_in;

            float output_fp32 = a->weight[i] * x_normalized + a->bias[i];
            int32_t q = qround(output_fp32 / a->scale_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            output_vec[i] = (int8_t)q;
        }
    }
}

void network_layernorm_int8_integer_parallel(
    const int8_t *input, int8_t *output,
    const float *weight, const float *bias,
    uint32_t total_elements, uint32_t normalized_dim,
    float scale_in, float scale_out
) {
    layernorm_int8_integer_args_t args = {
        .input = input,
        .output = output,
        .weight = weight,
        .bias = bias,
        .num_vectors = total_elements / normalized_dim,
        .normalized_dim = normalized_dim,
        .scale_in = scale_in,
        .scale_out = scale_out,
    };
    pi_cl_team_fork(NUM_CORES, layernorm_int8_integer_worker, &args);
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    const float *weight;
    const float *bias;
    const int32_t *means;
    const int32_t *stds;
    uint32_t batch;
    uint32_t channels;
    uint32_t spatial_size;
    uint32_t num_groups;
    uint32_t channels_per_group;
    float scale_in;
    float scale_out;
} groupnorm_int8_args_t;

typedef struct {
    const int8_t *input;
    int32_t *means;
    int32_t *stds;
    uint32_t batch;
    uint32_t channels;
    uint32_t spatial_size;
    uint32_t num_groups;
    uint32_t channels_per_group;
} groupnorm_stats_args_t;

static void groupnorm_stats_worker(void *arg) {
    groupnorm_stats_args_t *a = (groupnorm_stats_args_t *)arg;
    const uint32_t core_id = (uint32_t)pi_core_id();
    const uint32_t num_stats = a->batch * a->num_groups;
    const uint32_t chunk = (num_stats + (uint32_t)CL_NUM_CORES - 1U) / (uint32_t)CL_NUM_CORES;
    const uint32_t start = core_id * chunk;
    uint32_t end = start + chunk;
    if (end > num_stats) end = num_stats;

    const uint32_t group_elems = a->channels_per_group * a->spatial_size;

    for (uint32_t sidx = start; sidx < end; sidx++) {
        const uint32_t b = sidx / a->num_groups;
        const uint32_t g = sidx - b * a->num_groups;

        const uint32_t c0 = g * a->channels_per_group;
        const uint32_t base = (b * a->channels + c0) * a->spatial_size;
        const int8_t *ptr = a->input + base;

        int32_t sum_val = 0;
        int32_t sumsq_val = 0;
        if ((group_elems & 3U) == 0U) {
            const v4s ones = (v4s){1, 1, 1, 1};
            const v4s *p4 = (const v4s *)ptr;
            const uint32_t n4 = group_elems >> 2;
            for (uint32_t i = 0; i < n4; i++) {
                const v4s x = p4[i];
                sum_val = SumDotpSS(x, ones, sum_val);
                sumsq_val = SumDotpSS(x, x, sumsq_val);
            }
        } else {
            for (uint32_t i = 0; i < group_elems; i++) {
                const int32_t x = (int32_t)ptr[i];
                sum_val += x;
                sumsq_val += x * x;
            }
        }

        const int64_t mean = (int64_t)sum_val / (int64_t)group_elems;
        const int64_t var_sum =
            (int64_t)sumsq_val -
            2 * mean * (int64_t)sum_val +
            (int64_t)group_elems * mean * mean;
        int64_t variance = var_sum / (int64_t)group_elems;
        variance += 1;
        int32_t std = sqrt_int32(variance);
        if (std <= 0) std = 1;

        a->means[sidx] = (int32_t)mean;
        a->stds[sidx] = std;
    }
}

static void groupnorm_int8_worker(void *arg) {
    groupnorm_int8_args_t *a = (groupnorm_int8_args_t *)arg;
    const uint32_t core_id = (uint32_t)pi_core_id();
    const uint32_t channels = a->channels;
    const uint32_t spatial_size = a->spatial_size;
    const uint32_t total_ch = a->batch * channels;
    const uint32_t chunk = (total_ch + (uint32_t)CL_NUM_CORES - 1U) / (uint32_t)CL_NUM_CORES;
    const uint32_t start_ch = core_id * chunk;
    uint32_t end_ch = start_ch + chunk;
    if (end_ch > total_ch) end_ch = total_ch;

    for (uint32_t idx_ch = start_ch; idx_ch < end_ch; idx_ch++) {
        const uint32_t b = idx_ch / channels;
        const uint32_t c = idx_ch - b * channels;

        const uint32_t g = c / a->channels_per_group;
        const uint32_t gs = b * a->num_groups + g;
        const int32_t mean = a->means[gs];
        int32_t std = a->stds[gs];
        if (std <= 0) std = 1;

        const float gamma = a->weight ? a->weight[c] : 1.0f;
        const float beta = a->bias ? a->bias[c] : 0.0f;

        const uint32_t base = idx_ch * spatial_size;
        const int8_t *in = a->input + base;
        int8_t *out = a->output + base;

        for (uint32_t s = 0; s < spatial_size; s++) {
            const int64_t x_centered = (int64_t)in[s] - (int64_t)mean;
            float x_norm = (float)x_centered / (float)std;
            x_norm *= a->scale_in;

            const float y_fp32 = gamma * x_norm + beta;
            int32_t q = qround(y_fp32 / a->scale_out);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out[s] = (int8_t)q;
        }
    }
}

void network_groupnorm_int8_integer_parallel(
    const int8_t *input,
    int8_t *output,
    const float *weight,
    const float *bias,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial_size,
    uint32_t num_groups,
    float scale_input,
    float scale_output
) {
    if (num_groups == 0 || spatial_size == 0 || channels == 0 || batch == 0) return;
    const uint32_t channels_per_group = channels / num_groups;
    if (channels_per_group == 0) return;

    const uint32_t num_stats = batch * num_groups;
    int32_t means[num_stats];
    int32_t stds[num_stats];

    groupnorm_stats_args_t stats_args = {
        .input = input,
        .means = means,
        .stds = stds,
        .batch = batch,
        .channels = channels,
        .spatial_size = spatial_size,
        .num_groups = num_groups,
        .channels_per_group = channels_per_group,
    };
    pi_cl_team_fork(NUM_CORES, groupnorm_stats_worker, &stats_args);

    groupnorm_int8_args_t args = {
        .input = input,
        .output = output,
        .weight = weight,
        .bias = bias,
        .means = means,
        .stds = stds,
        .batch = batch,
        .channels = channels,
        .spatial_size = spatial_size,
        .num_groups = num_groups,
        .channels_per_group = channels_per_group,
        .scale_in = scale_input,
        .scale_out = scale_output,
    };
    pi_cl_team_fork(NUM_CORES, groupnorm_int8_worker, &args);
}
