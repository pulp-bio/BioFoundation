/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Elementwise Operations for ARES Runtime
 *
 * Add, Concat, Transpose operations.
 */

#include "ops/op_elementwise.h"
#include "network_kernels.h"
#include <math.h>
#include <string.h>
#include "pmsis.h"


#ifndef ADD_INT8_FIXEDPOINT_REQUANT
#define ADD_INT8_FIXEDPOINT_REQUANT 1
#endif

#ifndef ADD_INT8_REQUANT_SHIFT
#define ADD_INT8_REQUANT_SHIFT 24
#endif

void network_add_int8(
    const int8_t *input_a, const int8_t *input_b, int8_t *output,
    uint32_t size,
    float scale_a, float scale_b, float scale_out
) {
    int core_id = pi_core_id();
    int chunk = (size + CL_NUM_CORES - 1) / CL_NUM_CORES;
    int start = core_id * chunk;
    int end = (start + chunk > size) ? size : (start + chunk);

    const int use_identity = (scale_a == scale_b) && (scale_a == scale_out);

    if (use_identity) {
        for (int i = start; i < end; i++) {
            int32_t res = (int32_t)input_a[i] + (int32_t)input_b[i];
            if (res > 127) res = 127;
            if (res < -128) res = -128;
            output[i] = (int8_t)res;
        }
    } else {
#if ADD_INT8_FIXEDPOINT_REQUANT
        if (scale_out == 0.0f) {
            for (int i = start; i < end; i++) output[i] = 0;
            return;
        }
        const float inv_scale_out = 1.0f / scale_out;
        const int64_t den = 1LL << ADD_INT8_REQUANT_SHIFT;

        const int a_is_out = (scale_a == scale_out);
        const int b_is_out = (scale_b == scale_out);
        const int same_in = (scale_a == scale_b);

        if (same_in) {
            const float r = scale_a * inv_scale_out;
            const int32_t mul = qround(r * (float)(1 << ADD_INT8_REQUANT_SHIFT));
            for (int i = start; i < end; i++) {
                int32_t sum = (int32_t)input_a[i] + (int32_t)input_b[i];
                int32_t res = div_round_nearest_even_s64((int64_t)sum * (int64_t)mul, den);
                if (res > 127) res = 127;
                if (res < -128) res = -128;
                output[i] = (int8_t)res;
            }
        } else if (a_is_out) {
            const float r_b = scale_b * inv_scale_out;
            const int32_t mul_b = qround(r_b * (float)(1 << ADD_INT8_REQUANT_SHIFT));
            for (int i = start; i < end; i++) {
                int64_t acc = (int64_t)input_a[i] * den + (int64_t)input_b[i] * (int64_t)mul_b;
                int32_t res = div_round_nearest_even_s64(acc, den);
                if (res > 127) res = 127;
                if (res < -128) res = -128;
                output[i] = (int8_t)res;
            }
        } else if (b_is_out) {
            const float r_a = scale_a * inv_scale_out;
            const int32_t mul_a = qround(r_a * (float)(1 << ADD_INT8_REQUANT_SHIFT));
            for (int i = start; i < end; i++) {
                int64_t acc = (int64_t)input_a[i] * (int64_t)mul_a + (int64_t)input_b[i] * den;
                int32_t res = div_round_nearest_even_s64(acc, den);
                if (res > 127) res = 127;
                if (res < -128) res = -128;
                output[i] = (int8_t)res;
            }
        } else {
            const float r_a = scale_a * inv_scale_out;
            const float r_b = scale_b * inv_scale_out;
            const int32_t mul_a = qround(r_a * (float)(1 << ADD_INT8_REQUANT_SHIFT));
            const int32_t mul_b = qround(r_b * (float)(1 << ADD_INT8_REQUANT_SHIFT));
            for (int i = start; i < end; i++) {
                int64_t acc = (int64_t)input_a[i] * (int64_t)mul_a + (int64_t)input_b[i] * (int64_t)mul_b;
                int32_t res = div_round_nearest_even_s64(acc, den);
                if (res > 127) res = 127;
                if (res < -128) res = -128;
                output[i] = (int8_t)res;
            }
        }
#else
        for (int i = start; i < end; i++) {
            float val_a = (float)input_a[i] * scale_a;
            float val_b = (float)input_b[i] * scale_b;
            int32_t res = qround((val_a + val_b) / scale_out);
            if (res > 127) res = 127;
            if (res < -128) res = -128;
            output[i] = (int8_t)res;
        }
#endif
    }
}

void network_concat_int8(
    const int8_t **inputs, const float *input_scales, int8_t *output,
    uint16_t num_inputs, uint16_t batch,
    const uint16_t *channels_per_input,
    uint16_t height, uint16_t width,
    float scale_output
) {
    const int core_id = pi_core_id();
    const int spatial_size = height * width;

    uint16_t channel_offsets[num_inputs + 1];
    channel_offsets[0] = 0;
    for (int i = 0; i < num_inputs; i++) {
        channel_offsets[i + 1] = channel_offsets[i] + channels_per_input[i];
    }
    const int total_ch = channel_offsets[num_inputs];

    float scale_factors[num_inputs];
    uint8_t same_scale[num_inputs];
    for (int i = 0; i < num_inputs; i++) {
        const float scale_in = input_scales[i];
        if (fabsf(scale_in - scale_output) < 1e-5f) {
            same_scale[i] = 1;
            scale_factors[i] = 1.0f;
        } else {
            same_scale[i] = 0;
            scale_factors[i] = scale_in / scale_output;
        }
    }

    const int chunk = (total_ch + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start_ch = core_id * chunk;
    const int end_ch = (start_ch + chunk > total_ch) ? total_ch : (start_ch + chunk);

    for (int b = 0; b < batch; b++) {
        for (int ch = start_ch; ch < end_ch; ch++) {
            int input_idx = 0;
            while (ch >= channel_offsets[input_idx + 1]) {
                input_idx++;
            }
            const int local_ch = ch - channel_offsets[input_idx];
            const int8_t *src = inputs[input_idx] + (b * channels_per_input[input_idx] + local_ch) * spatial_size;
            int8_t *dst = output + (b * total_ch + ch) * spatial_size;

            if (same_scale[input_idx]) {
                memcpy(dst, src, (size_t)spatial_size);
            } else {
                const float scale = scale_factors[input_idx];
                for (int i = 0; i < spatial_size; i++) {
                    int32_t res = qround((float)src[i] * scale);
                    if (res > 127) res = 127;
                    if (res < -128) res = -128;
                    dst[i] = (int8_t)res;
                }
            }
        }
    }
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    int batch_size;
    int dim1;
    int dim2;
} transpose_2d_args_t;

static void transpose_2d_worker(void *arg) {
    transpose_2d_args_t *a = (transpose_2d_args_t *)arg;
    const int core_id = pi_core_id();
    const int total_rows = a->batch_size * a->dim1;
    const int chunk = (total_rows + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start = core_id * chunk;
    const int end = (start + chunk > total_rows) ? total_rows : (start + chunk);

    for (int row = start; row < end; row++) {
        const int b = row / a->dim1;
        const int i = row - b * a->dim1;
        const int8_t *in_row = a->input + (b * a->dim1 + i) * a->dim2;
        int8_t *out_base = a->output + (b * a->dim2) * a->dim1 + i;

        for (int j = 0; j < a->dim2; j++) {
            out_base[j * a->dim1] = in_row[j];
        }
    }
}

void network_transpose_2d_int8(
    const int8_t *input, int8_t *output,
    int batch_size, int dim1, int dim2
) {
    if (pi_core_id() != CL_ORCHESTRATOR_CORE_ID) return;

    transpose_2d_args_t args = {
        .input = input,
        .output = output,
        .batch_size = batch_size,
        .dim1 = dim1,
        .dim2 = dim2
    };
    pi_cl_team_fork(NUM_CORES, transpose_2d_worker, &args);
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    int channels;
    int in_h, in_w;
    int out_h, out_w;
    int pad_left, pad_right, pad_top, pad_bottom;
} zeropad2d_args_t;

static void zeropad2d_worker(void *arg) {
    zeropad2d_args_t *a = (zeropad2d_args_t *)arg;
    const int core_id = pi_core_id();

    // For few channels, parallelize across output rows instead
    if (a->channels < CL_NUM_CORES) {
        // Parallelize across rows for better utilization
        const int total_rows = a->channels * a->out_h;
        const int chunk = (total_rows + CL_NUM_CORES - 1) / CL_NUM_CORES;
        const int start_row = core_id * chunk;
        const int end_row = (start_row + chunk > total_rows) ? total_rows : (start_row + chunk);

        for (int row_idx = start_row; row_idx < end_row; row_idx++) {
            const int c = row_idx / a->out_h;
            const int out_y = row_idx % a->out_h;

            int8_t *out_row = a->output + c * a->out_h * a->out_w + out_y * a->out_w;

            // Check if this output row has corresponding input
            const int in_y = out_y - a->pad_top;
            if (in_y >= 0 && in_y < a->in_h) {
                // Row has data: zero left pad, copy data, zero right pad
                const int8_t *in_row = a->input + c * a->in_h * a->in_w + in_y * a->in_w;
                if (a->pad_left > 0) memset(out_row, 0, (size_t)a->pad_left);
                memcpy(out_row + a->pad_left, in_row, (size_t)a->in_w);
                if (a->pad_right > 0) memset(out_row + a->pad_left + a->in_w, 0, (size_t)a->pad_right);
            } else {
                // Row is all padding
                memset(out_row, 0, (size_t)a->out_w);
            }
        }
    } else {
        // Original path: parallelize across channels
        const int chunk = (a->channels + CL_NUM_CORES - 1) / CL_NUM_CORES;
        const int start_ch = core_id * chunk;
        const int end_ch = (start_ch + chunk > a->channels) ? a->channels : (start_ch + chunk);

        for (int c = start_ch; c < end_ch; c++) {
            const int8_t *in_plane = a->input + c * a->in_h * a->in_w;
            int8_t *out_plane = a->output + c * a->out_h * a->out_w;

            // Zero top padding rows
            if (a->pad_top > 0) {
                memset(out_plane, 0, (size_t)(a->pad_top * a->out_w));
            }

            // Process middle rows (with left/right padding)
            for (int y = 0; y < a->in_h; y++) {
                const int out_y = y + a->pad_top;
                int8_t *out_row = out_plane + out_y * a->out_w;

                if (a->pad_left > 0) memset(out_row, 0, (size_t)a->pad_left);
                memcpy(out_row + a->pad_left, in_plane + y * a->in_w, (size_t)a->in_w);
                if (a->pad_right > 0) memset(out_row + a->pad_left + a->in_w, 0, (size_t)a->pad_right);
            }

            // Zero bottom padding rows
            if (a->pad_bottom > 0) {
                memset(out_plane + (a->pad_top + a->in_h) * a->out_w, 0,
                       (size_t)(a->pad_bottom * a->out_w));
            }
        }
    }
}

void network_zeropad2d_int8(
    const int8_t *input, int8_t *output,
    int channels, int in_h, int in_w,
    int pad_left, int pad_right, int pad_top, int pad_bottom
) {
    if (pi_core_id() != CL_ORCHESTRATOR_CORE_ID) return;

    int out_h = in_h + pad_top + pad_bottom;
    int out_w = in_w + pad_left + pad_right;

    zeropad2d_args_t args = {
        .input = input,
        .output = output,
        .channels = channels,
        .in_h = in_h,
        .in_w = in_w,
        .out_h = out_h,
        .out_w = out_w,
        .pad_left = pad_left,
        .pad_right = pad_right,
        .pad_top = pad_top,
        .pad_bottom = pad_bottom
    };
    pi_cl_team_fork(NUM_CORES, zeropad2d_worker, &args);
}

// ---
// HWC Layout ZeroPad2D
// ---
// For HWC layout [H, W, C], each spatial position has channels contiguous.
// Padding adds zeros around the spatial dimensions while preserving channel layout.

typedef struct {
    const int8_t *input;
    int8_t *output;
    int channels;
    int in_h, in_w;
    int out_h, out_w;
    int pad_left, pad_right, pad_top, pad_bottom;
} zeropad2d_hwc_args_t;

static void zeropad2d_hwc_worker(void *arg) {
    zeropad2d_hwc_args_t *a = (zeropad2d_hwc_args_t *)arg;
    const int core_id = pi_core_id();

    // Parallelize across output rows
    const int chunk = (a->out_h + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start_row = core_id * chunk;
    const int end_row = (start_row + chunk > a->out_h) ? a->out_h : (start_row + chunk);

    const int row_bytes = a->out_w * a->channels;
    const int in_row_bytes = a->in_w * a->channels;
    const int left_pad_bytes = a->pad_left * a->channels;
    const int right_pad_bytes = a->pad_right * a->channels;

    for (int out_y = start_row; out_y < end_row; out_y++) {
        int8_t *out_row = a->output + out_y * row_bytes;
        const int in_y = out_y - a->pad_top;

        if (in_y >= 0 && in_y < a->in_h) {
            // This row has input data
            const int8_t *in_row = a->input + in_y * in_row_bytes;

            // Zero left padding
            if (left_pad_bytes > 0) {
                memset(out_row, 0, (size_t)left_pad_bytes);
            }

            // Copy input data (channels contiguous at each spatial position)
            memcpy(out_row + left_pad_bytes, in_row, (size_t)in_row_bytes);

            // Zero right padding
            if (right_pad_bytes > 0) {
                memset(out_row + left_pad_bytes + in_row_bytes, 0, (size_t)right_pad_bytes);
            }
        } else {
            // Entire row is padding (top or bottom)
            memset(out_row, 0, (size_t)row_bytes);
        }
    }
}

void network_zeropad2d_int8_hwc(
    const int8_t *input, int8_t *output,
    int channels, int in_h, int in_w,
    int pad_left, int pad_right, int pad_top, int pad_bottom
) {
    if (pi_core_id() != CL_ORCHESTRATOR_CORE_ID) return;

    int out_h = in_h + pad_top + pad_bottom;
    int out_w = in_w + pad_left + pad_right;

    zeropad2d_hwc_args_t args = {
        .input = input,
        .output = output,
        .channels = channels,
        .in_h = in_h,
        .in_w = in_w,
        .out_h = out_h,
        .out_w = out_w,
        .pad_left = pad_left,
        .pad_right = pad_right,
        .pad_top = pad_top,
        .pad_bottom = pad_bottom
    };
    pi_cl_team_fork(NUM_CORES, zeropad2d_hwc_worker, &args);
}

// ---
// Layout Conversion Functions: CHW <-> HWC
// ---
// CHW: [C, H, W] - channel-major, common for inference frameworks
// HWC: [H, W, C] - channel-last, efficient for SIMD with small channel counts

typedef struct {
    const int8_t *input;
    int8_t *output;
    int channels;
    int height;
    int width;
} layout_convert_args_t;

static void chw_to_hwc_worker(void *arg) {
    layout_convert_args_t *a = (layout_convert_args_t *)arg;
    const int core_id = pi_core_id();
    const int spatial = a->height * a->width;

    // Parallelize across spatial positions
    const int chunk = (spatial + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start = core_id * chunk;
    const int end = (start + chunk > spatial) ? spatial : start + chunk;

    for (int pos = start; pos < end; pos++) {
        const int h = pos / a->width;
        const int w = pos - h * a->width;

        int8_t *out_ptr = a->output + pos * a->channels;

        // Gather channels from scattered CHW positions to contiguous HWC
        for (int c = 0; c < a->channels; c++) {
            out_ptr[c] = a->input[c * spatial + pos];
        }
    }
}

void network_chw_to_hwc_int8(
    const int8_t *input, int8_t *output,
    int channels, int height, int width
) {
    if (pi_core_id() != CL_ORCHESTRATOR_CORE_ID) return;

    layout_convert_args_t args = {
        .input = input,
        .output = output,
        .channels = channels,
        .height = height,
        .width = width
    };
    pi_cl_team_fork(NUM_CORES, chw_to_hwc_worker, &args);
}

static void hwc_to_chw_worker(void *arg) {
    layout_convert_args_t *a = (layout_convert_args_t *)arg;
    const int core_id = pi_core_id();
    const int spatial = a->height * a->width;

    // Parallelize across channels
    const int chunk = (a->channels + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int start_ch = core_id * chunk;
    const int end_ch = (start_ch + chunk > a->channels) ? a->channels : start_ch + chunk;

    for (int c = start_ch; c < end_ch; c++) {
        int8_t *out_plane = a->output + c * spatial;

        // Gather spatial positions from strided HWC positions to contiguous CHW plane
        for (int pos = 0; pos < spatial; pos++) {
            out_plane[pos] = a->input[pos * a->channels + c];
        }
    }
}

void network_hwc_to_chw_int8(
    const int8_t *input, int8_t *output,
    int channels, int height, int width
) {
    if (pi_core_id() != CL_ORCHESTRATOR_CORE_ID) return;

    layout_convert_args_t args = {
        .input = input,
        .output = output,
        .channels = channels,
        .height = height,
        .width = width
    };
    pi_cl_team_fork(NUM_CORES, hwc_to_chw_worker, &args);
}
