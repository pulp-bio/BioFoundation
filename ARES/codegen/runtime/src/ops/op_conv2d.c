/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Conv2D Tile Workers for ARES Runtime
 *
 * Tile workers for L2 fallback and pipelined Conv2D execution.
 */

#include "ops/op_conv2d.h"
#include "network_kernels.h"
#include <math.h>
#include "pmsis.h"


void conv2d_tile_worker(void *arg) {
    conv2d_tile_args_t *t = (conv2d_tile_args_t *)arg;

    // Check for depthwise convolution (groups == in_ch)
    // Depthwise requires HWC layout
    if (t->groups > 1 && t->groups == t->in_ch && t->layout == 1) {
        // Depthwise convolution: each output channel convolves with one input channel
        // For Ko-tiling: t->out_ch is tile size, t->total_out_ch is full channels
        uint16_t total_ch = t->total_out_ch > 0 ? t->total_out_ch : t->out_ch;
        network_conv2d_depthwise_int8(t->tile_input_l1, t->weights_l2, t->bias_l2, t->tile_output_l1,
            t->tile_in_h, t->tile_in_w, t->out_ch,  // channels = tile size for Ko-tiling
            t->tile_out_h, t->tile_out_w,
            t->kernel_h, t->kernel_w, t->stride_h, t->stride_w, t->pad_h, t->pad_w,
            t->scale_input, t->scale_weight, t->scale_output,
            total_ch, t->out_ch_offset);  // Ko-tiling parameters
        return;
    }

    // Dispatch based on layout (standard convolution)
    if (t->layout == 1) {  // LAYOUT_HWC
        // For Ko-tiling: use total_out_ch as stride, out_ch_offset as starting offset
        uint16_t out_ch_stride = t->total_out_ch > 0 ? t->total_out_ch : t->out_ch;
        uint16_t out_ch_offset = t->out_ch_offset;
        uint16_t weight_row_stride = t->weight_row_stride;
        if (weight_row_stride == 0) {
            weight_row_stride = (uint16_t)(t->in_ch * t->kernel_h * t->kernel_w);
        }
        network_conv2d_int8_hwc(t->tile_input_l1, t->weights_l2, t->bias_l2, t->tile_output_l1,
            t->tile_in_h, t->tile_in_w, t->in_ch, t->tile_out_h, t->tile_out_w, t->out_ch,
            t->kernel_h, t->kernel_w, weight_row_stride, t->stride_h, t->stride_w, t->pad_h, t->pad_w,
            t->scale_input, t->scale_weight, t->scale_output,
            out_ch_stride, out_ch_offset);
    } else {  // LAYOUT_CHW (default)
        uint16_t weight_row_stride = t->weight_row_stride;
        if (weight_row_stride == 0) {
            weight_row_stride = (uint16_t)(t->in_ch * t->kernel_h * t->kernel_w);
        }
        network_conv2d_int8(t->tile_input_l1, t->weights_l2, t->bias_l2, t->tile_output_l1,
            t->tile_in_h, t->tile_in_w, t->in_ch, t->tile_out_h, t->tile_out_w, t->out_ch,
            t->kernel_h, t->kernel_w, weight_row_stride, t->stride_h, t->stride_w, t->pad_h, t->pad_w,
            t->scale_input, t->scale_weight, t->scale_output, t->cluster_dev);
    }
}

/**
 * Conv2D tile worker with fusion (for look-ahead/look-behind pattern)
 * All cores compute Conv2D in parallel, then parallelize fusion across cores
 */
void conv2d_tile_worker_with_fusion(void *arg) {
    conv2d_tile_args_t *t = (conv2d_tile_args_t *)arg;

    // Check for depthwise convolution (groups == in_ch)
    // Depthwise requires HWC layout
    if (t->groups > 1 && t->groups == t->in_ch && t->layout == 1) {
        // Depthwise convolution: each output channel convolves with one input channel
        // For Ko-tiling: t->out_ch is tile size, t->total_out_ch is full channels
        uint16_t total_ch = t->total_out_ch > 0 ? t->total_out_ch : t->out_ch;
        network_conv2d_depthwise_int8(t->tile_input_l1, t->weights_l2, t->bias_l2, t->tile_output_l1,
            t->tile_in_h, t->tile_in_w, t->out_ch,  // channels = tile size for Ko-tiling
            t->tile_out_h, t->tile_out_w,
            t->kernel_h, t->kernel_w, t->stride_h, t->stride_w, t->pad_h, t->pad_w,
            t->scale_input, t->scale_weight, t->scale_output,
            total_ch, t->out_ch_offset);  // Ko-tiling parameters
        // Continue to fusion handling below
    }
    // Dispatch based on layout (standard convolution)
    else if (t->layout == 1) {  // LAYOUT_HWC
        // For Ko-tiling: use total_out_ch as stride, out_ch_offset as starting offset
        uint16_t out_ch_stride = t->total_out_ch > 0 ? t->total_out_ch : t->out_ch;
        uint16_t out_ch_offset = t->out_ch_offset;
        uint16_t weight_row_stride = t->weight_row_stride;
        if (weight_row_stride == 0) {
            weight_row_stride = (uint16_t)(t->in_ch * t->kernel_h * t->kernel_w);
        }
        network_conv2d_int8_hwc(t->tile_input_l1, t->weights_l2, t->bias_l2, t->tile_output_l1,
            t->tile_in_h, t->tile_in_w, t->in_ch, t->tile_out_h, t->tile_out_w, t->out_ch,
            t->kernel_h, t->kernel_w, weight_row_stride, t->stride_h, t->stride_w, t->pad_h, t->pad_w,
            t->scale_input, t->scale_weight, t->scale_output,
            out_ch_stride, out_ch_offset);
    } else {  // LAYOUT_CHW (default)
        uint16_t weight_row_stride = t->weight_row_stride;
        if (weight_row_stride == 0) {
            weight_row_stride = (uint16_t)(t->in_ch * t->kernel_h * t->kernel_w);
        }
        network_conv2d_int8(t->tile_input_l1, t->weights_l2, t->bias_l2, t->tile_output_l1,
            t->tile_in_h, t->tile_in_w, t->in_ch, t->tile_out_h, t->tile_out_w, t->out_ch,
            t->kernel_h, t->kernel_w, weight_row_stride, t->stride_h, t->stride_w, t->pad_h, t->pad_w,
            t->scale_input, t->scale_weight, t->scale_output, t->cluster_dev);
    }

    pi_cl_team_barrier();

    if (t->fusion_relu || t->fusion_quant) {
        int core_id = pi_core_id();

        // For HWC Ko-tiling: output is interleaved, iterate over owned channels per pixel
        // For CHW or non-tiled: output is contiguous
        const int is_hwc_kotiling = (t->layout == 1) && (t->total_out_ch > 0);

        if (is_hwc_kotiling) {
            // HWC Ko-tiling: channels are interleaved per pixel
            // Each pixel has total_out_ch channels, but we only wrote out_ch channels starting at out_ch_offset
            const int total_pixels = t->tile_out_h * t->tile_out_w;
            const int pixels_per_core = (total_pixels + NUM_CORES - 1) / NUM_CORES;
            const int pixel_start = core_id * pixels_per_core;
            const int pixel_end = (pixel_start + pixels_per_core > total_pixels) ? total_pixels : pixel_start + pixels_per_core;

            if (t->fusion_quant) {
                int8_t map_int8[256];
                const float rescale = t->quant_scale_in / t->quant_scale_out;
                for (int v = -128; v <= 127; v++) {
                    int32_t vv = v;
                    if (t->fusion_relu && vv < 0) vv = 0;
                    float val_fp32 = (float)vv * rescale;
                    int32_t val_int32 = (int32_t)roundf(val_fp32);
                    if (val_int32 > 127) val_int32 = 127;
                    if (val_int32 < -128) val_int32 = -128;
                    map_int8[(uint32_t)(v + 128)] = (int8_t)val_int32;
                }
                for (int pixel = pixel_start; pixel < pixel_end; pixel++) {
                    int8_t *pixel_base = t->tile_output_l1 + pixel * t->total_out_ch + t->out_ch_offset;
                    for (int c = 0; c < t->out_ch; c++) {
                        pixel_base[c] = map_int8[(uint32_t)((int)pixel_base[c] + 128)];
                    }
                }
            } else if (t->fusion_relu) {
                for (int pixel = pixel_start; pixel < pixel_end; pixel++) {
                    int8_t *pixel_base = t->tile_output_l1 + pixel * t->total_out_ch + t->out_ch_offset;
                    for (int c = 0; c < t->out_ch; c++) {
                        if (pixel_base[c] < 0) {
                            pixel_base[c] = 0;
                        }
                    }
                }
            }
        } else {
            // CHW or non-tiled HWC: output is contiguous
            size_t total_size = (size_t)t->tile_out_h * t->tile_out_w * t->out_ch;
            size_t chunk_size = total_size / NUM_CORES;
            size_t start = core_id * chunk_size;
            size_t end = (core_id == NUM_CORES - 1) ? total_size : start + chunk_size;

            if (t->fusion_quant) {
                int8_t map_int8[256];
                const float rescale = t->quant_scale_in / t->quant_scale_out;
                for (int v = -128; v <= 127; v++) {
                    int32_t vv = v;
                    if (t->fusion_relu && vv < 0) vv = 0;
                    float val_fp32 = (float)vv * rescale;
                    int32_t val_int32 = (int32_t)roundf(val_fp32);
                    if (val_int32 > 127) val_int32 = 127;
                    if (val_int32 < -128) val_int32 = -128;
                    map_int8[(uint32_t)(v + 128)] = (int8_t)val_int32;
                }
                for (size_t i = start; i < end; i++) {
                    t->tile_output_l1[i] = map_int8[(uint32_t)((int)t->tile_output_l1[i] + 128)];
                }
            } else if (t->fusion_relu) {
                for (size_t i = start; i < end; i++) {
                    if (t->tile_output_l1[i] < 0) {
                        t->tile_output_l1[i] = 0;
                    }
                }
            }
        }
    }
}
