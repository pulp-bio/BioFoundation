/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * simple_executor.c - Data-driven executor for simple ops
 *
 * This executor handles ops with uniform interfaces via descriptor tables,
 * keeping generated code compact and readable. Complex ops (MHSA, Mamba, etc.)
 * remain as explicit generated code.
 *
 * Supported ops:
 *   - Activations: ReLU, GELU, SiLU, Softmax
 *   - Element-wise: Add
 *   - Pooling: MaxPool2D, AvgPool2D, AdaptiveAvgPool1D
 *   - Shape: Flatten, Squeeze, Transpose2D
 *   - Requantize
 */

#include "layer_descriptors.h"
#include "network_kernels.h"
#include "ares_config.h"

#include <stdint.h>
#include <stddef.h>

// Execute a single simple layer
// Returns 0 on success, -1 on error
static int execute_simple_layer(
    const SimpleLayerSpec* layer,
    const BufferSlot* buffers,
    int8_t* l2_arena
) {
    int8_t* in  = l2_arena + buffers[layer->input_slot].offset;
    int8_t* out = l2_arena + buffers[layer->output_slot].offset;

    switch (layer->type) {
        case SIMPLE_RELU:
            // In-place ReLU
            relu_int8_inplace(in, layer->params.activation.numel);
            break;

        case SIMPLE_GELU:
            // In-place GELU with rescaling
            network_gelu_int8_inplace(
                in,
                layer->params.activation.numel,
                layer->params.activation.scale_in,
                layer->params.activation.scale_out
            );
            break;

        case SIMPLE_SILU:
            // In-place SiLU via LUT (uses global silu_lut defined in network_kernels.c)
            // Simple executor routes SiLU through the explicit executor path.
            break;

        case SIMPLE_SOFTMAX:
            // Softmax remains in the explicit executor path (row-wise with i_softmax_lut).
            break;

        case SIMPLE_ADD: {
            int8_t* in2 = l2_arena + buffers[layer->params.binary.input2_slot].offset;
            network_add_int8(
                in, in2, out,
                layer->params.binary.numel,
                layer->params.binary.scale_a,
                layer->params.binary.scale_b,
                layer->params.binary.scale_out
            );
            break;
        }

        case SIMPLE_MAXPOOL2D:
            // Simple executor handles CHW pooling; HWC pooling uses explicit execution.
            network_maxpool_int8(
                in, out,
                layer->params.pool.h,
                layer->params.pool.w,
                layer->params.pool.c,
                layer->params.pool.out_h,
                layer->params.pool.out_w,
                layer->params.pool.kh,
                layer->params.pool.kw,
                layer->params.pool.stride_h,
                layer->params.pool.stride_w,
                layer->params.pool.pad_h,
                layer->params.pool.pad_w
            );
            break;

        case SIMPLE_AVGPOOL2D:
            // Simple executor handles CHW pooling; HWC pooling uses explicit execution.
            network_avgpool_int8(
                in, out,
                layer->params.pool.h,
                layer->params.pool.w,
                layer->params.pool.c,
                layer->params.pool.out_h,
                layer->params.pool.out_w,
                layer->params.pool.kh,
                layer->params.pool.kw,
                layer->params.pool.stride_h,
                layer->params.pool.stride_w,
                layer->params.pool.scale_in,
                layer->params.pool.scale_out
            );
            break;

        case SIMPLE_ADAPTIVE_AVGPOOL1D:
            network_adaptive_avgpool1d_int8(
                in, out,
                layer->params.adaptive_pool.batch,
                layer->params.adaptive_pool.channels,
                layer->params.adaptive_pool.length,
                layer->params.adaptive_pool.output_size,
                layer->params.adaptive_pool.stride_ch,
                layer->params.adaptive_pool.stride_len,
                layer->params.adaptive_pool.batch_stride
            );
            break;

        case SIMPLE_REQUANTIZE:
            // Requantize is in-place
            requantize_int8_inplace(
                in,
                layer->params.requantize.numel,
                layer->params.requantize.scale_in,
                layer->params.requantize.scale_out
            );
            break;

        case SIMPLE_FLATTEN:
        case SIMPLE_SQUEEZE:
            // No-op: these just change logical shape, data stays in place
            break;

        case SIMPLE_TRANSPOSE_2D:
            network_transpose_2d_int8(
                in, out,
                layer->params.transpose.dim0,
                layer->params.transpose.dim1,
                layer->params.transpose.dim2
            );
            break;

        default:
            // Unknown layer type
            return -1;
    }

    return 0;
}

// Execute all simple layers in sequence
// This is the main entry point for the data-driven executor
void execute_simple_layers(
    const SimpleLayerSpec* layers,
    int num_layers,
    const BufferSlot* buffers,
    int8_t* l2_arena
) {
    for (int i = 0; i < num_layers; i++) {
        execute_simple_layer(&layers[i], buffers, l2_arena);
    }
}

// Execute a single simple layer by index (for interleaved execution)
// Used when simple and complex ops must execute in specific order
void execute_simple_layer_by_index(
    const SimpleLayerSpec* layers,
    int layer_idx,
    const BufferSlot* buffers,
    int8_t* l2_arena
) {
    execute_simple_layer(&layers[layer_idx], buffers, l2_arena);
}
