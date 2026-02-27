/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Network Executor - Data-Driven Layer Dispatch
 *
 * This module implements the runtime dispatch loop for executing neural network
 * layers based on their LayerSpec configurations.
 *
 * Architecture:
 *   1. execute_layer() - Main dispatcher that switches on layer->type
 *   2. execute_*() - Per-op-type functions that handle L1/L2 branching
 *   3. Pipeline functions - Called for L1 tiled execution (unchanged)
 *
 * CRITICAL: This file does NOT modify the optimized pipeline functions in
 * network_dma_pipeline.c. All DMA patterns remain unchanged.
 */

#include <stdio.h>
#include <string.h>
#include "network_executor.h"
#include "network_dma_pipeline.h"
#include "network_kernels.h"
#include "ops/op_elementwise.h"
#include "ops/op_pool.h"
#include "ops/op_mhsa.h"
#include "ops/op_norm.h"
#include "ops/op_activation.h"
#include "mem.h"
#include "pmsis.h"

#ifdef ARES_USE_NE16
#include "ne16/ne16_driver.h"
#include "ne16/ne16_linear.h"
#endif

// Current kernels use CL_NUM_CORES for work distribution, so we always fork NUM_CORES.

// Worker function types (same as in network.c.mako)
typedef struct { int8_t *data; size_t size; } relu_args_t;
typedef struct { int8_t *data; size_t size; float scale_in; float scale_out; } requantize_args_t;

// MaxPool args for fused conv+maxpool
typedef struct {
    const int8_t *input;
    int8_t *output;
    uint16_t in_h, in_w, in_ch, out_h, out_w;
    uint16_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
    TensorLayout layout;
} maxpool_args_t;

// Forward declaration for maxpool worker (used in fused conv+maxpool)
static void maxpool_worker(void *arg);

// Worker functions
static void relu_worker(void *arg) {
    relu_args_t *a = (relu_args_t *)arg;
    relu_int8_inplace(a->data, a->size);
}

static void requantize_worker(void *arg) {
    requantize_args_t *a = (requantize_args_t *)arg;
    requantize_int8_inplace(a->data, a->size, a->scale_in, a->scale_out);
}

// L1 versions of workers
static void relu_l1_worker(void *arg) {
    relu_args_t *a = (relu_args_t *)arg;
    relu_int8_inplace_l1(a->data, a->size);
}


/* --- Main Dispatcher --- */

int execute_layer(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    if (!layer || !ctx) return -1;

#ifndef MINIMAL_OUTPUT
    printf("CL: Executing layer '%s' (type=%d)\n", layer->name, layer->type);
#endif

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        memset(ctx->perf_counter, 0, sizeof(layer_perf_t));
        perf_layer_start(layer->name);
    }
#endif

    switch (layer->type) {
        case OP_CONV2D:
            execute_conv2d(layer, ctx);
            break;
        case OP_LINEAR_INT8:
            execute_linear_int8(layer, ctx);
            break;
        case OP_LINEAR_FP32:
            execute_linear_fp32(layer, ctx);
            break;
        case OP_MAXPOOL:
            execute_maxpool(layer, ctx);
            break;
        case OP_AVGPOOL:
            execute_avgpool(layer, ctx);
            break;
        case OP_GLOBAL_AVGPOOL:
            execute_global_avgpool(layer, ctx);
            break;
        case OP_MHSA:
            execute_mhsa(layer, ctx);
            break;
        case OP_CROSS_ATTENTION:
            execute_cross_attention(layer, ctx);
            break;
        case OP_RELU:
            execute_relu(layer, ctx);
            break;
        case OP_REQUANTIZE:
            execute_requantize(layer, ctx);
            break;
        case OP_ADD:
            execute_add(layer, ctx);
            break;
        case OP_CONCAT:
            execute_concat(layer, ctx);
            break;
        case OP_LAYERNORM:
            execute_layernorm(layer, ctx);
            break;
        case OP_GELU:
            execute_gelu(layer, ctx);
            break;
        case OP_MEAN:
            execute_mean(layer, ctx);
            break;
        case OP_ALTERNATING_ATTENTION:
            execute_alternating_attention(layer, ctx);
            break;
        case OP_CROSS_ATTN_SELF_REFINE:
            // Dispatched via template-generated code in network.c.mako
            break;
        case OP_CLASSIFICATION_HEAD_MLP:
            // Dispatched via template-generated code in network.c.mako
            break;
        case OP_TRANSPOSE_2D:
            execute_transpose_2d(layer, ctx);
            break;
        case OP_EMBEDDING:
            execute_embedding(layer, ctx);
            break;
        case OP_GROUPNORM:
            execute_groupnorm(layer, ctx);
            break;
        case OP_RFFT:
            execute_rfft(layer, ctx);
            break;
        case OP_CONV1D_DEPTHWISE:
            execute_conv1d_depthwise(layer, ctx);
            break;
        case OP_SILU:
            execute_silu(layer, ctx);
            break;
        case OP_SSM:
            execute_ssm(layer, ctx);
            break;
        case OP_MAMBA_BLOCK:
            execute_mamba_block(layer, ctx);
            break;
        case OP_MAMBA_WRAPPER:
            execute_mamba_wrapper(layer, ctx);
            break;
        case OP_PATCH_EMBED:
            execute_patch_embed(layer, ctx);
            break;
        case OP_POS_EMBED:
            execute_pos_embed(layer, ctx);
            break;
        case OP_ZEROPAD2D:
            execute_zeropad2d(layer, ctx);
            break;
        case OP_FLATTEN:
        case OP_SQUEEZE:
        case OP_RESHAPE:
            // No-op layers - just update buffer tracking
            break;
        case OP_SOFTMAX:
            // Softmax is typically fused into MHSA
            break;
        case OP_CHW_TO_HWC:
            execute_chw_to_hwc(layer, ctx);
            break;
        case OP_HWC_TO_CHW:
            execute_hwc_to_chw(layer, ctx);
            break;
#ifdef ARES_LLAMA_SUPPORT
        case OP_RMSNORM:
            execute_rmsnorm(layer, ctx);
            break;
        case OP_SWIGLU_FFN:
            execute_swiglu_ffn(layer, ctx);
            break;
        case OP_LLAMA_BLOCK:
            execute_llama_block(layer, ctx);
            break;
        case OP_MHSA_AUTOREGRESSIVE:
            execute_mhsa_autoregressive(layer, ctx);
            break;
        case OP_RESIDUAL_ADD:
            execute_residual_add(layer, ctx);
            break;
#endif // ARES_LLAMA_SUPPORT
#ifdef ARES_USE_NE16
        case OP_LINEAR_NE16:
            execute_linear_ne16(layer, ctx);
            break;
        case OP_CONV2D_1X1_NE16:
            execute_conv2d_1x1_ne16(layer, ctx);
            break;
        case OP_CONV2D_3X3_NE16:
            execute_conv2d_3x3_ne16(layer, ctx);
            break;
#ifdef ARES_NE16_DEPTHWISE
        case OP_CONV2D_3X3_DW_NE16:
            if (layer->params.conv2d_ne16.ne16_dw_spatial_tiling) {
                execute_conv2d_3x3_dw_ne16_tiled(layer, ctx);
            } else {
                execute_conv2d_3x3_dw_ne16(layer, ctx);
            }
            break;
#endif // ARES_NE16_DEPTHWISE
#endif
        default:
            printf("CL: ERROR - Unknown layer type %d for '%s'\n",
                   layer->type, layer->name);
            return -1;
    }

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        perf_layer_end(layer->name, ctx->perf_counter);
        perf_layer_record(layer->name, ctx->perf_counter);
    }
#endif

    return 0;
}

/* --- Conv2D Execution --- */

// We use conv2d_tile_worker from network_kernels.h (declared extern)

void execute_conv2d(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const conv2d_pipeline_config_t *params = &layer->params.conv2d;

    // Check if we have L1 buffer and it's large enough for double-buffered tiling
    // IMPORTANT: Must match template's calculation including weights for weight_tiling_enabled
    size_t single_buffer_size = params->l1_input_size + params->l1_output_size;
    if (params->weight_tiling_enabled) {
        single_buffer_size += params->l1_weight_size;
    }
    size_t required_l1 = 2 * single_buffer_size;  // Double-buffering

    // Calculate actual full tensor sizes (for single-tile fast path)
    size_t input_bytes = (size_t)params->in_h * params->in_w * params->in_ch;
    size_t conv_output_bytes = (size_t)params->out_h * params->out_w * params->out_ch;
    size_t weight_bytes = (size_t)params->out_ch * params->in_ch * params->kernel_h * params->kernel_w;

    // For fused maxpool: need space for conv output (temp) + pooled output
    size_t pool_output_bytes = 0;
    size_t output_bytes = conv_output_bytes;
    if (params->fusion_maxpool) {
        pool_output_bytes = (size_t)params->pool_out_h * params->pool_out_w * params->out_ch;
        output_bytes = conv_output_bytes + pool_output_bytes;  // Need both buffers in L1
    }
    size_t actual_single_buffer = input_bytes + output_bytes + weight_bytes;

    // For very small single-tile convolutions, pipeline overhead can exceed compute time.
    // Use L2-only when: num_tiles=1 AND output is tiny (< 1KB).
    // This avoids ~100K cycle DMA pipeline setup cost for tiny workloads.
    int small_single_tile = (params->num_tiles == 1 && output_bytes < 1024);

    // Single-tile fast path: Skip pipeline overhead for single-tile L1 execution
    // Uses simple DMA sequence instead of full double-buffered pipeline
    // Use actual full tensor sizes (not tile sizes which include halos)
    // Allow weight_tiling_enabled if num_out_ch_tiles=1 (all weights fit in one tile)
    int weights_fit_single_tile = !params->weight_tiling_enabled ||
                                  (params->weight_tiling_enabled && params->num_out_ch_tiles == 1);
    int use_single_tile_l1 = (params->num_tiles == 1) && !small_single_tile &&
                              ctx->l1_buffer && (actual_single_buffer <= ctx->l1_buffer_size) &&
                              weights_fit_single_tile;

    int use_l1_tiling = ctx->l1_buffer && ctx->l1_buffer_size >= required_l1 &&
                        params->num_tiles > 0 && !small_single_tile && !use_single_tile_l1;

    if (use_single_tile_l1) {
        // Fast path for single-tile: Simple DMA sequence (no pipeline overhead)
        // This avoids the ~100K cycle overhead of double-buffered pipeline setup
#ifndef MINIMAL_OUTPUT
        if (params->fusion_maxpool) {
            printf("CL: %s using L1 single-tile fast path with fused MaxPool (%zu bytes)\n",
                   layer->name, actual_single_buffer);
        } else {
            printf("CL: %s using L1 single-tile fast path (%zu bytes)\n", layer->name, actual_single_buffer);
        }
#endif

        // Layout: [input | conv_output | (pool_output if fused) | weights] in L1
        int8_t *input_l1 = ctx->l1_buffer;
        int8_t *conv_output_l1 = ctx->l1_buffer + input_bytes;
        int8_t *pool_output_l1 = params->fusion_maxpool ?
                                  (ctx->l1_buffer + input_bytes + conv_output_bytes) : NULL;
        int8_t *weights_l1 = ctx->l1_buffer + input_bytes + output_bytes;

        // DMA: Copy input and weights L2 -> L1 (can overlap)
        pi_cl_dma_copy_t dma_in, dma_wt;

        dma_in.ext = (uint32_t)ctx->input_buffer_l2;
        dma_in.loc = (uint32_t)input_l1;
        dma_in.size = input_bytes;
        dma_in.dir = PI_CL_DMA_DIR_EXT2LOC;
        dma_in.merge = 0;
        pi_cl_dma_memcpy(&dma_in);

        dma_wt.ext = (uint32_t)ctx->weight_l2;
        dma_wt.loc = (uint32_t)weights_l1;
        dma_wt.size = weight_bytes;
        dma_wt.dir = PI_CL_DMA_DIR_EXT2LOC;
        dma_wt.merge = 0;
        pi_cl_dma_memcpy(&dma_wt);

        pi_cl_dma_wait(&dma_in);
        pi_cl_dma_wait(&dma_wt);

        // Compute conv (+ fused relu if enabled) on L1
        conv2d_tile_args_t conv_args = {
            .tile_input_l1 = input_l1,
            .tile_output_l1 = conv_output_l1,
            .weights_l2 = weights_l1,  // Actually L1 now
            .weight_row_stride = 0,
            .bias_l2 = (const int32_t *)ctx->bias_l2,  // Bias stays in L2 (small)
            .tile_in_h = params->in_h,
            .tile_in_w = params->in_w,
            .tile_out_h = params->out_h,
            .tile_out_w = params->out_w,
            .in_ch = params->in_ch,
            .out_ch = params->out_ch,
            .groups = params->groups,
            .kernel_h = params->kernel_h,
            .kernel_w = params->kernel_w,
            .stride_h = params->stride_h,
            .stride_w = params->stride_w,
            .pad_h = params->pad_h,
            .pad_w = params->pad_w,
            .scale_input = params->scale_input,
            .scale_weight = params->scale_weight,
            .scale_output = params->scale_output,
            .cluster_dev = ctx->cluster_dev,
            .fusion_relu = params->fusion_relu,
            .fusion_quant = params->fusion_maxpool ? 0 : params->fusion_quant,  // Defer quant if pooling
            .quant_scale_in = params->quant_scale_in,
            .quant_scale_out = params->quant_scale_out,
            .layout = (uint8_t)params->layout
        };

#ifdef ENABLE_PERF_COUNTERS
        perf_compute_start();
#endif
        if (params->fusion_relu || params->fusion_quant) {
            pi_cl_team_fork(NUM_CORES, conv2d_tile_worker_with_fusion, &conv_args);
        } else {
            pi_cl_team_fork(NUM_CORES, conv2d_tile_worker, &conv_args);
        }

        // Apply fused MaxPool if enabled (conv output is already in L1)
        if (params->fusion_maxpool) {
#ifndef MINIMAL_OUTPUT
            printf("CL: %s fused maxpool: conv_out_l1=%p pool_out_l1=%p fused_l2=%p\n",
                   layer->name, (void*)conv_output_l1, (void*)pool_output_l1,
                   (void*)ctx->fused_output_buffer_l2);
            printf("CL: %s fused maxpool dims: in=%dx%dx%d out=%dx%dx%d kernel=%dx%d stride=%dx%d\n",
                   layer->name, params->out_h, params->out_w, params->out_ch,
                   params->pool_out_h, params->pool_out_w, params->out_ch,
                   params->pool_kernel_h, params->pool_kernel_w,
                   params->pool_stride_h, params->pool_stride_w);
            printf("CL: %s conv_out_l1[0..4]: %d %d %d %d %d\n", layer->name,
                   conv_output_l1[0], conv_output_l1[1], conv_output_l1[2],
                   conv_output_l1[3], conv_output_l1[4]);
#endif
            // MaxPool on L1: conv_output_l1 -> pool_output_l1
            maxpool_args_t pool_args = {
                .input = conv_output_l1,
                .output = pool_output_l1,
                .in_h = params->out_h,
                .in_w = params->out_w,
                .in_ch = params->out_ch,
                .out_h = params->pool_out_h,
                .out_w = params->pool_out_w,
                .kernel_h = params->pool_kernel_h,
                .kernel_w = params->pool_kernel_w,
                .stride_h = params->pool_stride_h,
                .stride_w = params->pool_stride_w,
                .pad_h = 0,
                .pad_w = 0,
                .layout = params->layout
            };
            pi_cl_team_fork(NUM_CORES, maxpool_worker, &pool_args);
#ifndef MINIMAL_OUTPUT
            printf("CL: %s pool_out_l1[0..4]: %d %d %d %d %d\n", layer->name,
                   pool_output_l1[0], pool_output_l1[1], pool_output_l1[2],
                   pool_output_l1[3], pool_output_l1[4]);
#endif
        }
#ifdef ENABLE_PERF_COUNTERS
        if (ctx->perf_counter) {
            ctx->perf_counter->compute_cycles += perf_compute_end();
        }
#endif

        // DMA: Copy final output L1 -> L2
        pi_cl_dma_copy_t dma_out;
        if (params->fusion_maxpool) {
            // Output is the pooled result, goes to fused_output_buffer_l2
            dma_out.ext = (uint32_t)ctx->fused_output_buffer_l2;
            dma_out.loc = (uint32_t)pool_output_l1;
            dma_out.size = pool_output_bytes;
        } else {
            dma_out.ext = (uint32_t)ctx->output_buffer_l2;
            dma_out.loc = (uint32_t)conv_output_l1;
            dma_out.size = conv_output_bytes;
        }
        dma_out.dir = PI_CL_DMA_DIR_LOC2EXT;
        dma_out.merge = 0;
        pi_cl_dma_memcpy(&dma_out);
        pi_cl_dma_wait(&dma_out);

    } else if (use_l1_tiling) {
        // L1 tiled execution - use pipeline function
#ifndef MINIMAL_OUTPUT
        printf("CL: %s using L1 double-buffer tiling: %d tiles (%dx%d) [PIPELINED]\n",
               layer->name, params->num_tiles, params->num_tiles_h, params->num_tiles_w);
        printf("CL: DEBUG %s params: in_h=%d in_w=%d in_ch=%d out_h=%d out_w=%d out_ch=%d\n",
               layer->name, params->in_h, params->in_w, params->in_ch,
               params->out_h, params->out_w, params->out_ch);
        printf("CL: DEBUG %s params: weight_tiling=%d tile_out_ch=%d num_out_ch_tiles=%d\n",
               layer->name, params->weight_tiling_enabled, params->tile_out_ch, params->num_out_ch_tiles);
        printf("CL: DEBUG %s params: l1_input=%zu l1_output=%zu l1_weight=%zu\n",
               layer->name, params->l1_input_size, params->l1_output_size, params->l1_weight_size);
#endif

        // Create a mutable copy of the config and patch runtime pointers
        conv2d_pipeline_config_t cfg = *params;
        cfg.layer_name = layer->name;
        cfg.input_buffer_l2 = ctx->input_buffer_l2;
        cfg.output_buffer_l2 = ctx->output_buffer_l2;
        cfg.weight_l2 = ctx->weight_l2;
        cfg.bias_l2 = (int32_t *)ctx->bias_l2;
        cfg.l1_buffer = ctx->l1_buffer;
        cfg.l1_buffer_size = ctx->l1_buffer_size;
        cfg.cluster_dev = ctx->cluster_dev;
        cfg.ram_dev = ctx->ram_dev;
#ifdef ENABLE_PERF_COUNTERS
        cfg.perf_counter = ctx->perf_counter;
#endif
        cfg.golden_buffer = NULL;
        cfg.golden_size = 0;
        cfg.compare_buffer = NULL;

#ifndef MINIMAL_OUTPUT
        printf("CL: DEBUG %s cfg: in_buf=%p out_buf=%p wt_buf=%p bias=%p\n",
               cfg.layer_name,
               (void *)cfg.input_buffer_l2,
               (void *)cfg.output_buffer_l2,
               (void *)cfg.weight_l2,
               (void *)cfg.bias_l2);
        printf("CL: DEBUG %s cfg: l1_buf=%p l1_size=%zu\n",
               cfg.layer_name, (void *)cfg.l1_buffer, cfg.l1_buffer_size);
#endif

        conv2d_tiled_l1_pipeline(&cfg);
    } else {
        // L2-only execution (no tiling)
#ifndef MINIMAL_OUTPUT
        printf("CL: %s using L2-only execution (no L1 tiling)\n", layer->name);
#endif

        conv2d_tile_args_t conv_args = {
            .tile_input_l1 = ctx->input_buffer_l2,
            .tile_output_l1 = ctx->output_buffer_l2,
            .weights_l2 = (const int8_t *)ctx->weight_l2,
            .weight_row_stride = 0,  // Use default (in_ch * kernel_h * kernel_w)
            .bias_l2 = (const int32_t *)ctx->bias_l2,
            .tile_in_h = params->in_h,
            .tile_in_w = params->in_w,
            .tile_out_h = params->out_h,
            .tile_out_w = params->out_w,
            .in_ch = params->in_ch,
            .out_ch = params->out_ch,
            .kernel_h = params->kernel_h,
            .kernel_w = params->kernel_w,
            .stride_h = params->stride_h,
            .stride_w = params->stride_w,
            .pad_h = params->pad_h,
            .pad_w = params->pad_w,
            .scale_input = params->scale_input,
            .scale_weight = params->scale_weight,
            .scale_output = params->scale_output,
            .cluster_dev = ctx->cluster_dev,
            .fusion_relu = params->fusion_relu,
            .fusion_quant = params->fusion_quant,
            .quant_scale_in = params->quant_scale_in,
            .quant_scale_out = params->quant_scale_out,
            .layout = (uint8_t)params->layout
        };

#ifdef ENABLE_PERF_COUNTERS
        perf_compute_start();
#endif
        // Use fusion-aware worker if fusion is enabled
        if (params->fusion_relu || params->fusion_quant) {
            pi_cl_team_fork(NUM_CORES, conv2d_tile_worker_with_fusion, &conv_args);
        } else {
            pi_cl_team_fork(NUM_CORES, conv2d_tile_worker, &conv_args);
        }
#ifdef ENABLE_PERF_COUNTERS
        if (ctx->perf_counter) {
            ctx->perf_counter->compute_cycles += perf_compute_end();
        }
#endif
    }
}

/* --- Linear INT8 Execution --- */

void execute_linear_int8(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const linear_int8_pipeline_config_t *params = &layer->params.linear_int8;

    // Check if we have L1 buffer for tiled execution
    size_t required_l1 = params->l1_input_size + params->l1_output_size + params->l1_weight_size;
    int use_l1_tiling = ctx->l1_buffer && ctx->l1_buffer_size >= required_l1 &&
                        params->num_tiles > 0;

    if (use_l1_tiling) {
#ifndef MINIMAL_OUTPUT
        printf("CL: %s using L1 tiled linear: %d tiles [PIPELINED]\n",
               layer->name, params->num_tiles);
#endif

        // Create mutable copy and patch runtime pointers
        linear_int8_pipeline_config_t cfg = *params;
        cfg.layer_name = layer->name;
        cfg.input_buffer_l2 = ctx->input_buffer_l2;
        cfg.output_buffer_l2 = ctx->output_buffer_l2;
        cfg.weight_l2 = ctx->weight_l2;
        cfg.bias_l2 = (int32_t *)ctx->bias_l2;
        cfg.l1_buffer = ctx->l1_buffer;
        cfg.l1_buffer_size = ctx->l1_buffer_size;
        cfg.ram_dev = ctx->ram_dev;
#ifdef ENABLE_PERF_COUNTERS
        cfg.perf_counter = ctx->perf_counter;
#endif
        cfg.golden_buffer = NULL;
        cfg.golden_size = 0;
        cfg.compare_buffer = NULL;

        // L3 tiling: Patch L3 source addresses for dynamic streaming
        if (ctx->l3_tiling_enabled) {
            cfg.l3_weight_addr = ctx->l3_weight_addr;
            cfg.l3_bias_addr = ctx->l3_bias_addr;
            cfg.l3_output_addr = ctx->l3_output_addr;
        }

        linear_int8_tiled_l1_pipeline(&cfg);
    } else {
        // L2-only execution (weights too large for L1 tiling)
#ifndef MINIMAL_OUTPUT
        printf("CL: %s using L2-only linear execution (required=%zu > L1=%zu)\n",
               layer->name, required_l1, ctx->l1_buffer_size);
#endif

#ifdef ENABLE_PERF_COUNTERS
        perf_compute_start();
#endif
        // Process all tokens in a single fork call to reduce synchronization overhead.
        // network_linear_int8 uses pi_core_id() internally for parallelization across
        // output features, so it MUST be called via pi_cl_team_fork.
        // Batching all tokens in one fork amortizes the ~80 cycle fork overhead.
        linear_tile_batched_args_t linear_args = {
            .input_base = ctx->input_buffer_l2,
            .weights_l2 = ctx->weight_l2,
            .bias_l2 = ctx->bias_l2,
            .output_base = ctx->output_buffer_l2,
            .dim_in = params->in_features,
            .dim_out = params->out_features,
            .batch_tokens = params->batch_tokens,
            .scale_input = params->scale_input,
            .scale_weight = params->scale_weight,
            .scale_output = params->scale_output
        };
        pi_cl_team_fork(NUM_CORES, linear_tile_batched_worker, &linear_args);
#ifdef ENABLE_PERF_COUNTERS
        if (ctx->perf_counter) {
            ctx->perf_counter->compute_cycles += perf_compute_end();
        }
#endif

        // Apply fused operations.
        // Keep L2-only fallback numerically aligned with L1-tiled path:
        // 1) ReLU in-place
        // 2) Optional rescale from linear output scale -> ReLU advertised output scale
        // 3) Optional final quant fuse (quant_scale_in -> quant_scale_out)
        if (params->fusion_relu) {
            size_t output_size = params->batch_tokens * params->out_features;
            relu_args_t relu_args = { .data = ctx->output_buffer_l2, .size = output_size };
            pi_cl_team_fork(NUM_CORES, relu_worker, &relu_args);

            if (params->relu_output_scale != 0.0f &&
                fabsf(params->relu_output_scale - params->scale_output) > 1e-5f) {
                requantize_args_t relu_requant_args = {
                    .data = ctx->output_buffer_l2,
                    .size = output_size,
                    .scale_in = params->scale_output,
                    .scale_out = params->relu_output_scale
                };
                pi_cl_team_fork(NUM_CORES, requantize_worker, &relu_requant_args);
            }
        }
        if (params->fusion_quant) {
            size_t output_size = params->batch_tokens * params->out_features;
            requantize_args_t quant_args = {
                .data = ctx->output_buffer_l2,
                .size = output_size,
                .scale_in = params->quant_scale_in,
                .scale_out = params->quant_scale_out
            };
            pi_cl_team_fork(NUM_CORES, requantize_worker, &quant_args);
        }
    }
}

/* --- Linear FP32 Execution (Final Layer) --- */

// Worker args for L2-only linear FP32 execution
typedef struct {
    const int8_t *input;
    const int8_t *weights;
    const float *bias;
    float *output;
    uint16_t in_features;
    uint16_t out_features;
    float scale_input;
    float scale_weight;
} linear_fp32_exec_args_t;

static void linear_fp32_exec_worker(void *arg) {
    linear_fp32_exec_args_t *a = (linear_fp32_exec_args_t *)arg;
    network_linear_int8_to_fp32(
        a->input, a->weights, a->bias, a->output,
        a->in_features, a->out_features,
        a->scale_input, a->scale_weight
    );
}

void execute_linear_fp32(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const linear_fp32_pipeline_config_t *params = &layer->params.linear_fp32;

    // Check if we have L1 buffer for tiled execution
    size_t required_l1 = params->l1_input_size + params->l1_output_size + params->l1_weight_size;
    int use_l1_tiling = ctx->l1_buffer && ctx->l1_buffer_size >= required_l1 &&
                        params->num_tiles > 0;

    if (use_l1_tiling) {
#ifndef MINIMAL_OUTPUT
        printf("CL: %s using L1 tiled fp32 linear: %d tiles [PIPELINED]\n",
               layer->name, params->num_tiles);
#endif

        linear_fp32_pipeline_config_t cfg = *params;
        cfg.layer_name = layer->name;
        cfg.input_buffer_l2 = ctx->input_buffer_l2;
        cfg.output_buffer_l2 = (float *)ctx->output_buffer_l2;
        cfg.weight_l2 = ctx->weight_l2;
        cfg.bias_l2 = (float *)ctx->bias_l2;
        cfg.l1_buffer = ctx->l1_buffer;
        cfg.l1_buffer_size = ctx->l1_buffer_size;
        cfg.ram_dev = ctx->ram_dev;
#ifdef ENABLE_PERF_COUNTERS
        cfg.perf_counter = ctx->perf_counter;
#endif

        linear_fp32_tiled_l1_pipeline(&cfg);
    } else {
#ifndef MINIMAL_OUTPUT
        printf("CL: %s using L2-only fp32 linear execution\n", layer->name);
#endif

        // Must use pi_cl_team_fork because network_linear_int8_to_fp32 uses pi_core_id()
        // to parallelize across cores. Direct call from Core 8 would skip all work.
        linear_fp32_exec_args_t fp32_args = {
            .input = ctx->input_buffer_l2,
            .weights = ctx->weight_l2,
            .bias = (const float *)ctx->bias_l2,
            .output = (float *)ctx->output_buffer_l2,
            .in_features = params->in_features,
            .out_features = params->out_features,
            .scale_input = params->scale_input,
            .scale_weight = params->scale_weight
        };

#ifdef ENABLE_PERF_COUNTERS
        perf_compute_start();
#endif
        pi_cl_team_fork(NUM_CORES, linear_fp32_exec_worker, &fp32_args);
#ifdef ENABLE_PERF_COUNTERS
        if (ctx->perf_counter) {
            ctx->perf_counter->compute_cycles += perf_compute_end();
        }
#endif

    }
}

/* --- ReLU Execution --- */

void execute_relu(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const struct { void *buffer; int size; float scale_in, scale_out; } *params = &layer->params.relu;
    size_t numel = params->size;

    // Step 1: ReLU is inplace on the input buffer
    relu_args_t relu_args = {
        .data = ctx->input_buffer_l2,
        .size = numel
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, relu_worker, &relu_args);

    // Step 2: Requantize if scales differ (QuantReLU can change scale)
    float scale_diff = params->scale_in - params->scale_out;
    if (scale_diff < 0) scale_diff = -scale_diff;
    if (scale_diff > 1e-8f) {
        // Scales differ, need to requantize
        requantize_args_t quant_args = {
            .data = ctx->input_buffer_l2,
            .size = numel,
            .scale_in = params->scale_in,
            .scale_out = params->scale_out
        };
        pi_cl_team_fork(NUM_CORES, requantize_worker, &quant_args);
    }

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

/* --- Requantize Execution --- */

void execute_requantize(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const struct { void *buffer; int size; float scale_in, scale_out; } *params =
        &layer->params.requantize;

    // Check for identity requantize (same scales - no-op)
    float scale_diff = params->scale_in - params->scale_out;
    if (scale_diff < 0) scale_diff = -scale_diff;
    if (scale_diff < 1e-8f) {
#ifndef MINIMAL_OUTPUT
        printf("CL: %s identity requantize (scale unchanged) - skipping\n", layer->name);
#endif
        return;  // Skip - no actual requantization needed
    }

    requantize_args_t quant_args = {
        .data = ctx->input_buffer_l2,
        .size = params->size,
        .scale_in = params->scale_in,
        .scale_out = params->scale_out
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, requantize_worker, &quant_args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

/* --- Pooling Execution (MaxPool, AvgPool, GlobalAvgPool) --- */

// maxpool_args_t is defined at the top of the file for use in fused conv+maxpool

static void maxpool_worker(void *arg) {
    maxpool_args_t *a = (maxpool_args_t *)arg;
    if (a->layout == LAYOUT_HWC) {
        network_maxpool_int8_hwc(a->input, a->output, a->in_h, a->in_w, a->in_ch,
                                  a->out_h, a->out_w, a->kernel_h, a->kernel_w,
                                  a->stride_h, a->stride_w, a->pad_h, a->pad_w);
    } else {
        network_maxpool_int8(a->input, a->output, a->in_h, a->in_w, a->in_ch,
                             a->out_h, a->out_w, a->kernel_h, a->kernel_w,
                             a->stride_h, a->stride_w, a->pad_h, a->pad_w);
    }
}

void execute_maxpool(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.maxpool) *params = &layer->params.maxpool;

    maxpool_args_t pool_args = {
        .input = ctx->input_buffer_l2,
        .output = ctx->output_buffer_l2,
        .in_h = params->in_h,
        .in_w = params->in_w,
        .in_ch = params->channels,
        .out_h = params->out_h,
        .out_w = params->out_w,
        .kernel_h = params->kernel_h,
        .kernel_w = params->kernel_w,
        .stride_h = params->stride_h,
        .stride_w = params->stride_w,
        .pad_h = params->pad_h,
        .pad_w = params->pad_w,
        .layout = params->layout
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, maxpool_worker, &pool_args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

    // Handle fused operations
    if (params->fusion_quant) {
        size_t output_size = params->out_h * params->out_w * params->channels;
        requantize_args_t quant_args = {
            .data = ctx->output_buffer_l2,
            .size = output_size,
            .scale_in = params->quant_scale_in,
            .scale_out = params->quant_scale_out
        };
        pi_cl_team_fork(NUM_CORES, requantize_worker, &quant_args);
    }
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    uint16_t in_h, in_w, in_ch, out_h, out_w;
    uint16_t kernel_h, kernel_w, stride_h, stride_w;
    float scale_in, scale_out;
    TensorLayout layout;  // CHW or HWC
} avgpool_args_t;

static void avgpool_worker(void *arg) {
    avgpool_args_t *a = (avgpool_args_t *)arg;
    if (a->layout == LAYOUT_HWC) {
        network_avgpool_int8_hwc(a->input, a->output, a->in_h, a->in_w, a->in_ch,
                                  a->out_h, a->out_w, a->kernel_h, a->kernel_w,
                                  a->stride_h, a->stride_w, a->scale_in, a->scale_out);
    } else {
        network_avgpool_int8(a->input, a->output, a->in_h, a->in_w, a->in_ch,
                             a->out_h, a->out_w, a->kernel_h, a->kernel_w,
                             a->stride_h, a->stride_w, a->scale_in, a->scale_out);
    }
}

void execute_avgpool(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.avgpool) *params = &layer->params.avgpool;

    avgpool_args_t pool_args = {
        .input = ctx->input_buffer_l2,
        .output = ctx->output_buffer_l2,
        .in_h = params->in_h,
        .in_w = params->in_w,
        .in_ch = params->channels,
        .out_h = params->out_h,
        .out_w = params->out_w,
        .kernel_h = params->kernel_h,
        .kernel_w = params->kernel_w,
        .stride_h = params->stride_h,
        .stride_w = params->stride_w,
        .scale_in = params->scale_in,
        .scale_out = params->scale_out,
        .layout = params->layout
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, avgpool_worker, &pool_args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    uint16_t batch, ch, h, w;
    float scale_in, scale_out;
    TensorLayout layout;
} global_avgpool_args_t;

static void global_avgpool_worker(void *arg) {
    global_avgpool_args_t *a = (global_avgpool_args_t *)arg;
    if (a->layout == LAYOUT_HWC) {
        network_global_avgpool_int8_hwc(a->input, a->output, a->ch, a->h, a->w,
                                        a->scale_in, a->scale_out);
    } else {
        network_global_avgpool_int8(a->input, a->output, a->batch, a->ch, a->h, a->w,
                                    a->scale_in, a->scale_out);
    }
}

void execute_global_avgpool(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.global_avgpool) *params = &layer->params.global_avgpool;

    global_avgpool_args_t pool_args = {
        .input = ctx->input_buffer_l2,
        .output = ctx->output_buffer_l2,
        .batch = params->batch,
        .ch = params->channels,
        .h = params->h,
        .w = params->w,
        .scale_in = params->scale_in,
        .scale_out = params->scale_out,
        .layout = params->layout
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, global_avgpool_worker, &pool_args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

/* --- MHSA Execution --- */

void execute_mhsa(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const mhsa_pipeline_config_t *params = &layer->params.mhsa;

    // MHSA always uses the pipeline function
#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing MHSA (seq_len=%d, num_heads=%d, head_dim=%d)\n",
           layer->name, params->seq_len, params->num_heads, params->head_dim);
#endif

    // Create mutable copy and patch runtime pointers
    mhsa_pipeline_config_t cfg = *params;
    cfg.layer_name = layer->name;
    cfg.input_buffer_l2 = ctx->input_buffer_l2;
    cfg.output_buffer_l2 = ctx->output_buffer_l2;
    cfg.l1_buffer = ctx->l1_buffer;
    cfg.l1_buffer_size = ctx->l1_buffer_size;
    cfg.ram_dev = ctx->ram_dev;

    // Patch MHSA weight pointers from runtime context
    cfg.q_weight_l2 = ctx->mhsa_q_weight_l2;
    cfg.q_bias_l2 = ctx->mhsa_q_bias_l2;
    cfg.k_weight_l2 = ctx->mhsa_k_weight_l2;
    cfg.k_bias_l2 = ctx->mhsa_k_bias_l2;
    cfg.v_weight_l2 = ctx->mhsa_v_weight_l2;
    cfg.v_bias_l2 = ctx->mhsa_v_bias_l2;
    cfg.out_weight_l2 = ctx->mhsa_out_weight_l2;
    cfg.out_bias_l2 = ctx->mhsa_out_bias_l2;

    // Patch MHSA intermediate projection buffers
    cfg.q_buffer_l2 = ctx->mhsa_q_buffer_l2;
    cfg.k_buffer_l2 = ctx->mhsa_k_buffer_l2;
    cfg.v_buffer_l2 = ctx->mhsa_v_buffer_l2;

    // Head-contiguous optimization: in-place permute using output buffer as scratch
    cfg.use_head_contiguous = 1;
    cfg.use_inplace_permute = 1;
    cfg.q_permuted_l2 = ctx->mhsa_q_buffer_l2;  // Q permutes in-place
    cfg.k_permuted_l2 = ctx->mhsa_k_buffer_l2;  // K permutes in-place
    cfg.v_permuted_l2 = ctx->mhsa_v_buffer_l2;  // V permutes in-place
    cfg.m_permuted_l2 = ctx->output_buffer_l2;  // M uses output buffer
    cfg.permute_scratch_l2 = ctx->output_buffer_l2;  // Scratch = output buffer

    // Auto-enable integer softmax if LUT is provided
    // This is critical for L1 tiled path performance (reduces L1 requirement by 29KB)
    cfg.use_integer_softmax = (cfg.softmax_lut != NULL) ? 1 : 0;

#ifdef ENABLE_PERF_COUNTERS
    cfg.perf_counter = ctx->perf_counter;
#endif
    cfg.golden_buffer = NULL;
    cfg.golden_size = 0;
    cfg.compare_buffer = NULL;

    mhsa_tiled_l1_pipeline(&cfg);
}

/* --- Cross-Attention Execution --- */

void execute_cross_attention(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.cross_attention) *params = &layer->params.cross_attention;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing cross-attention (kv_len=%d, num_queries=%d, embed_dim=%d, num_heads=%d)\n",
           layer->name, params->kv_len, params->num_queries, params->embed_dim, params->num_heads);
#endif

    // Cross-attention:
    //   - Q comes from learned query embedding (cross_attn_query_embed)
    //   - K, V come from input (kv_input = ctx->input_buffer_l2)
    //   - Output goes to ctx->output_buffer_l2

    // Compute requant parameters for softmax (same as MHSA)
    int32_t requant_mul = 1;
    int32_t requant_shift = 0;  // Not used with float softmax

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    network_cross_attention_int8_parallel(
        ctx->input_buffer_l2,           // kv_input
        ctx->output_buffer_l2,          // output
        ctx->cross_attn_q_proj_out,     // q_proj_out (scratch)
        ctx->cross_attn_k_proj_out,     // k_proj_out (scratch)
        ctx->cross_attn_v_proj_out,     // v_proj_out (scratch)
        ctx->cross_attn_context_out,    // context_out (scratch)
        ctx->cross_attn_query_embed,    // query_embed (learned)
        ctx->mhsa_q_weight_l2, ctx->mhsa_q_bias_l2,     // Q projection
        ctx->mhsa_k_weight_l2, ctx->mhsa_k_bias_l2,     // K projection
        ctx->mhsa_v_weight_l2, ctx->mhsa_v_bias_l2,     // V projection
        ctx->mhsa_out_weight_l2, ctx->mhsa_out_bias_l2, // Output projection
        params->batch,
        params->kv_len,
        params->num_queries,
        params->embed_dim,
        params->num_heads,
        params->scale_kv_in,
        params->scale_query_in,
        params->scale_q_weight,
        params->scale_k_weight,
        params->scale_v_weight,
        params->scale_out_weight,
        params->scale_q,
        params->scale_k,
        params->scale_v,
        params->scale_output,
        params->softmax_scale,
        requant_mul,
        requant_shift,
        ctx->l1_buffer,                 // L1 scratch
        ctx->l1_buffer_size
    );

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

/* --- Add Execution (Element-wise) --- */

// Worker args and function for add - must be outside execute_add (C doesn't allow nested functions)
typedef struct {
    const int8_t *input_a, *input_b;
    int8_t *output;
    uint32_t size;
    float scale_a, scale_b, scale_out;
} add_exec_args_t;

static void add_exec_worker(void *arg) {
    add_exec_args_t *a = (add_exec_args_t *)arg;
    network_add_int8(a->input_a, a->input_b, a->output, a->size,
                    a->scale_a, a->scale_b, a->scale_out);
}

void execute_add(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.add) *params = &layer->params.add;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing element-wise add (%d elements)\n",
           layer->name, params->size);
    printf("CL: DEBUG %s input_a=%p input_b=%p output=%p\n",
           layer->name,
           (void *)ctx->input_buffer_l2,
           (void *)ctx->input_b_buffer_l2,
           (void *)ctx->output_buffer_l2);
    if (ctx->input_buffer_l2) {
        printf("CL: DEBUG %s input_a[0..4]: %d %d %d %d %d\n", layer->name,
               ctx->input_buffer_l2[0], ctx->input_buffer_l2[1], ctx->input_buffer_l2[2],
               ctx->input_buffer_l2[3], ctx->input_buffer_l2[4]);
    }
    if (ctx->input_b_buffer_l2) {
        printf("CL: DEBUG %s input_b[0..4]: %d %d %d %d %d\n", layer->name,
               ctx->input_b_buffer_l2[0], ctx->input_b_buffer_l2[1], ctx->input_b_buffer_l2[2],
               ctx->input_b_buffer_l2[3], ctx->input_b_buffer_l2[4]);
    }
    printf("CL: DEBUG %s scales: a=%f b=%f out=%f\n", layer->name,
           params->scale_a, params->scale_b, params->scale_out);
#endif

    // Add kernel uses pi_core_id() for parallelization - must fork to cores 0-7
    add_exec_args_t add_args = {
        .input_a = ctx->input_buffer_l2,
        .input_b = ctx->input_b_buffer_l2,
        .output = ctx->output_buffer_l2,
        .size = params->size,
        .scale_a = params->scale_a,
        .scale_b = params->scale_b,
        .scale_out = params->scale_out
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, add_exec_worker, &add_args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

/* --- Concat Execution --- */

// Worker args for concat - must be outside execute_concat (C doesn't allow nested functions)
typedef struct {
    const int8_t **inputs;
    const float *input_scales;
    int8_t *output;
    uint16_t num_inputs;
    const uint16_t *channels_per_input;
    uint16_t height;
    uint16_t width;
    float scale_output;
} concat_exec_args_t;

static void concat_exec_worker(void *arg) {
    concat_exec_args_t *a = (concat_exec_args_t *)arg;
    network_concat_int8(a->inputs, a->input_scales, a->output,
                        a->num_inputs, 1,  // batch=1
                        a->channels_per_input, a->height, a->width,
                        a->scale_output);
}

void execute_concat(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.concat) *params = &layer->params.concat;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing concat (%d inputs, h=%d, w=%d)\n",
           layer->name, params->num_inputs, params->height, params->width);
    for (int i = 0; i < ctx->concat_num_inputs && i < MAX_CONCAT_INPUTS; i++) {
        printf("CL: DEBUG concat input[%d]: ptr=%p scale=%.6f ch=%d\n",
               i, (void *)ctx->concat_inputs[i], ctx->concat_scales[i], ctx->concat_channels[i]);
        if (ctx->concat_inputs[i]) {
            printf("CL: DEBUG concat input[%d] values: %d %d %d %d %d\n", i,
                   ctx->concat_inputs[i][0], ctx->concat_inputs[i][1],
                   ctx->concat_inputs[i][2], ctx->concat_inputs[i][3],
                   ctx->concat_inputs[i][4]);
        }
    }
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Build args for worker function
    concat_exec_args_t concat_args = {
        .inputs = ctx->concat_inputs,
        .input_scales = ctx->concat_scales,
        .output = ctx->output_buffer_l2,
        .num_inputs = (uint16_t)ctx->concat_num_inputs,
        .channels_per_input = ctx->concat_channels,
        .height = (uint16_t)params->height,
        .width = (uint16_t)params->width,
        .scale_output = params->scale_output
    };

    // Fork to worker cores (network_concat_int8 uses pi_core_id() for parallelization)
    pi_cl_team_fork(NUM_CORES, concat_exec_worker, &concat_args);

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

#ifndef MINIMAL_OUTPUT
    printf("CL: DEBUG concat output: %d %d %d %d %d\n",
           ctx->output_buffer_l2[0], ctx->output_buffer_l2[1],
           ctx->output_buffer_l2[2], ctx->output_buffer_l2[3],
           ctx->output_buffer_l2[4]);
#endif
}

/* --- LayerNorm Execution --- */

void execute_layernorm(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.layernorm) *params = &layer->params.layernorm;
    const uint32_t total_elements = params->num_tokens * params->embed_dim;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing i-LayerNorm (tokens=%d, dim=%d, total=%d)\n",
           layer->name, params->num_tokens, params->embed_dim, total_elements);
    printf("CL: DEBUG layernorm: in=%p out=%p weight=%p bias=%p\n",
           (void *)ctx->input_buffer_l2, (void *)ctx->output_buffer_l2,
           (void *)ctx->layernorm_weight, (void *)ctx->layernorm_bias);
    if (ctx->input_buffer_l2) {
        printf("CL: DEBUG layernorm input: %d %d %d %d %d\n",
               ctx->input_buffer_l2[0], ctx->input_buffer_l2[1],
               ctx->input_buffer_l2[2], ctx->input_buffer_l2[3],
               ctx->input_buffer_l2[4]);
    }
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Call the parallel integer LayerNorm kernel
    network_layernorm_int8_integer_parallel(
        ctx->input_buffer_l2,
        ctx->output_buffer_l2,
        ctx->layernorm_weight,
        ctx->layernorm_bias,
        total_elements,
        params->embed_dim,
        params->scale_in,
        params->scale_out
    );

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

#ifndef MINIMAL_OUTPUT
    printf("CL: DEBUG layernorm output: %d %d %d %d %d\n",
           ctx->output_buffer_l2[0], ctx->output_buffer_l2[1],
           ctx->output_buffer_l2[2], ctx->output_buffer_l2[3],
           ctx->output_buffer_l2[4]);
#endif
}

#ifdef ARES_LLAMA_SUPPORT
/* ---
 * RMSNorm Execution (for Llama/LLMs)
 * Simpler than LayerNorm - no mean subtraction, just RMS normalization.
 * --- */

void execute_rmsnorm(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.rmsnorm) *params = &layer->params.rmsnorm;
    const uint32_t total_elements = params->num_vectors * params->normalized_dim;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing i-RMSNorm (vectors=%d, dim=%d, total=%d)\n",
           layer->name, params->num_vectors, params->normalized_dim, total_elements);
    printf("CL: DEBUG rmsnorm: in=%p out=%p weight=%p\n",
           (void *)ctx->input_buffer_l2, (void *)ctx->output_buffer_l2,
           (void *)ctx->rmsnorm_weight);
    if (ctx->input_buffer_l2) {
        printf("CL: DEBUG rmsnorm input: %d %d %d %d %d\n",
               ctx->input_buffer_l2[0], ctx->input_buffer_l2[1],
               ctx->input_buffer_l2[2], ctx->input_buffer_l2[3],
               ctx->input_buffer_l2[4]);
    }
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Call the parallel integer RMSNorm kernel
    network_rmsnorm_int8_integer_parallel(
        ctx->input_buffer_l2,
        ctx->output_buffer_l2,
        ctx->rmsnorm_weight,
        total_elements,
        params->normalized_dim,
        params->scale_in,
        params->scale_out
    );

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

#ifndef MINIMAL_OUTPUT
    printf("CL: DEBUG rmsnorm output: %d %d %d %d %d\n",
           ctx->output_buffer_l2[0], ctx->output_buffer_l2[1],
           ctx->output_buffer_l2[2], ctx->output_buffer_l2[3],
           ctx->output_buffer_l2[4]);
#endif
}
#endif // ARES_LLAMA_SUPPORT

/* --- GELU Execution --- */

void execute_gelu(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.gelu) *params = &layer->params.gelu;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing GELU (%d elements)\n",
           layer->name, params->num_elements);
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    // Use parallel LUT-based GELU (8-core fork) for large activations.
    // Falls back to single-core out-of-place for small buffers.
    if (params->num_elements >= 1024) {
        // For non-aliased buffers, copy to output first so input is preserved.
        if (ctx->output_buffer_l2 != ctx->input_buffer_l2) {
            memcpy(ctx->output_buffer_l2, ctx->input_buffer_l2,
                   params->num_elements * sizeof(int8_t));
        }
        network_gelu_int8_lut_inplace_parallel(
            ctx->output_buffer_l2,
            params->num_elements,
            params->scale_in,
            params->scale_out
        );
    } else if (ctx->input_buffer_l2 == ctx->output_buffer_l2) {
        network_gelu_int8_lut_inplace_parallel(
            ctx->input_buffer_l2,
            params->num_elements,
            params->scale_in,
            params->scale_out
        );
    } else {
        network_gelu_int8(
            ctx->input_buffer_l2,
            ctx->output_buffer_l2,
            params->num_elements,
            params->scale_in,
            params->scale_out
        );
    }
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

/* --- Mean Pool Execution --- */

void execute_mean(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.mean) *params = &layer->params.mean;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing mean pool (batch=%d, seq_len=%d, features=%d)\n",
           layer->name, params->batch, params->seq_len, params->features);
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Mean pool: [B, seq_len, features] -> [B, features]
    const int8_t *input = ctx->input_buffer_l2;
    int8_t *output = ctx->output_buffer_l2;
    const uint32_t batch = params->batch;
    const uint32_t seq_len = params->seq_len;
    const uint32_t features = params->features;
    const float scale_in = params->scale_in;
    const float scale_out = params->scale_out;

    const float scale_ratio = scale_in / scale_out;
    const int same_scale = (fabsf(scale_in - scale_out) < 1e-12f);

    for (uint32_t b = 0; b < batch; b++) {
        const int8_t *batch_in = input + b * seq_len * features;
        int8_t *batch_out = output + b * features;

        for (uint32_t f = 0; f < features; f++) {
            int32_t sum = 0;
            for (uint32_t s = 0; s < seq_len; s++) {
                sum += (int32_t)batch_in[s * features + f];
            }

            int32_t result;
            if (same_scale) {
                result = (sum + ((int32_t)seq_len >> 1)) / (int32_t)seq_len;
            } else {
                float mean_fp = (float)sum / (float)seq_len * scale_ratio;
                result = qround(mean_fp);
            }

            if (result > 127) result = 127;
            if (result < -128) result = -128;
            batch_out[f] = (int8_t)result;
        }
    }

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

#ifndef MINIMAL_OUTPUT
    printf("CL: DEBUG mean output: %d %d %d %d %d\n",
           ctx->output_buffer_l2[0], ctx->output_buffer_l2[1],
           ctx->output_buffer_l2[2], ctx->output_buffer_l2[3],
           ctx->output_buffer_l2[4]);
#endif
}

/* --- Alternating Attention Execution (Cerebro Transformer) --- */

// Worker args for alternating attention parallel operations
typedef struct {
    const int8_t *input;
    int8_t *output;
    const int8_t *weight;
    const int32_t *bias;
    int in_features;
    int out_features;
    int num_tokens;
    float scale_in;
    float scale_weight;
    float scale_out;
} alt_attn_linear_args_t;

// Parallel linear projection for alternating attention
static void alt_attn_linear_worker(void *arg) {
    alt_attn_linear_args_t *a = (alt_attn_linear_args_t *)arg;
    const int core_id = pi_core_id();
    const int tokens_per_core = (a->num_tokens + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int token_start = core_id * tokens_per_core;
    int token_end = token_start + tokens_per_core;
    if (token_end > a->num_tokens) token_end = a->num_tokens;

    const float combined_scale = a->scale_in * a->scale_weight / a->scale_out;
    const int simd_count = a->in_features >> 2;

    for (int t = token_start; t < token_end; t++) {
        const int8_t *in_row = a->input + t * a->in_features;
        int8_t *out_row = a->output + t * a->out_features;

        for (int o = 0; o < a->out_features; o++) {
            const int8_t *w_row = a->weight + o * a->in_features;
            int32_t acc = 0;

            // SIMD inner product
            const v4s *pA = (const v4s *)in_row;
            const v4s *pB = (const v4s *)w_row;
            for (int k = 0; k < simd_count; k++) {
                acc = SumDotpSS(pA[k], pB[k], acc);
            }
            // Handle remainder
            for (int k = simd_count * 4; k < a->in_features; k++) {
                acc += (int32_t)in_row[k] * (int32_t)w_row[k];
            }

            // Add bias and scale
            if (a->bias) {
                acc += a->bias[o];
            }
            float val = (float)acc * combined_scale;
            int32_t q = qround(val);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_row[o] = (int8_t)q;
        }
    }
    pi_cl_team_barrier();
}

// LUT parameters for i-softmax (must match Python and kernel_softmax.c)
#define ALT_ATTN_SOFTMAX_INPUT_MIN (-8.0f)
#define ALT_ATTN_SOFTMAX_INPUT_STEP (8.0f / 1024.0f)
#define ALT_ATTN_SOFTMAX_NUM_ENTRIES 1024
#define ALT_ATTN_SOFTMAX_OUTPUT_SCALE 32767

// Reference to the softmax LUT (defined in kernel_softmax.c)
extern const int16_t i_softmax_lut[1024];

// Softmax row computation using integer LUT (matches Python i_softmax exactly)
static void alt_attn_softmax_row(const int8_t *scores, int8_t *output, int len,
                                  float scale_in, float scale_out) {
    // Find max for numerical stability
    int8_t max_val = scores[0];
    for (int i = 1; i < len; i++) {
        if (scores[i] > max_val) max_val = scores[i];
    }

    // Compute exp values using LUT and accumulate sum
    int32_t exp_vals[256];  // Max seq_len we support
    int32_t exp_sum = 0;

    for (int i = 0; i < len; i++) {
        // Compute normalized score in FP32
        float norm_score = ((float)scores[i] - (float)max_val) * scale_in;

        // Quantize to LUT index with rounding
        int idx = (int)(norm_score - ALT_ATTN_SOFTMAX_INPUT_MIN) * (1.0f / ALT_ATTN_SOFTMAX_INPUT_STEP) + 0.5f;

        // Clip to valid range
        if (idx < 0) idx = 0;
        if (idx >= ALT_ATTN_SOFTMAX_NUM_ENTRIES) idx = ALT_ATTN_SOFTMAX_NUM_ENTRIES - 1;

        // LUT lookup
        exp_vals[i] = (int32_t)i_softmax_lut[idx];
        exp_sum += exp_vals[i];
    }

    // Normalize and quantize to INT8
    // Output scale: convert from ~[0,1] probabilities to INT8
    for (int i = 0; i < len; i++) {
        // Normalize: attn = exp_val / sum
        // Result is in [0, 1], we quantize with scale_out
        int32_t attn_int16 = (exp_vals[i] * ALT_ATTN_SOFTMAX_OUTPUT_SCALE) / exp_sum;
        float prob = (float)attn_int16 / (float)ALT_ATTN_SOFTMAX_OUTPUT_SCALE;

        int32_t q = qround(prob / scale_out);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        output[i] = (int8_t)q;
    }
}

// Parallel INT8 matrix multiply arguments
typedef struct {
    const int8_t *A;
    const int8_t *B;
    int8_t *C;
    int M, K, N;
    float combined_scale;
} alt_attn_matmul_args_t;

// Parallel worker for INT8 matrix multiply
static void alt_attn_matmul_worker(void *arg) {
    alt_attn_matmul_args_t *a = (alt_attn_matmul_args_t *)arg;
    const int core_id = pi_core_id();
    const int rows_per_core = (a->M + CL_NUM_CORES - 1) / CL_NUM_CORES;
    const int row_start = core_id * rows_per_core;
    int row_end = row_start + rows_per_core;
    if (row_end > a->M) row_end = a->M;

    const int K = a->K;
    const int N = a->N;
    const float combined_scale = a->combined_scale;
    const int simd_k = K >> 2;

    for (int m = row_start; m < row_end; m++) {
        const int8_t *a_row = a->A + m * K;
        int8_t *c_row = a->C + m * N;

        for (int n = 0; n < N; n++) {
            int32_t acc = 0;

            // SIMD inner product (B is transposed logically, so access B[k][n])
            const v4s *pA = (const v4s *)a_row;
            for (int k = 0; k < simd_k; k++) {
                // Manual unroll for B access (not contiguous)
                int32_t b0 = a->B[(k*4 + 0) * N + n];
                int32_t b1 = a->B[(k*4 + 1) * N + n];
                int32_t b2 = a->B[(k*4 + 2) * N + n];
                int32_t b3 = a->B[(k*4 + 3) * N + n];
                v4s a_vec = pA[k];
                acc += (int32_t)a_vec[0] * b0;
                acc += (int32_t)a_vec[1] * b1;
                acc += (int32_t)a_vec[2] * b2;
                acc += (int32_t)a_vec[3] * b3;
            }
            // Handle remainder
            for (int k = simd_k * 4; k < K; k++) {
                acc += (int32_t)a_row[k] * (int32_t)a->B[k * N + n];
            }

            float val = (float)acc * combined_scale;
            int32_t q = qround(val);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            c_row[n] = (int8_t)q;
        }
    }
    pi_cl_team_barrier();
}

// INT8 matrix multiply: C = A @ B with INT32 accumulation (parallelized)
static void alt_attn_matmul_int8(
    const int8_t *A, const int8_t *B, int8_t *C,
    int M, int K, int N,
    float scale_a, float scale_b, float scale_c
) {
    alt_attn_matmul_args_t args = {
        .A = A, .B = B, .C = C,
        .M = M, .K = K, .N = N,
        .combined_scale = scale_a * scale_b / scale_c
    };
    pi_cl_team_fork(CL_NUM_CORES, alt_attn_matmul_worker, &args);
}

void execute_alternating_attention(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.alternating_attention) *params = &layer->params.alternating_attention;

    const int is_channel_attn = (params->block_idx % 2 == 0);
    const char *attn_type = is_channel_attn ? "channel" : "temporal";

    // Dimensions
    const int seq_len = params->seq_len;        // = C * T
    const int embed_dim = params->embed_dim;    // = D
    const int num_heads = params->num_heads;
    const int head_dim = params->head_dim;
    const int num_channels = params->num_channels;  // C
    const int temporal_len = params->temporal_len;  // T

    // Local attention dimensions based on attention type
    int local_batch, local_seq;
    if (is_channel_attn) {
        // Channel attention: attend across channels for each time step
        // Reshape: [B, C*T, D] -> [B*T, C, D]
        local_batch = temporal_len;  // B*T batches
        local_seq = num_channels;    // C tokens per batch
    } else {
        // Temporal attention: attend across time for each channel
        // Reshape: [B, C*T, D] -> [B*C, T, D]
        local_batch = num_channels;  // B*C batches
        local_seq = temporal_len;    // T tokens per batch
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing alternating attention (%s, block=%d)\n",
           layer->name, attn_type, params->block_idx);
    printf("CL:   seq_len=%d, embed_dim=%d, heads=%d, C=%d, T=%d\n",
           seq_len, embed_dim, num_heads, num_channels, temporal_len);
    printf("CL:   local_batch=%d, local_seq=%d, head_dim=%d\n",
           local_batch, local_seq, head_dim);
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Check for weight pointers
    if (!ctx->alt_attn_qkv_weight_l2 || !ctx->alt_attn_out_weight_l2) {
        printf("CL: WARNING %s: Missing attention weights, falling back to scaled copy\n", layer->name);
        // Fallback to simple copy
        const int8_t *input = ctx->input_buffer_l2;
        int8_t *output = ctx->output_buffer_l2;
        const float scale_ratio = params->scale_in / params->scale_out;
        for (int i = 0; i < seq_len * embed_dim; i++) {
            int32_t val = (int32_t)(input[i] * scale_ratio);
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            output[i] = (int8_t)val;
        }
        goto cleanup;
    }

    // Allocate scratch buffers in L1
    // QKV projection output: seq_len * 3 * embed_dim
    // Attention scores: local_batch * num_heads * local_seq * local_seq
    // Context: local_batch * num_heads * local_seq * head_dim
    const size_t qkv_size = seq_len * 3 * embed_dim;
    const size_t scores_size = local_batch * num_heads * local_seq * local_seq;
    const size_t context_size = local_batch * num_heads * local_seq * head_dim;
    const size_t total_scratch = qkv_size + scores_size + context_size + embed_dim * seq_len;

    int8_t *scratch = NULL;
    int used_l1 = 0;

    if (ctx->l1_buffer && ctx->l1_buffer_size >= total_scratch) {
        scratch = ctx->l1_buffer;
        used_l1 = 1;
    } else {
        // Fallback to L2 allocation
        scratch = (int8_t *)pi_l2_malloc(total_scratch);
        if (!scratch) {
            printf("CL: ERROR %s: Failed to allocate scratch (%zu bytes)\n",
                   layer->name, total_scratch);
            goto cleanup;
        }
    }

    int8_t *qkv_buf = scratch;
    int8_t *scores_buf = scratch + qkv_size;
    int8_t *context_buf = scratch + qkv_size + scores_size;
    int8_t *proj_out_buf = scratch + qkv_size + scores_size + context_size;

    // Step 1: QKV Projection
    // Input: [seq_len, embed_dim] -> Output: [seq_len, 3*embed_dim]
#ifdef ARES_USE_NE16
    // Use NE16 for QKV projection if packed weights are available
    if (params->use_ne16_qkv && ctx->alt_attn_qkv_ne16_packed_l2 && ctx->alt_attn_qkv_ne16_bias_l2) {
#ifndef MINIMAL_OUTPUT
        printf("CL: %s using NE16 for QKV projection\n", layer->name);
#endif
        ne16_soft_clear_all();

        // NE16 needs scratch: input_u8 (seq_len * embed_dim) and output_s32 (tile * 3*embed_dim)
        const int tile_tokens = 16;  // Process 16 tokens at a time
        const size_t ne16_input_size = (size_t)tile_tokens * embed_dim;
        const size_t ne16_output_size = (size_t)tile_tokens * 3 * embed_dim * sizeof(int32_t);

        // Allocate NE16 scratch in L1 (after our regular scratch)
        uint8_t *ne16_input_u8 = NULL;
        int32_t *ne16_output_s32 = NULL;
        const size_t ne16_scratch_needed = ne16_input_size + ne16_output_size;
        size_t scratch_offset = total_scratch;

        if (ctx->l1_buffer && ctx->l1_buffer_size >= total_scratch + ne16_scratch_needed) {
            ne16_input_u8 = (uint8_t *)(ctx->l1_buffer + scratch_offset);
            ne16_output_s32 = (int32_t *)(ctx->l1_buffer + scratch_offset + ne16_input_size);
        } else {
            // Fallback to L2 scratch for NE16
            ne16_input_u8 = (uint8_t *)pi_l2_malloc(ne16_input_size);
            ne16_output_s32 = (int32_t *)pi_l2_malloc(ne16_output_size);
        }

        if (ne16_input_u8 && ne16_output_s32) {
            ne16_linear_int8_packed(
                ctx->input_buffer_l2,
                ctx->alt_attn_qkv_ne16_packed_l2,
                ctx->alt_attn_qkv_ne16_bias_l2,
                qkv_buf,
                seq_len,
                embed_dim,
                3 * embed_dim,
                3 * embed_dim,  // out_stride
                params->scale_in,
                params->scale_qkv_weight,
                params->scale_qkv_out,
                tile_tokens,
                ne16_input_u8,
                ne16_output_s32
            );

            // Free L2 scratch if we allocated it
            if (!ctx->l1_buffer || ctx->l1_buffer_size < total_scratch + ne16_scratch_needed) {
                if (ne16_input_u8) pi_l2_free(ne16_input_u8, ne16_input_size);
                if (ne16_output_s32) pi_l2_free(ne16_output_s32, ne16_output_size);
            }
        } else {
            // NE16 scratch allocation failed, fall back to SW
#ifndef MINIMAL_OUTPUT
            printf("CL: %s NE16 scratch alloc failed, using SW\n", layer->name);
#endif
            alt_attn_linear_args_t qkv_args = {
                .input = ctx->input_buffer_l2,
                .output = qkv_buf,
                .weight = ctx->alt_attn_qkv_weight_l2,
                .bias = ctx->alt_attn_qkv_bias_l2,
                .in_features = embed_dim,
                .out_features = 3 * embed_dim,
                .num_tokens = seq_len,
                .scale_in = params->scale_in,
                .scale_weight = params->scale_qkv_weight,
                .scale_out = params->scale_qkv_out
            };
            pi_cl_team_fork(CL_NUM_CORES, alt_attn_linear_worker, &qkv_args);
        }
    } else
#endif  // ARES_USE_NE16
    {
        // Software QKV projection
        alt_attn_linear_args_t qkv_args = {
            .input = ctx->input_buffer_l2,
            .output = qkv_buf,
            .weight = ctx->alt_attn_qkv_weight_l2,
            .bias = ctx->alt_attn_qkv_bias_l2,
            .in_features = embed_dim,
            .out_features = 3 * embed_dim,
            .num_tokens = seq_len,
            .scale_in = params->scale_in,
            .scale_weight = params->scale_qkv_weight,
            .scale_out = params->scale_qkv_out
        };
        pi_cl_team_fork(CL_NUM_CORES, alt_attn_linear_worker, &qkv_args);
    }

    // Step 2: Attention computation (Q@K^T, softmax, context)
    const float scaling_factor = params->scaling_factor;
    const float scale_q = params->scale_q;

    // Process attention for each local batch
    for (int lb = 0; lb < local_batch; lb++) {
        // Map local batch index to sequence indices
        int seq_start, seq_stride;
        if (is_channel_attn) {
            seq_start = lb;
            seq_stride = temporal_len;
        } else {
            seq_start = lb * temporal_len;
            seq_stride = 1;
        }

        for (int h = 0; h < num_heads; h++) {
            int8_t *scores = scores_buf + (lb * num_heads + h) * local_seq * local_seq;
            int8_t *ctx_out = context_buf + (lb * num_heads + h) * local_seq * head_dim;

            const int simd_head = head_dim >> 2;
            const float score_scale = params->scale_qkv_out * params->scale_qkv_out * scaling_factor / scale_q;

            // Q @ K^T with SIMD
            for (int qi = 0; qi < local_seq; qi++) {
                int q_token_idx = seq_start + qi * seq_stride;
                const int8_t *q_row = qkv_buf + q_token_idx * 3 * embed_dim + h * head_dim;
                const v4s *pQ = (const v4s *)q_row;

                for (int ki = 0; ki < local_seq; ki++) {
                    int k_token_idx = seq_start + ki * seq_stride;
                    const int8_t *k_row = qkv_buf + k_token_idx * 3 * embed_dim + embed_dim + h * head_dim;
                    const v4s *pK = (const v4s *)k_row;

                    int32_t acc = 0;
                    for (int d = 0; d < simd_head; d++) {
                        acc = SumDotpSS(pQ[d], pK[d], acc);
                    }
                    for (int d = simd_head * 4; d < head_dim; d++) {
                        acc += (int32_t)q_row[d] * (int32_t)k_row[d];
                    }

                    int32_t q_score = qround((float)acc * score_scale);
                    if (q_score > 127) q_score = 127;
                    if (q_score < -128) q_score = -128;
                    scores[qi * local_seq + ki] = (int8_t)q_score;
                }
            }

            // Softmax per row
            for (int qi = 0; qi < local_seq; qi++) {
                alt_attn_softmax_row(
                    scores + qi * local_seq,
                    scores + qi * local_seq,
                    local_seq,
                    scale_q,
                    1.0f / 128.0f
                );
            }

            // Context = Attn @ V with INT32 accumulation
            // Optimized: swap loop order for contiguous V access, use local accumulators
            const float ctx_scale = (1.0f / 128.0f) * params->scale_qkv_out / params->scale_v;
            const int simd_head4 = head_dim >> 2;

            for (int qi = 0; qi < local_seq; qi++) {
                const int8_t *attn_row = scores + qi * local_seq;
                int8_t *ctx_row = ctx_out + qi * head_dim;

                // Local accumulators (head_dim is typically 36, fits on stack)
                int32_t acc[64];  // Max head_dim we support
                for (int d = 0; d < head_dim; d++) acc[d] = 0;

                // Accumulate: loop over vi first for contiguous V access
                for (int vi = 0; vi < local_seq; vi++) {
                    int v_token_idx = seq_start + vi * seq_stride;
                    const int8_t *v_row = qkv_buf + v_token_idx * 3 * embed_dim + 2 * embed_dim + h * head_dim;
                    int32_t a_val = (int32_t)attn_row[vi];

                    // SIMD-friendly unrolled loop across head_dim
                    for (int d4 = 0; d4 < simd_head4; d4++) {
                        int d = d4 << 2;
                        acc[d]   += a_val * (int32_t)v_row[d];
                        acc[d+1] += a_val * (int32_t)v_row[d+1];
                        acc[d+2] += a_val * (int32_t)v_row[d+2];
                        acc[d+3] += a_val * (int32_t)v_row[d+3];
                    }
                    // Handle remainder
                    for (int d = simd_head4 << 2; d < head_dim; d++) {
                        acc[d] += a_val * (int32_t)v_row[d];
                    }
                }

                // Quantize all head dimensions
                for (int d = 0; d < head_dim; d++) {
                    float val = (float)acc[d] * ctx_scale;
                    int32_t q_val = qround(val);
                    if (q_val > 127) q_val = 127;
                    if (q_val < -128) q_val = -128;
                    ctx_row[d] = (int8_t)q_val;
                }
            }
        }
    }

    // Step 3: Reassemble context
    // Copy from context_buf [local_batch, num_heads, local_seq, head_dim]
    // to proj_out_buf [seq_len, embed_dim]
    for (int lb = 0; lb < local_batch; lb++) {
        int seq_start = is_channel_attn ? lb : lb * temporal_len;
        int seq_stride = is_channel_attn ? temporal_len : 1;

        for (int li = 0; li < local_seq; li++) {
            int token_idx = seq_start + li * seq_stride;
            int8_t *out_row = proj_out_buf + token_idx * embed_dim;

            // Gather from all heads
            for (int h = 0; h < num_heads; h++) {
                const int8_t *ctx_head = context_buf + (lb * num_heads + h) * local_seq * head_dim + li * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    out_row[h * head_dim + d] = ctx_head[d];
                }
            }
        }
    }

    // Step 4: Output projection
#ifdef ARES_USE_NE16
    // Use NE16 for output projection if packed weights are available
    if (params->use_ne16_out && ctx->alt_attn_out_ne16_packed_l2 && ctx->alt_attn_out_ne16_bias_l2) {
#ifndef MINIMAL_OUTPUT
        printf("CL: %s using NE16 for output projection\n", layer->name);
#endif
        ne16_soft_clear_all();

        // NE16 needs scratch: input_u8 (seq_len * embed_dim) and output_s32 (tile * embed_dim)
        const int tile_tokens = 16;
        const size_t ne16_input_size = (size_t)tile_tokens * embed_dim;
        const size_t ne16_output_size = (size_t)tile_tokens * embed_dim * sizeof(int32_t);

        // Allocate NE16 scratch in L1 (after our regular scratch)
        uint8_t *ne16_input_u8 = NULL;
        int32_t *ne16_output_s32 = NULL;
        const size_t ne16_scratch_needed = ne16_input_size + ne16_output_size;
        size_t scratch_offset = total_scratch;

        if (ctx->l1_buffer && ctx->l1_buffer_size >= total_scratch + ne16_scratch_needed) {
            ne16_input_u8 = (uint8_t *)(ctx->l1_buffer + scratch_offset);
            ne16_output_s32 = (int32_t *)(ctx->l1_buffer + scratch_offset + ne16_input_size);
        } else {
            // Fallback to L2 scratch for NE16
            ne16_input_u8 = (uint8_t *)pi_l2_malloc(ne16_input_size);
            ne16_output_s32 = (int32_t *)pi_l2_malloc(ne16_output_size);
        }

        if (ne16_input_u8 && ne16_output_s32) {
            ne16_linear_int8_packed(
                proj_out_buf,
                ctx->alt_attn_out_ne16_packed_l2,
                ctx->alt_attn_out_ne16_bias_l2,
                ctx->output_buffer_l2,
                seq_len,
                embed_dim,
                embed_dim,
                embed_dim,  // out_stride
                params->scale_v,
                params->scale_out_weight,
                params->scale_out,
                tile_tokens,
                ne16_input_u8,
                ne16_output_s32
            );

            // Free L2 scratch if we allocated it
            if (!ctx->l1_buffer || ctx->l1_buffer_size < total_scratch + ne16_scratch_needed) {
                if (ne16_input_u8) pi_l2_free(ne16_input_u8, ne16_input_size);
                if (ne16_output_s32) pi_l2_free(ne16_output_s32, ne16_output_size);
            }
        } else {
            // NE16 scratch allocation failed, fall back to SW
#ifndef MINIMAL_OUTPUT
            printf("CL: %s NE16 output scratch alloc failed, using SW\n", layer->name);
#endif
            alt_attn_linear_args_t out_args = {
                .input = proj_out_buf,
                .output = ctx->output_buffer_l2,
                .weight = ctx->alt_attn_out_weight_l2,
                .bias = ctx->alt_attn_out_bias_l2,
                .in_features = embed_dim,
                .out_features = embed_dim,
                .num_tokens = seq_len,
                .scale_in = params->scale_v,
                .scale_weight = params->scale_out_weight,
                .scale_out = params->scale_out
            };
            pi_cl_team_fork(CL_NUM_CORES, alt_attn_linear_worker, &out_args);
        }
    } else
#endif  // ARES_USE_NE16
    {
        // Software output projection
        alt_attn_linear_args_t out_args = {
            .input = proj_out_buf,
            .output = ctx->output_buffer_l2,
            .weight = ctx->alt_attn_out_weight_l2,
            .bias = ctx->alt_attn_out_bias_l2,
            .in_features = embed_dim,
            .out_features = embed_dim,
            .num_tokens = seq_len,
            .scale_in = params->scale_v,
            .scale_weight = params->scale_out_weight,
            .scale_out = params->scale_out
        };
        pi_cl_team_fork(CL_NUM_CORES, alt_attn_linear_worker, &out_args);
    }

    // Free scratch if we allocated from L2
    if (!used_l1 && scratch) {
        pi_l2_free(scratch, total_scratch);
    }

cleanup:
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
    return;
}

#ifdef ARES_LLAMA_SUPPORT
/* ---
 * SwiGLU FFN Execution (Llama-style Feed-Forward Network)
 * Formula: out = W2 @ (silu(W1 @ x) * (W3 @ x))
 * --- */

/* Worker wrappers for parallel kernels that use pi_core_id() internally.
 * These must be called via pi_cl_team_fork, not directly from Core 8. */
typedef struct {
    int8_t *buffer;
    const int8_t *lut;
    int num_elements;
} swiglu_silu_args_t;

static void swiglu_silu_worker(void *arg) {
    swiglu_silu_args_t *a = (swiglu_silu_args_t *)arg;
    network_silu_int8_lut_inplace(a->buffer, a->lut, a->num_elements);
}

typedef struct {
    const int8_t *a;
    const int8_t *b;
    int8_t *output;
    int num_elements;
    float scale_a;
    float scale_b;
    float scale_out;
} swiglu_emul_args_t;

static void swiglu_emul_worker(void *arg) {
    swiglu_emul_args_t *a = (swiglu_emul_args_t *)arg;
    network_elementwise_mul_int8(a->a, a->b, a->output, a->num_elements,
                                  a->scale_a, a->scale_b, a->scale_out);
}

void execute_swiglu_ffn(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.swiglu_ffn) *params = &layer->params.swiglu_ffn;
    const int seq_len = params->seq_len;
    const int dim = params->dim;
    const int hidden_dim = params->hidden_dim;
    const int hidden_size = seq_len * hidden_dim;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing SwiGLU FFN (seq=%d, dim=%d, hidden=%d)\n",
           layer->name, seq_len, dim, hidden_dim);
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Allocate scratch buffers for gate and up projections
    // Use provided scratch if available, otherwise allocate
    int8_t *gate_buf = NULL;
    int8_t *up_buf = NULL;
    int allocated_gate = 0, allocated_up = 0;

    if (ctx->swiglu_scratch && ctx->swiglu_scratch_size >= 2 * hidden_size) {
        gate_buf = ctx->swiglu_scratch;
        up_buf = ctx->swiglu_scratch + hidden_size;
    } else {
        gate_buf = (int8_t *)pi_l2_malloc(hidden_size);
        up_buf = (int8_t *)pi_l2_malloc(hidden_size);
        allocated_gate = 1;
        allocated_up = 1;
    }

    if (!gate_buf || !up_buf) {
        printf("CL: ERROR - SwiGLU FFN failed to allocate scratch buffers\n");
        if (allocated_gate && gate_buf) pi_l2_free(gate_buf, hidden_size);
        if (allocated_up && up_buf) pi_l2_free(up_buf, hidden_size);
        return;
    }

    // Step 1: gate = W1 @ x  [seq_len, dim] @ [dim, hidden_dim]T -> [seq_len, hidden_dim]
    network_linear_int8_parallel_tokens(
        ctx->input_buffer_l2,
        ctx->swiglu_w1_weight,
        ctx->swiglu_w1_bias,
        gate_buf,
        seq_len,
        dim,
        hidden_dim,
        params->scale_input,
        params->scale_w1,
        params->scale_hidden
    );

    // Step 2: up = W3 @ x  [seq_len, dim] @ [dim, hidden_dim]T -> [seq_len, hidden_dim]
    network_linear_int8_parallel_tokens(
        ctx->input_buffer_l2,
        ctx->swiglu_w3_weight,
        ctx->swiglu_w3_bias,
        up_buf,
        seq_len,
        dim,
        hidden_dim,
        params->scale_input,
        params->scale_w3,
        params->scale_hidden
    );

    // Step 3: Apply SiLU to gate (in-place)
    // Generate SiLU LUT on stack (256 bytes)
    int8_t silu_lut[256];
    generate_silu_lut_int8(silu_lut, params->scale_hidden, params->scale_hidden);

    {
        swiglu_silu_args_t silu_args = { .buffer = gate_buf, .lut = silu_lut, .num_elements = hidden_size };
        pi_cl_team_fork(CL_NUM_CORES, swiglu_silu_worker, &silu_args);
    }

    // Step 4: hidden = silu(gate) * up (element-wise, store in gate_buf)
    {
        swiglu_emul_args_t emul_args = {
            .a = gate_buf, .b = up_buf, .output = gate_buf,
            .num_elements = hidden_size,
            .scale_a = params->scale_hidden,
            .scale_b = params->scale_hidden,
            .scale_out = params->scale_hidden
        };
        pi_cl_team_fork(CL_NUM_CORES, swiglu_emul_worker, &emul_args);
    }

    // Step 5: out = W2 @ hidden  [seq_len, hidden_dim] @ [hidden_dim, dim]T -> [seq_len, dim]
    network_linear_int8_parallel_tokens(
        gate_buf,
        ctx->swiglu_w2_weight,
        ctx->swiglu_w2_bias,
        ctx->output_buffer_l2,
        seq_len,
        hidden_dim,
        dim,
        params->scale_hidden,
        params->scale_w2,
        params->scale_output
    );

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

    // Free scratch buffers if we allocated them
    if (allocated_gate && gate_buf) pi_l2_free(gate_buf, hidden_size);
    if (allocated_up && up_buf) pi_l2_free(up_buf, hidden_size);

#ifndef MINIMAL_OUTPUT
    printf("CL: %s SwiGLU FFN complete, output[0..4]=%d %d %d %d %d\n",
           layer->name,
           ctx->output_buffer_l2[0], ctx->output_buffer_l2[1],
           ctx->output_buffer_l2[2], ctx->output_buffer_l2[3],
           ctx->output_buffer_l2[4]);
#endif
}

/* ---
 * Llama Block Execution (Complete Transformer Decoder Block)
 * Structure: RMSNorm -> MHSA -> Add -> RMSNorm -> SwiGLU FFN -> Add
 * --- */

void execute_llama_block(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.llama_block) *params = &layer->params.llama_block;
    const int seq_len = params->seq_len;
    const int dim = params->dim;
    const int total_elements = seq_len * dim;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing Llama Block (seq=%d, dim=%d, heads=%d, kv_heads=%d)\n",
           layer->name, seq_len, dim, params->num_heads, params->n_kv_heads);
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Llama block structure (pre-norm transformer decoder block):
    // 1. norm1 = rmsnorm(x)
    // 2. attn = mhsa(norm1)      // with GQA support via n_kv_heads
    // 3. x = x + attn            // residual
    // 4. norm2 = rmsnorm(x)
    // 5. ffn = swiglu_ffn(norm2)
    // 6. x = x + ffn             // residual

    // Individual Llama components (RMSNorm, MHSA with GQA, SwiGLU FFN) are
    // dispatched as separate LayerSpec entries via the layer-by-layer executor.
    // This monolithic path performs RMSNorm and passes through.

    // Allocate scratch buffer for intermediate results
    int8_t *scratch = (int8_t *)pi_l2_malloc(total_elements);
    if (!scratch) {
        printf("CL: ERROR - Llama block failed to allocate scratch buffer\n");
        return;
    }

    // Step 1: RMSNorm (attention pre-norm)
    // norm1 = rmsnorm(input)
    network_rmsnorm_int8_integer_parallel(
        ctx->input_buffer_l2,
        scratch,
        ctx->rmsnorm_weight,  // attention_norm weight
        total_elements,
        dim,
        params->scale_input,
        params->scale_input
    );

    // Pass through: attention + FFN sub-layers handled by per-layer dispatch
    for (int i = 0; i < total_elements; i++) {
        ctx->output_buffer_l2[i] = ctx->input_buffer_l2[i];
    }

    pi_l2_free(scratch, total_elements);

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

#ifndef MINIMAL_OUTPUT
    printf("CL: %s Llama block complete\n", layer->name);
#endif
}

/* ---
 * Autoregressive MHSA Execution (Single-Token with KV Cache)
 *
 * Processes a single token through multi-head attention:
 * 1. Project Q/K/V for current token
 * 2. Apply RoPE to Q and K
 * 3. Store K/V in cache, compute attention over all cached positions
 * 4. Output projection
 * --- */

void execute_mhsa_autoregressive(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.mhsa_autoregressive) *params = &layer->params.mhsa_autoregressive;
    const int dim = params->dim;
    const int num_heads = params->num_heads;
    const int n_kv_heads = params->n_kv_heads;
    const int head_dim = params->head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int pos = ctx->kv_cache_pos;
    const int layer_idx = ctx->current_layer_idx;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing autoregressive MHSA (pos=%d, dim=%d, heads=%d, kv_heads=%d)\n",
           layer->name, pos, dim, num_heads, n_kv_heads);
#endif

    // Input: [1, dim] in ctx->input_buffer_l2
    // Weights: Q/K/V/O projection weights in L2 (DMA'd from L3)

    // Step 1: Project Q/K/V for current token
    // Q: [dim] -> [dim]  (via wq weight [dim, dim])
    // K: [dim] -> [kv_dim]  (via wk weight [kv_dim, dim])
    // V: [dim] -> [kv_dim]  (via wv weight [kv_dim, dim])
    int8_t *q_buf = ctx->mhsa_ar_q_buffer;
    int8_t *k_buf = ctx->mhsa_ar_k_buffer;
    int8_t *v_buf = ctx->mhsa_ar_v_buffer;

    // Q projection
    network_linear_int8_parallel_tokens(
        ctx->input_buffer_l2, ctx->mhsa_q_weight_l2, ctx->mhsa_q_bias_l2,
        q_buf, 1, dim, dim,
        params->scale_input, params->scale_q_weight, params->scale_q
    );

    // K projection
    network_linear_int8_parallel_tokens(
        ctx->input_buffer_l2, ctx->mhsa_k_weight_l2, ctx->mhsa_k_bias_l2,
        k_buf, 1, dim, kv_dim,
        params->scale_input, params->scale_k_weight, params->scale_k
    );

    // V projection
    network_linear_int8_parallel_tokens(
        ctx->input_buffer_l2, ctx->mhsa_v_weight_l2, ctx->mhsa_v_bias_l2,
        v_buf, 1, dim, kv_dim,
        params->scale_input, params->scale_v_weight, params->scale_v
    );

    // Step 2: Apply RoPE to Q and K
    // Q is [num_heads, 1, head_dim], K is [n_kv_heads, 1, head_dim]
    if (ctx->rope_cos_q15 && ctx->rope_sin_q15) {
        network_rope_int8_inplace_parallel(
            q_buf, ctx->rope_cos_q15, ctx->rope_sin_q15,
            num_heads, 1, head_dim, pos
        );
        network_rope_int8_inplace_parallel(
            k_buf, ctx->rope_cos_q15, ctx->rope_sin_q15,
            n_kv_heads, 1, head_dim, pos
        );
    }

    // Step 3: Store K/V in cache and compute attention
    mhsa_single_query_attention(
        q_buf,
        ctx->kv_cache_k, ctx->kv_cache_v,
        k_buf, v_buf,
        ctx->mhsa_ar_q_buffer,  // Reuse Q buffer for attention output [dim]
        ctx->mhsa_ar_scores,
        ctx->mhsa_ar_context,
        num_heads, n_kv_heads, head_dim,
        layer_idx, pos,
        ctx->kv_cache_max_seq_len, kv_dim,
        params->scale_q, params->scale_k, params->scale_v,
        params->scale_output, params->softmax_scale
    );

    // Step 4: Output projection
    // context [dim] -> output [dim] via wo weight [dim, dim]
    // Quantize context to INT8 first (already done in mhsa_single_query_attention)
    int8_t *context_int8 = ctx->mhsa_ar_q_buffer;  // Reused

    network_linear_int8_parallel_tokens(
        context_int8, ctx->mhsa_out_weight_l2, ctx->mhsa_out_bias_l2,
        ctx->output_buffer_l2, 1, dim, dim,
        params->scale_v,  // Context was quantized with V scale
        params->scale_out_weight, params->scale_output
    );

#ifndef MINIMAL_OUTPUT
    printf("CL: DEBUG mhsa_ar output: %d %d %d %d %d\n",
           ctx->output_buffer_l2[0], ctx->output_buffer_l2[1],
           ctx->output_buffer_l2[2], ctx->output_buffer_l2[3],
           ctx->output_buffer_l2[4]);
#endif
}

/* ---
 * Residual Add with FP32 Accumulation
 *
 * Dequantizes both inputs to FP32, adds, and requantizes to INT8.
 * Used in the autoregressive loop where scales may differ between branches.
 * --- */

void execute_residual_add(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.residual_add) *params = &layer->params.residual_add;
    const int size = params->size;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing residual add (size=%d, scale_a=%.6f, scale_b=%.6f)\n",
           layer->name, size, params->scale_a, params->scale_b);
#endif

    const int8_t *a = ctx->input_buffer_l2;
    const int8_t *b = ctx->residual_buffer_l2;
    int8_t *out = ctx->output_buffer_l2;

    const float sa = params->scale_a;
    const float sb = params->scale_b;
    const float inv_so = 1.0f / params->scale_out;

    for (int i = 0; i < size; i++) {
        float val = (float)a[i] * sa + (float)b[i] * sb;
        int32_t q = (int32_t)(val * inv_so + (val >= 0 ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: DEBUG residual_add output: %d %d %d %d %d\n",
           out[0], out[1], out[2], out[3], out[4]);
#endif
}

/* --- Argmax over FP32 logits (greedy sampling for text generation) --- */

int argmax_fp32(const float *logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

#endif // ARES_LLAMA_SUPPORT

/* --- Transpose 2D Execution --- */

typedef struct {
    const int8_t *input;
    int8_t *output;
    uint16_t dim1, dim2;
} transpose2d_args_t;

static void transpose2d_worker(void *arg) {
    transpose2d_args_t *a = (transpose2d_args_t *)arg;
    const int core_id = pi_core_id();
    const int chunk = (a->dim1 + NUM_CORES - 1) / NUM_CORES;
    const int start = core_id * chunk;
    const int end = (start + chunk > a->dim1) ? a->dim1 : (start + chunk);

    for (int d1 = start; d1 < end; d1++) {
        const int8_t *in_row = a->input + d1 * a->dim2;
        for (int d2 = 0; d2 < a->dim2; d2++) {
            a->output[d2 * a->dim1 + d1] = in_row[d2];
        }
    }
    pi_cl_team_barrier();
}

void execute_transpose_2d(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.transpose_2d) *params = &layer->params.transpose_2d;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing transpose_2d (%d x %d)\n",
           layer->name, params->dim0, params->dim1);
#endif

    transpose2d_args_t args = {
        .input = ctx->input_buffer_l2,
        .output = ctx->output_buffer_l2,
        .dim1 = params->dim0,
        .dim2 = params->dim1
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, transpose2d_worker, &args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

void execute_zeropad2d(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.zeropad2d) *params = &layer->params.zeropad2d;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing zeropad2d (%dx%d)->(%dx%d) pad=[%d,%d,%d,%d]%s\n",
           layer->name, params->in_h, params->in_w, params->out_h, params->out_w,
           params->pad_left, params->pad_right, params->pad_top, params->pad_bottom,
           params->layout == LAYOUT_HWC ? " [HWC]" : "");
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    if (params->layout == LAYOUT_HWC) {
        network_zeropad2d_int8_hwc(
            ctx->input_buffer_l2,
            ctx->output_buffer_l2,
            params->channels,
            params->in_h,
            params->in_w,
            params->pad_left,
            params->pad_right,
            params->pad_top,
            params->pad_bottom
        );
    } else {
        network_zeropad2d_int8(
            ctx->input_buffer_l2,
            ctx->output_buffer_l2,
            params->channels,
            params->in_h,
            params->in_w,
            params->pad_left,
            params->pad_right,
            params->pad_top,
            params->pad_bottom
        );
    }
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

/* --- Layout Conversion Operations (CHW <-> HWC) --- */

void execute_chw_to_hwc(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.layout_convert) *params = &layer->params.layout_convert;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s converting CHW->HWC (ch=%d, h=%d, w=%d)\n",
           layer->name, params->channels, params->height, params->width);
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    network_chw_to_hwc_int8(
        ctx->input_buffer_l2,
        ctx->output_buffer_l2,
        params->channels,
        params->height,
        params->width
    );
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

void execute_hwc_to_chw(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.layout_convert) *params = &layer->params.layout_convert;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s converting HWC->CHW (ch=%d, h=%d, w=%d)\n",
           layer->name, params->channels, params->height, params->width);
#endif

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    network_hwc_to_chw_int8(
        ctx->input_buffer_l2,
        ctx->output_buffer_l2,
        params->channels,
        params->height,
        params->width
    );
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

/* Executor dispatch functions for data-driven layer execution.
 *
 * Operations below are dispatched from execute_layer() based on LayerSpec type.
 * Some operations (embedding, Mamba, MHSA) are currently handled via
 * template-generated code in network.c instead of these functions. */

void execute_embedding(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.embedding) *params = &layer->params.embedding;

    // Input buffer contains a single int32 token ID
    int32_t token_id = *((int32_t *)ctx->input_buffer_l2);
    int embed_dim = params->embed_dim;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing embedding (token_id=%d, embed_dim=%d)\n",
           layer->name, token_id, embed_dim);
#endif

    if (token_id < 0 || token_id >= params->vocab_size) {
        printf("CL: ERROR: token_id %d out of range [0, %d)\n",
               token_id, params->vocab_size);
        return;
    }

    // Embedding table lives in L3. DMA a single row (embed_dim bytes) to L2.
    if (ctx->l3_weight_addr) {
        // Compute offset into L3 embedding table: token_id * embed_dim
        void *l3_row_addr = (void *)((uint8_t *)ctx->l3_weight_addr
                                     + (uint32_t)token_id * (uint32_t)embed_dim);

        // DMA from L3 to output buffer
        pi_ram_read(ctx->ram_dev, (uint32_t)l3_row_addr,
                    ctx->output_buffer_l2, embed_dim);
    } else {
        // Weights already in L2 (small model or pre-loaded)
        int8_t *weight_l2 = ctx->weight_l2;
        int offset = token_id * embed_dim;
        for (int i = 0; i < embed_dim; i++) {
            ctx->output_buffer_l2[i] = weight_l2[offset + i];
        }
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: DEBUG embedding output: %d %d %d %d %d\n",
           ctx->output_buffer_l2[0], ctx->output_buffer_l2[1],
           ctx->output_buffer_l2[2], ctx->output_buffer_l2[3],
           ctx->output_buffer_l2[4]);
#endif
}

void execute_groupnorm(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    // GroupNorm is dispatched via template-generated code in network.c.
    // This executor entry exists for dispatch table completeness.
    (void)layer; (void)ctx;
}

void execute_rfft(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    // RFFT is dispatched via template-generated code in network.c.
    (void)layer; (void)ctx;
}

// Worker args for conv1d_depthwise (kernel uses pi_core_id, needs fork)
typedef struct {
    const int8_t *input;
    const int8_t *weights;
    const int32_t *bias;
    int8_t *output;
    int channels;
    int length;
    int kernel_size;
    int causal;
    float scale_in;
    float scale_w;
    float scale_out;
} conv1d_dw_args_t;

static void conv1d_depthwise_worker(void *arg) {
    conv1d_dw_args_t *a = (conv1d_dw_args_t *)arg;
    network_conv1d_depthwise_int8(
        a->input, a->weights, a->bias, a->output,
        a->channels, a->length, a->kernel_size, a->causal,
        a->scale_in, a->scale_w, a->scale_out
    );
}

void execute_conv1d_depthwise(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.conv1d_depthwise) *params = &layer->params.conv1d_depthwise;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing conv1d_depthwise (C=%d, L=%d, K=%d, causal=%d)\n",
           layer->name, params->channels, params->length, params->kernel_size, params->causal);
#endif

    // Input: [C, L] from ctx->input_buffer_l2
    // Weights: [C, K] from ctx->weight_l2
    // Bias: [C] from ctx->bias_l2 (optional)
    // Output: [C, L] to ctx->output_buffer_l2

    // Must use pi_cl_team_fork because kernel uses pi_core_id() for parallelization
    conv1d_dw_args_t args = {
        .input = ctx->input_buffer_l2,
        .weights = ctx->weight_l2,
        .bias = (const int32_t *)ctx->bias_l2,
        .output = ctx->output_buffer_l2,
        .channels = params->channels,
        .length = params->length,
        .kernel_size = params->kernel_size,
        .causal = params->causal,
        .scale_in = params->scale_in,
        .scale_w = params->scale_w,
        .scale_out = params->scale_out
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, conv1d_depthwise_worker, &args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

// Static SiLU LUT buffer (generated once per scale combination)
static int8_t s_silu_lut[256];
static float s_silu_scale_in = 0.0f;
static float s_silu_scale_out = 0.0f;

// Worker args for silu (kernel uses pi_core_id, needs fork)
typedef struct {
    int8_t *buffer;
    const int8_t *lut;
    int num_elements;
} silu_args_t;

static void silu_worker(void *arg) {
    silu_args_t *a = (silu_args_t *)arg;
    network_silu_int8_lut_inplace(a->buffer, a->lut, a->num_elements);
}

void execute_silu(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.silu) *params = &layer->params.silu;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing silu (%d elements)\n", layer->name, params->num_elements);
#endif

    // SiLU in-place via LUT
    // LUT sources (in priority order):
    // 1. layer->params.silu.lut (pre-generated)
    // 2. ctx->weight_l2 (loaded from binary)
    // 3. Generate on-the-fly using scales

    const int8_t *lut = params->lut;
    if (!lut) {
        lut = ctx->weight_l2;
    }
    if (!lut) {
        // Generate LUT on-the-fly if scales changed
        if (s_silu_scale_in != params->scale_in || s_silu_scale_out != params->scale_out) {
#ifndef MINIMAL_OUTPUT
            printf("CL: %s generating SiLU LUT (scale_in=%f, scale_out=%f)\n",
                   layer->name, params->scale_in, params->scale_out);
#endif
            generate_silu_lut_int8(s_silu_lut, params->scale_in, params->scale_out);
            s_silu_scale_in = params->scale_in;
            s_silu_scale_out = params->scale_out;
        }
        lut = s_silu_lut;
    }

    // Must use pi_cl_team_fork because kernel uses pi_core_id() for parallelization
    silu_args_t args = {
        .buffer = ctx->input_buffer_l2,
        .lut = lut,
        .num_elements = params->num_elements
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, silu_worker, &args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif
}

void execute_ssm(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    // SSM is dispatched via template-generated code in network.c.
    (void)layer; (void)ctx;
}

void execute_mamba_block(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    // Mamba block is dispatched via template-generated code in network.c.
    (void)layer; (void)ctx;
}

void execute_mamba_wrapper(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    // Mamba wrapper is dispatched via template-generated code in network.c.
    (void)layer; (void)ctx;
}


void execute_patch_embed(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.patch_embed) *params = &layer->params.patch_embed;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing patch_embed (B=%d, [%d,%d,%d] -> [%d,%d,%d])\n",
           layer->name, params->batch, params->in_chans, params->inp_h, params->inp_w,
           params->seq_len, params->d_model, params->embed_dim);
#endif

    // PatchEmbed = Conv2D(kernel=patch_size) + Reshape + Permute
    // Input: [B, in_chans, H, W] -> Conv2D -> [B, embed_dim, grid_h, grid_w]
    //                            -> Reshape+Permute -> [B, seq_len, d_model]
    // where seq_len = grid_w, d_model = embed_dim * grid_h

    const int B = params->batch;
    const int in_chans = params->in_chans;
    const int H = params->inp_h;
    const int W = params->inp_w;
    const int patch_h = params->patch_h;
    const int patch_w = params->patch_w;
    const int stride_h = params->stride_h;
    const int stride_w = params->stride_w;
    const int embed_dim = params->embed_dim;
    const int grid_h = params->grid_h;
    const int grid_w = params->grid_w;
    const int seq_len = params->seq_len;
    const int d_model = params->d_model;

    // Conv output size = B * embed_dim * grid_h * grid_w = output_size
    const int conv_out_size = B * embed_dim * grid_h * grid_w;

    // Use mamba_scratch as intermediate buffer for conv output
    // If not available, fall back to L1 buffer
    int8_t *conv_out = NULL;
    if (ctx->mamba_scratch && ctx->mamba_scratch_size >= (size_t)conv_out_size) {
        conv_out = (int8_t *)ctx->mamba_scratch;
    } else if (ctx->l1_buffer && ctx->l1_buffer_size >= (size_t)conv_out_size) {
        conv_out = ctx->l1_buffer;
    } else {
        // Fallback: allocate temporary L2 buffer
        conv_out = (int8_t *)pi_l2_malloc(conv_out_size);
        if (!conv_out) {
            printf("CL: ERROR - %s failed to allocate conv scratch (%d bytes)\n",
                   layer->name, conv_out_size);
            return;
        }
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: [PatchEmbed] Conv2D: [%d,%d,%d,%d] -> [%d,%d,%d,%d]\n",
           B, in_chans, H, W, B, embed_dim, grid_h, grid_w);
#endif

    // Step 1: Run Conv2D kernel (patch extraction)
    conv2d_tile_args_t conv_args = {
        .tile_input_l1 = ctx->input_buffer_l2,
        .tile_output_l1 = conv_out,
        .weights_l2 = (const int8_t *)ctx->weight_l2,
        .weight_row_stride = 0,  // Default stride
        .bias_l2 = (const int32_t *)ctx->bias_l2,
        .tile_in_h = H,
        .tile_in_w = W,
        .tile_out_h = grid_h,
        .tile_out_w = grid_w,
        .in_ch = in_chans,
        .out_ch = embed_dim,
        .kernel_h = patch_h,
        .kernel_w = patch_w,
        .stride_h = stride_h,
        .stride_w = stride_w,
        .pad_h = 0,
        .pad_w = 0,
        .scale_input = params->scale_in,
        .scale_weight = params->scale_weight,
        .scale_output = params->scale_out,
        .cluster_dev = ctx->cluster_dev,
        .fusion_relu = 0,
        .fusion_quant = 0,
        .quant_scale_in = 0.0f,
        .quant_scale_out = 0.0f,
        .layout = 0  // CHW layout for patch embedding
    };

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif
    pi_cl_team_fork(NUM_CORES, conv2d_tile_worker, &conv_args);
#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

    // Step 2: Reshape [B, embed_dim, grid_h, grid_w] -> [B, grid_w, embed_dim*grid_h]
    // This is equivalent to permute(0, 3, 1, 2).reshape(B, grid_w, embed_dim*grid_h)
    // But implemented as a direct index mapping for efficiency
    //
    // Source: conv_out[b, e, h, w] at index (b*E + e)*H*W + h*W + w
    // Dest:   output[b, t, d] at index (b*T + t)*D + d
    // where t = w, d = e*grid_h + h, T = grid_w (seq_len), D = embed_dim*grid_h (d_model)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < seq_len; t++) {  // t iterates over grid_w
            for (int e = 0; e < embed_dim; e++) {
                for (int h = 0; h < grid_h; h++) {
                    // Source: [b, e, h, t] in conv_out [B, embed_dim, grid_h, grid_w]
                    const int src_idx = ((b * embed_dim + e) * grid_h + h) * grid_w + t;
                    // Dest: [b, t, d] where d = e*grid_h + h
                    const int d = e * grid_h + h;
                    const int dst_idx = (b * seq_len + t) * d_model + d;
                    ctx->output_buffer_l2[dst_idx] = conv_out[src_idx];
                }
            }
        }
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: [PatchEmbed] Output shape: [%d, %d, %d], first 8: [%d,%d,%d,%d,%d,%d,%d,%d]\n",
           B, seq_len, d_model,
           ctx->output_buffer_l2[0], ctx->output_buffer_l2[1],
           ctx->output_buffer_l2[2], ctx->output_buffer_l2[3],
           ctx->output_buffer_l2[4], ctx->output_buffer_l2[5],
           ctx->output_buffer_l2[6], ctx->output_buffer_l2[7]);
#endif

    // Free temporary buffer if we allocated it
    if (conv_out != ctx->mamba_scratch && conv_out != ctx->l1_buffer) {
        pi_l2_free(conv_out, conv_out_size);
    }
}

void execute_pos_embed(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.pos_embed) *params = &layer->params.pos_embed;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing pos_embed (B=%d, seq=%d, d_model=%d)\n",
           layer->name, params->batch, params->seq_len, params->d_model);
#endif

    // Positional embedding: in-place add of learnable position weights
    // Input/Output: [B, seq_len, d_model]
    // Pos weights: [1, seq_len, d_model] (broadcast over batch)

    const int B = params->batch;
    const int seq_len = params->seq_len;
    const int d_model = params->d_model;
    const int pos_embed_bytes = seq_len * d_model;
    const int total_elements = B * seq_len * d_model;

    const float scale_input = params->scale_input;
    const float scale_pos = params->scale_pos;
    const float scale_out = params->scale_out;

    // Get positional embedding weights
    // Priority: 1) Already in L2 (ctx->weight_l2), 2) Stream from L3 into scratch
    int8_t *pos_embed_weight = NULL;
    int allocated_pos = 0;

    if (ctx->weight_l2) {
        // Weights already in L2
        pos_embed_weight = (int8_t *)ctx->weight_l2;
    } else if (ctx->pos_embed_weight_l3 && ctx->mamba_scratch) {
        // Stream from L3 into scratch buffer
        pos_embed_weight = (int8_t *)ctx->mamba_scratch;
#ifndef MINIMAL_OUTPUT
        printf("CL: [PosEmbed] Streaming %d bytes from L3 into scratch\n", pos_embed_bytes);
#endif
#ifdef ENABLE_PERF_COUNTERS
        perf_dma_load_start();
#endif
        struct pi_device *ram_ptr = get_ram_ptr();
        pi_cl_ram_req_t req;
        pi_cl_ram_read(ram_ptr, (uint32_t)ctx->pos_embed_weight_l3, pos_embed_weight,
                       pos_embed_bytes, &req);
        pi_cl_ram_read_wait(&req);
#ifdef ENABLE_PERF_COUNTERS
        if (ctx->perf_counter) {
            ctx->perf_counter->dma_load_cycles += perf_dma_load_end();
        }
#endif
    } else {
        printf("CL: ERROR - %s: no pos_embed weights available\n", layer->name);
        return;
    }

    // Compute scale ratios
    const float ratio_input = scale_input / scale_out;
    const float ratio_pos = scale_pos / scale_out;

#ifndef MINIMAL_OUTPUT
    printf("CL: [PosEmbed] scales: in=%.6f, pos=%.6f, out=%.6f, ratios: in=%.4f, pos=%.4f\n",
           scale_input, scale_pos, scale_out, ratio_input, ratio_pos);
#endif

    // In-place addition with scale matching
#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Check if scales are equal (common fast path)
    if (fabsf(ratio_input - 1.0f) < 1e-6f && fabsf(ratio_pos - 1.0f) < 1e-6f) {
        // Fast path: scales match, simple integer addition with clipping
        for (int i = 0; i < total_elements; i++) {
            int pos_idx = i % pos_embed_bytes;  // Broadcast over batch
            int32_t sum = (int32_t)ctx->input_buffer_l2[i] + (int32_t)pos_embed_weight[pos_idx];
            // Clip to INT8 range
            ctx->input_buffer_l2[i] = (int8_t)(sum < -128 ? -128 : (sum > 127 ? 127 : sum));
        }
    } else {
        // Slow path: need to rescale before adding
        for (int i = 0; i < total_elements; i++) {
            int pos_idx = i % pos_embed_bytes;
            float val_in = (float)ctx->input_buffer_l2[i] * ratio_input;
            float val_pos = (float)pos_embed_weight[pos_idx] * ratio_pos;
            float sum = val_in + val_pos;
            int32_t rounded = (int32_t)roundf(sum);
            ctx->input_buffer_l2[i] = (int8_t)(rounded < -128 ? -128 : (rounded > 127 ? 127 : rounded));
        }
    }

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

#ifndef MINIMAL_OUTPUT
    printf("CL: [PosEmbed] Output first 8: [%d,%d,%d,%d,%d,%d,%d,%d]\n",
           ctx->input_buffer_l2[0], ctx->input_buffer_l2[1],
           ctx->input_buffer_l2[2], ctx->input_buffer_l2[3],
           ctx->input_buffer_l2[4], ctx->input_buffer_l2[5],
           ctx->input_buffer_l2[6], ctx->input_buffer_l2[7]);
#endif
}

/* --- NE16 Accelerator Execution --- */

#ifdef ARES_USE_NE16

#include "ne16/ne16_linear.h"

static inline int32_t ne16_scale_bias_wrap_i32(int32_t bias_corr, uint8_t scale)
{
    const int64_t prod = (int64_t)bias_corr * (int64_t)(uint32_t)scale;
    const uint32_t prod_u32 = (uint32_t)prod;  /* modulo 2^32 */
    int32_t out;
    memcpy(&out, &prod_u32, sizeof(out));
    return out;
}

static inline int32_t ne16_add_wrap_i32(int32_t val, uint32_t add)
{
    uint32_t v_u32;
    memcpy(&v_u32, &val, sizeof(v_u32));
    v_u32 += add;  /* modulo 2^32 */
    int32_t out;
    memcpy(&out, &v_u32, sizeof(out));
    return out;
}

void execute_linear_ne16(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.linear_ne16) *params = &layer->params.linear_ne16;

    // HW outquant (NE16 OUTQUANT) is not golden-exact vs our SW nearest-even requant.
    // We still enable it by default when compiled in, since the main motivation is throughput
    // (avoid INT32 streamout + CPU requant). For golden-exact runs, build without
    // `ARES_NE16_HW_REQUANT` or force-disable it via the runtime ctx (see below).
    int hw_requant_enabled = 0;
#ifdef ARES_NE16_HW_REQUANT
    // Enabled by default when compiled in.
    hw_requant_enabled = 1;

    // Optional runtime override:
    //   ctx->ne16_use_hw_requant < 0  => force-disable (e.g. for golden validation without rebuild)
    //   ctx->ne16_use_hw_requant > 0  => force-enable
    if (ctx->ne16_use_hw_requant < 0) {
        hw_requant_enabled = 0;
    } else if (ctx->ne16_use_hw_requant > 0) {
        hw_requant_enabled = 1;
    } else if (params->use_hw_requant != 0) {
        // Per-layer override from generated descriptors.
        hw_requant_enabled = 1;
    }
#endif
    // Get per-channel scale arrays (reference-compatible). If NULL, use scalar fallback.
    const uint8_t *hw_scale = ctx->ne16_hw_scale ? ctx->ne16_hw_scale : params->hw_scale;
    const uint8_t *hw_scale_shift = ctx->ne16_hw_scale_shift ? ctx->ne16_hw_scale_shift : params->hw_scale_shift;
#ifndef ARES_NE16_HW_REQUANT
    (void)hw_scale;
    (void)hw_scale_shift;
#endif

    /* Clear NE16 state between layers to avoid contamination */
    ne16_soft_clear_all();

    /* Optional bring-up selftest (disabled by default; enable with -DARES_NE16_SELFTEST). */
#if defined(ARES_NE16_SELFTEST) && !defined(MINIMAL_OUTPUT)
    static int selftest_done = 0;
    if (!selftest_done) {
        selftest_done = 1;
        printf("CL: === Running NE16 selftest ===\n");
        int rc = ne16_linear_selftest(ctx->l1_buffer, ctx->l1_buffer_size);
        if (rc != 0) {
            printf("CL: WARNING: NE16 selftest failed (rc=%d), continuing anyway...\n", rc);
        }
        ne16_soft_clear_all();  /* Clear state after selftest */
    }
#endif

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing linear_ne16 (batch=%d, tokens=%d, in=%d, out=%d)\n",
           layer->name, params->batch, params->num_tokens,
           params->in_features, params->out_features);
#endif

    // Validate inputs
    if (!ctx->input_buffer_l2 || !ctx->output_buffer_l2) {
        printf("CL: ERROR - %s: null input/output buffer\n", layer->name);
        return;
    }

    // Get pre-packed weights from context (runtime) or fallback to LayerSpec (compile-time)
    const int8_t *weights_packed = ctx->ne16_weights_packed ? ctx->ne16_weights_packed : params->weights_packed;
    const int32_t *bias_corrected = ctx->ne16_bias_corrected ? ctx->ne16_bias_corrected : params->bias_corrected;

    // CRITICAL: NE16 on gvsoc only reads weights correctly from L1, not L2!
    // Strategy: Use pre-packed weights from offline codegen when available (fast path).
    // Only fall back to runtime packing if pre-packed weights are NULL.
    uint8_t *runtime_packed = NULL;
    int32_t *runtime_bias_corr = NULL;
    const size_t bias_size = (size_t)params->out_features * sizeof(int32_t);
    int weights_in_l1 = 0;  // Track if we allocated in L1
    size_t allocated_weight_size = 0;  // Track allocated size for cleanup

    // Calculate packed weight size (padded to 16-byte Ki groups)
    const int nb_ki = (params->in_features + 15) / 16;
    const size_t packed_size = (size_t)params->out_features * (size_t)nb_ki * 16;

    // FAST PATH: Pre-packed weights available from offline codegen
    // Use async DMA from L2 to L1 for better performance
    int used_persistent_l1 = 0;  // Track if we used persistent buffers (no cleanup needed)
    if (weights_packed && bias_corrected) {
#ifndef MINIMAL_OUTPUT
        printf("CL: NE16 FAST PATH: Using pre-packed weights for %s\n", layer->name);
#endif
        // Try persistent L1 buffers first (no alloc/free overhead)
        if (ctx->ne16_weight_l1 && ctx->ne16_weight_l1_size >= packed_size &&
            ctx->ne16_bias_l1 && ctx->ne16_bias_l1_size >= bias_size) {
            // Use persistent L1 buffers - fastest path
            runtime_packed = ctx->ne16_weight_l1;
            runtime_bias_corr = ctx->ne16_bias_l1;
            weights_in_l1 = 1;
            used_persistent_l1 = 1;  // Don't free these later
            allocated_weight_size = 0;  // No cleanup needed

            // Use async DMA for parallel L2→L1 transfers
            pi_cl_dma_copy_t dma_weights, dma_bias;

            dma_weights.ext = (uint32_t)weights_packed;
            dma_weights.loc = (uint32_t)runtime_packed;
            dma_weights.size = packed_size;
            dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_weights.merge = 0;
            pi_cl_dma_memcpy(&dma_weights);

            dma_bias.ext = (uint32_t)bias_corrected;
            dma_bias.loc = (uint32_t)runtime_bias_corr;
            dma_bias.size = bias_size;
            dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_bias.merge = 0;
            pi_cl_dma_memcpy(&dma_bias);

            pi_cl_dma_wait(&dma_weights);
            pi_cl_dma_wait(&dma_bias);

#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 PERSISTENT L1: DMA'd %zu+%zu bytes for %s\n",
                   packed_size, bias_size, layer->name);
#endif
            weights_packed = (const int8_t *)runtime_packed;
            bias_corrected = runtime_bias_corr;
        } else {
            // Fallback: per-layer L1 allocation
            // In ARES we usually reserve most of L1 as a single arena buffer (ctx->l1_buffer).
            // In that configuration, pi_cl_l1_malloc() is both noisy (allocator dump) and
            // unlikely to succeed. Prefer the explicit tiling path which partitions ctx->l1_buffer.
            if (!ctx->l1_buffer || ctx->l1_buffer_size == 0) {
                runtime_packed = (uint8_t *)pi_cl_l1_malloc(NULL, packed_size);
                runtime_bias_corr = (int32_t *)pi_cl_l1_malloc(NULL, bias_size);
            }
            if (runtime_packed && runtime_bias_corr) {
                weights_in_l1 = 1;
                allocated_weight_size = packed_size;

                pi_cl_dma_copy_t dma_weights, dma_bias;

                dma_weights.ext = (uint32_t)weights_packed;
                dma_weights.loc = (uint32_t)runtime_packed;
                dma_weights.size = packed_size;
                dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
                dma_weights.merge = 0;
                pi_cl_dma_memcpy(&dma_weights);

                dma_bias.ext = (uint32_t)bias_corrected;
                dma_bias.loc = (uint32_t)runtime_bias_corr;
                dma_bias.size = bias_size;
                dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
                dma_bias.merge = 0;
                pi_cl_dma_memcpy(&dma_bias);

                pi_cl_dma_wait(&dma_weights);
                pi_cl_dma_wait(&dma_bias);

#ifndef MINIMAL_OUTPUT
                printf("CL: NE16 PER-LAYER L1: DMA'd %zu+%zu bytes for %s\n",
                       packed_size, bias_size, layer->name);
#endif
                weights_packed = (const int8_t *)runtime_packed;
                bias_corrected = runtime_bias_corr;
            } else {
                // L1 allocation failed - use OUTPUT CHANNEL TILING
                // Process output channels in L1-sized chunks
#ifndef MINIMAL_OUTPUT
                printf("CL: NE16 OUTPUT TILING: Weights too large for L1 (%zu bytes), using tiled execution\n", packed_size);
#endif
                if (runtime_packed) pi_cl_l1_free(NULL, runtime_packed, packed_size);
                if (runtime_bias_corr) pi_cl_l1_free(NULL, runtime_bias_corr, bias_size);

                // DOUBLE-BUFFERED OUTPUT CHANNEL TILING
                // Overlap DMA with NE16 compute for better performance
                // L1 partition: [weights_A | weights_B | bias_A | bias_B | input_scratch | output_scratch]
                const size_t weight_per_ko = (size_t)nb_ki * 16;  // Bytes per output channel
                const int tile_tokens = params->tile_tokens > 0 ? params->tile_tokens : params->num_tokens;

                // Check if we have a pre-allocated L1 buffer
                if (!ctx->l1_buffer || ctx->l1_buffer_size == 0) {
                    printf("CL: ERROR - %s: no L1 buffer provided for NE16 tiling\n", layer->name);
                    return;
                }

                // Fixed scratch sizes based on input dimensions
                const size_t input_scratch_size = (size_t)tile_tokens * (size_t)params->in_features;

                // For double-buffering, we need 2x the weight and bias buffers
                // Available L1 = 2 * tile_packed + 2 * tile_bias + input_scratch + output_scratch
                // tile needs: 2 * tile_ko * weight_per_ko + 2 * tile_ko * 4 + tile_ko * tile_tokens * 4
                const size_t bytes_per_ko_double = 2 * (weight_per_ko + sizeof(int32_t));
                const size_t scratch_per_ko = (size_t)tile_tokens * (hw_requant_enabled ? sizeof(int8_t) : sizeof(int32_t));
                const size_t outquant_per_ko = hw_requant_enabled ? 2u : 0u;  /* scale + scale_shift */
                const size_t total_per_ko = bytes_per_ko_double + scratch_per_ko + outquant_per_ko;

                // Available L1 after input scratch
                size_t available_for_tiling = ctx->l1_buffer_size - input_scratch_size;
                if (available_for_tiling <= 0 || ctx->l1_buffer_size < input_scratch_size) {
                    printf("CL: ERROR - %s: L1 buffer too small for NE16 tiling (%zu needed, %zu available)\n",
                           layer->name, input_scratch_size, ctx->l1_buffer_size);
                    return;
                }

	                // Calculate tile_ko for double-buffering
	                int tile_ko = (int)(available_for_tiling / total_per_ko);
	                if (tile_ko > params->out_features) tile_ko = params->out_features;
	                if (tile_ko < 1) tile_ko = 1;
	                // NE16 processes output channels in 32-wide subtiles. When tiling, picking a tile_ko that
	                // is a multiple of 32 avoids a partially-utilized last subtile in *every* tile (which can
	                // significantly hurt MACs/cycle when tile_ko is small).
	                if (tile_ko >= 32) {
	                    tile_ko = (tile_ko / 32) * 32;
	                    if (tile_ko < 32) tile_ko = 32;
	                }

	                // Optional: for HW outquant, a single-buffer strategy can fit a much larger tile_ko
	                // (since we don't need 2x weight buffers). A larger tile_ko often outweighs the
	                // benefit of overlapping DMA, especially for transformer MLP shapes where
	                // `weight_per_ko` is large and the double-buffered tile becomes too small.
	                if (hw_requant_enabled) {
	                    const size_t bytes_per_ko_single = (weight_per_ko + sizeof(int32_t));
	                    const size_t total_per_ko_single = bytes_per_ko_single + scratch_per_ko + outquant_per_ko;
	                    int tile_ko_single = (int)(available_for_tiling / total_per_ko_single);
	                    if (tile_ko_single > params->out_features) tile_ko_single = params->out_features;
	                    if (tile_ko_single < 1) tile_ko_single = 1;
	                    if (tile_ko_single >= 32) {
	                        tile_ko_single = (tile_ko_single / 32) * 32;
	                        if (tile_ko_single < 32) tile_ko_single = 32;
	                    }

	                    // Heuristic: if single-buffer can fit at least one extra full 32-channel subtile,
	                    // use it (fewer tiles and better NE16 utilization). Otherwise keep double-buffering.
	                    if (tile_ko_single >= tile_ko + 32) {
	                        const int full_out_features = params->out_features;
	                        const int out_stride = params->out_stride > 0 ? params->out_stride : full_out_features;
	                        const int total_tokens = params->batch * params->num_tokens;

	                        const int tile_ko_sb = tile_ko_single;
	                        const size_t tile_packed_size_sb = (size_t)tile_ko_sb * weight_per_ko;
	                        const size_t tile_bias_size_sb = (size_t)tile_ko_sb * sizeof(int32_t);
	                        const size_t output_scratch_size_sb =
	                            (size_t)tile_tokens * (size_t)tile_ko_sb * sizeof(int8_t);
	                        const size_t total_l1_needed_sb =
	                            tile_packed_size_sb + tile_bias_size_sb + input_scratch_size +
	                            output_scratch_size_sb + 2 * (size_t)tile_ko_sb;

	                        if (total_l1_needed_sb > ctx->l1_buffer_size) {
	                            // Should not happen (tile_ko_single is derived from the same budget), but keep safe.
	                            printf("CL: ERROR - %s: L1 single-buffer partitioning failed (%zu > %zu)\n",
	                                   layer->name, total_l1_needed_sb, ctx->l1_buffer_size);
	                            return;
	                        }

	                        // Layout (single-buffer): [weights | bias | input_u8 | out_s8_tile | scale | shift]
	                        uint8_t *tile_weights = (uint8_t *)ctx->l1_buffer;
	                        int32_t *tile_bias = (int32_t *)(tile_weights + tile_packed_size_sb);
	                        uint8_t *input_u8_scratch = (uint8_t *)tile_bias + tile_bias_size_sb;
	                        int8_t *output_s8_scratch = (int8_t *)(input_u8_scratch + input_scratch_size);
	                        uint8_t *outquant_scale = (uint8_t *)(output_s8_scratch + output_scratch_size_sb);
	                        uint8_t *outquant_scale_shift = outquant_scale + tile_ko_sb;

	                        uint8_t outquant_qbias = 0;
	                        uint8_t outquant_qnorm = 0;
	                        const float combined_scale = params->scale_input * params->scale_weight / params->scale_output;
	                        ne16_compute_outquant_scale(combined_scale, &outquant_qbias, &outquant_qnorm);

	                        const int num_tiles = (full_out_features + tile_ko_sb - 1) / tile_ko_sb;

#ifndef MINIMAL_OUTPUT
	                        printf("CL: NE16 SINGLE-BUFFER: tile_ko=%d, %d tiles, L1: w=%zu b=%zu in=%zu out=%zu (total=%zu/%zu)\n",
	                               tile_ko_sb, num_tiles, tile_packed_size_sb, tile_bias_size_sb,
	                               input_scratch_size, output_scratch_size_sb, total_l1_needed_sb, ctx->l1_buffer_size);
#endif

#ifdef ENABLE_PERF_COUNTERS
	                        perf_compute_start();
#endif

	                        for (int ko_start = 0; ko_start < full_out_features; ko_start += tile_ko_sb) {
	                            const int current_tile_ko =
	                                (ko_start + tile_ko_sb > full_out_features) ? (full_out_features - ko_start) : tile_ko_sb;
	                            const size_t current_weight_size = (size_t)current_tile_ko * weight_per_ko;
	                            const size_t current_bias_size = (size_t)current_tile_ko * sizeof(int32_t);

	                            pi_cl_dma_copy_t dma_w, dma_b;
	                            dma_w.ext = (uint32_t)(weights_packed + (size_t)ko_start * weight_per_ko);
	                            dma_w.loc = (uint32_t)tile_weights;
	                            dma_w.size = current_weight_size;
	                            dma_w.dir = PI_CL_DMA_DIR_EXT2LOC;
	                            dma_w.merge = 0;
	                            pi_cl_dma_memcpy(&dma_w);

	                            dma_b.ext = (uint32_t)(bias_corrected + ko_start);
	                            dma_b.loc = (uint32_t)tile_bias;
	                            dma_b.size = current_bias_size;
	                            dma_b.dir = PI_CL_DMA_DIR_EXT2LOC;
	                            dma_b.merge = 0;
	                            pi_cl_dma_memcpy(&dma_b);

	                            pi_cl_dma_wait(&dma_w);
	                            pi_cl_dma_wait(&dma_b);

	                            // Prepare outquant params for this tile
	                            if (hw_scale && hw_scale_shift) {
	                                memcpy(outquant_scale, hw_scale + ko_start, (size_t)current_tile_ko);
	                                memcpy(outquant_scale_shift, hw_scale_shift + ko_start, (size_t)current_tile_ko);
	                            } else {
	                                memset(outquant_scale, outquant_qbias, (size_t)current_tile_ko);
	                                memset(outquant_scale_shift, outquant_qnorm, (size_t)current_tile_ko);
	                            }

	                            // Convert bias into NE16 ScaleBias domain in-place (tile-local buffer).
	                            for (int i = 0; i < current_tile_ko; i++) {
	                                tile_bias[i] = ne16_scale_bias_wrap_i32(tile_bias[i], outquant_scale[i]);
	                                const uint8_t sh = outquant_scale_shift[i];
	                                if (sh > 0 && sh < 32) {
	                                    tile_bias[i] = ne16_add_wrap_i32(tile_bias[i], 1u << (sh - 1));
	                                }
	                            }
	                            asm volatile("fence iorw, iorw" ::: "memory");

	                            int8_t *tile_output = ctx->output_buffer_l2 + ko_start;
	                            ne16_linear_int8_packed_hw_requant(
	                                ctx->input_buffer_l2,
	                                (const int8_t *)tile_weights,
	                                tile_bias,
	                                outquant_scale,
	                                outquant_scale_shift,
	                                tile_output,
	                                total_tokens,
	                                params->in_features,
	                                current_tile_ko,
	                                out_stride,
	                                tile_tokens,
	                                input_u8_scratch,
	                                /*input_u8_pong=*/NULL,
	                                output_s8_scratch
	                            );
	                        }

#ifdef ENABLE_PERF_COUNTERS
	                        if (ctx->perf_counter) {
	                            ctx->perf_counter->compute_cycles += perf_compute_end();
	                        }
#endif
	                        return;  // Early return - tiled execution complete
	                    }
	                }

	                const size_t tile_packed_size = (size_t)tile_ko * weight_per_ko;
	                const size_t tile_bias_size = (size_t)tile_ko * sizeof(int32_t);
	                const size_t output_scratch_size = (size_t)tile_tokens * (size_t)tile_ko *
	                                                   (hw_requant_enabled ? sizeof(int8_t) : sizeof(int32_t));

                // Total L1 needed with double-buffering
	                const size_t total_l1_needed = 2 * tile_packed_size + 2 * tile_bias_size +
	                                                input_scratch_size + output_scratch_size +
	                                                (hw_requant_enabled ? 2 * (size_t)tile_ko : 0);
                if (total_l1_needed > ctx->l1_buffer_size) {
                    printf("CL: ERROR - %s: L1 buffer partitioning failed (%zu > %zu)\n",
                           layer->name, total_l1_needed, ctx->l1_buffer_size);
                    return;
                }

                // Partition L1 buffer for double-buffering
                // Layout: [weights_A | weights_B | bias_A | bias_B | input_scratch | output_scratch]
                uint8_t *tile_weights_A = (uint8_t *)ctx->l1_buffer;
                uint8_t *tile_weights_B = tile_weights_A + tile_packed_size;
                int32_t *tile_bias_A = (int32_t *)(tile_weights_B + tile_packed_size);
                int32_t *tile_bias_B = (int32_t *)((uint8_t *)tile_bias_A + tile_bias_size);
	                uint8_t *input_u8_scratch = (uint8_t *)tile_bias_B + tile_bias_size;
	                int32_t *output_s32_scratch = NULL;
	                int8_t *output_s8_scratch = NULL;
	                uint8_t *outquant_scale = NULL;
	                uint8_t *outquant_scale_shift = NULL;
	                if (hw_requant_enabled) {
	                    output_s8_scratch = (int8_t *)(input_u8_scratch + input_scratch_size);
	                    outquant_scale = (uint8_t *)(output_s8_scratch + output_scratch_size);
	                    outquant_scale_shift = outquant_scale + tile_ko;
	                } else {
	                    output_s32_scratch = (int32_t *)(input_u8_scratch + input_scratch_size);
	                }

	                const int total_tokens = params->batch * params->num_tokens;
	                const int full_out_features = params->out_features;
	                const int out_stride = params->out_stride > 0 ? params->out_stride : full_out_features;
	                uint8_t outquant_qbias = 0;
	                uint8_t outquant_qnorm = 0;
	                if (hw_requant_enabled) {
	                    const float combined_scale = params->scale_input * params->scale_weight / params->scale_output;
	                    ne16_compute_outquant_scale(combined_scale, &outquant_qbias, &outquant_qnorm);
	                }
	                const int num_tiles = (full_out_features + tile_ko - 1) / tile_ko;

#ifndef MINIMAL_OUTPUT
	                printf("CL: NE16 DOUBLE-BUFFERED: tile_ko=%d, %d tiles, L1: 2*w=%zu 2*b=%zu in=%zu out=%zu (total=%zu/%zu)\n",
	                       tile_ko, num_tiles, 2*tile_packed_size, 2*tile_bias_size,
	                       input_scratch_size, output_scratch_size, total_l1_needed, ctx->l1_buffer_size);
#endif

#ifdef ENABLE_PERF_COUNTERS
	                perf_compute_start();
#endif

	                // Double-buffered pipelined execution
	                pi_cl_dma_copy_t dma_w_active, dma_b_active;
	                pi_cl_dma_copy_t dma_w_next, dma_b_next;
	                int active_buf = 0;  // 0 = A, 1 = B

                // PROLOGUE: Start DMA for first tile
                int ko_start = 0;
                int current_tile_ko = (ko_start + tile_ko > full_out_features) ?
                                       (full_out_features - ko_start) : tile_ko;
                size_t current_weight_size = (size_t)current_tile_ko * weight_per_ko;
                size_t current_bias_size = (size_t)current_tile_ko * sizeof(int32_t);

                dma_w_active.ext = (uint32_t)(weights_packed);
                dma_w_active.loc = (uint32_t)tile_weights_A;
                dma_w_active.size = current_weight_size;
                dma_w_active.dir = PI_CL_DMA_DIR_EXT2LOC;
                dma_w_active.merge = 0;
                pi_cl_dma_memcpy(&dma_w_active);

                dma_b_active.ext = (uint32_t)(bias_corrected);
                dma_b_active.loc = (uint32_t)tile_bias_A;
                dma_b_active.size = current_bias_size;
                dma_b_active.dir = PI_CL_DMA_DIR_EXT2LOC;
                dma_b_active.merge = 0;
                pi_cl_dma_memcpy(&dma_b_active);

                // Main pipelined loop
                for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                    ko_start = tile_idx * tile_ko;
                    current_tile_ko = (ko_start + tile_ko > full_out_features) ?
                                       (full_out_features - ko_start) : tile_ko;
                    current_weight_size = (size_t)current_tile_ko * weight_per_ko;
                    current_bias_size = (size_t)current_tile_ko * sizeof(int32_t);

                    // Wait for current tile's DMA to complete
                    pi_cl_dma_wait(&dma_w_active);
                    pi_cl_dma_wait(&dma_b_active);

                    // Start DMA for NEXT tile (overlaps with current tile's compute)
                    int has_next = (tile_idx + 1) < num_tiles;
                    if (has_next) {
                        int next_ko_start = (tile_idx + 1) * tile_ko;
                        int next_tile_ko = (next_ko_start + tile_ko > full_out_features) ?
                                            (full_out_features - next_ko_start) : tile_ko;
                        size_t next_weight_offset = (size_t)next_ko_start * weight_per_ko;
                        size_t next_weight_size = (size_t)next_tile_ko * weight_per_ko;
                        size_t next_bias_size = (size_t)next_tile_ko * sizeof(int32_t);

                        // DMA to the OTHER buffer (ping-pong)
                        uint8_t *next_weights = (active_buf == 0) ? tile_weights_B : tile_weights_A;
                        int32_t *next_bias = (active_buf == 0) ? tile_bias_B : tile_bias_A;

                        dma_w_next.ext = (uint32_t)(weights_packed + next_weight_offset);
                        dma_w_next.loc = (uint32_t)next_weights;
                        dma_w_next.size = next_weight_size;
                        dma_w_next.dir = PI_CL_DMA_DIR_EXT2LOC;
                        dma_w_next.merge = 0;
                        pi_cl_dma_memcpy(&dma_w_next);

                        dma_b_next.ext = (uint32_t)(bias_corrected + next_ko_start);
                        dma_b_next.loc = (uint32_t)next_bias;
                        dma_b_next.size = next_bias_size;
                        dma_b_next.dir = PI_CL_DMA_DIR_EXT2LOC;
                        dma_b_next.merge = 0;
                        pi_cl_dma_memcpy(&dma_b_next);
                    }

                    // COMPUTE current tile (while next tile's DMA runs in background)
	                    uint8_t *current_weights = (active_buf == 0) ? tile_weights_A : tile_weights_B;
	                    int32_t *current_bias = (active_buf == 0) ? tile_bias_A : tile_bias_B;
	                    int8_t *tile_output = ctx->output_buffer_l2 + ko_start;

                    if (hw_requant_enabled) {
                        // Use per-channel scale arrays if available, otherwise scalar fallback
                        if (hw_scale && hw_scale_shift) {
                            memcpy(outquant_scale, hw_scale + ko_start, (size_t)current_tile_ko);
                            memcpy(outquant_scale_shift, hw_scale_shift + ko_start, (size_t)current_tile_ko);
                        } else {
                            memset(outquant_scale, outquant_qbias, (size_t)current_tile_ko);
                            memset(outquant_scale_shift, outquant_qnorm, (size_t)current_tile_ko);
                        }

                        // IMPORTANT: NE16 norm/quant adds `ScaleBias` after applying the scale multiplier.
                        // To match our standard `((acc + bias_corr) * scale) >> shift` formulation, we must
                        // pre-scale the bias in-place: `scale_bias = bias_corr * scale`.
                        for (int i = 0; i < current_tile_ko; i++) {
                            current_bias[i] = ne16_scale_bias_wrap_i32(current_bias[i], outquant_scale[i]);
                            const uint8_t sh = outquant_scale_shift[i];
                            if (sh > 0 && sh < 32) {
                                // Reference rounding: (x + 2^(sh-1)) >> sh
                                current_bias[i] = ne16_add_wrap_i32(current_bias[i], 1u << (sh - 1));
                            }
                        }
                        asm volatile("fence iorw, iorw" ::: "memory");

	                        ne16_linear_int8_packed_hw_requant(
	                            ctx->input_buffer_l2,
		                            (const int8_t *)current_weights,
		                            current_bias,
		                            outquant_scale,
		                            outquant_scale_shift,
		                            tile_output,
		                            total_tokens,
		                            params->in_features,
		                            current_tile_ko,
		                            out_stride,
		                            tile_tokens,
		                            input_u8_scratch,
		                            /*input_u8_pong=*/NULL,
		                            output_s8_scratch
		                        );
	                    } else {
	                        ne16_linear_int8_packed(
	                            ctx->input_buffer_l2,
	                            (const int8_t *)current_weights,
	                            current_bias,
	                            tile_output,
	                            total_tokens,
	                            params->in_features,
	                            current_tile_ko,
	                            out_stride,
	                            params->scale_input,
	                            params->scale_weight,
	                            params->scale_output,
	                            tile_tokens,
	                            input_u8_scratch,
	                            output_s32_scratch
	                        );
	                    }

                    // Swap buffers for next iteration
                    if (has_next) {
                        active_buf = 1 - active_buf;
                        dma_w_active = dma_w_next;
                        dma_b_active = dma_b_next;
                    }
                }

#ifndef MINIMAL_OUTPUT
	                printf("CL: %s NE16 double-buffered tiling complete\n", layer->name);
#endif

#ifdef ENABLE_PERF_COUNTERS
	                if (ctx->perf_counter) {
	                    ctx->perf_counter->compute_cycles += perf_compute_end();
	                }
#endif
	                return;  // Early return - tiled execution complete
	            }
	        }
	    }
    // SLOW PATH: No pre-packed weights - pack at runtime from original weights
    else if (ctx->weight_l2 && ctx->bias_l2) {
#ifndef MINIMAL_OUTPUT
        printf("CL: NE16 SLOW PATH: Runtime packing for %s (consider generating pre-packed weights)\n", layer->name);
#endif
        // Forward declaration for in-place packing function
        extern void ne16_pack_weights_inplace_with_bias(
            int8_t *weights, int32_t *bias_corr, const int32_t *bias,
            int in_features, int out_features);

        const size_t weight_size = (size_t)params->out_features * (size_t)params->in_features;

        // CRITICAL: Allocate weights in L1 (NE16 only reads correctly from L1 on gvsoc)
        runtime_packed = (uint8_t *)pi_cl_l1_malloc(NULL, weight_size);
        runtime_bias_corr = (int32_t *)pi_cl_l1_malloc(NULL, bias_size);
        if (runtime_packed && runtime_bias_corr) {
            weights_in_l1 = 1;
            allocated_weight_size = weight_size;
            memcpy(runtime_packed, ctx->weight_l2, weight_size);
            asm volatile("fence" ::: "memory");

            // In-place packing with bias correction
            ne16_pack_weights_inplace_with_bias(
                (int8_t *)runtime_packed, runtime_bias_corr, (const int32_t *)ctx->bias_l2,
                params->in_features, params->out_features);
            asm volatile("fence" ::: "memory");

#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 SLOW: Runtime packed %zu bytes in L1 for %s\n", weight_size, layer->name);
#endif
            weights_packed = (const int8_t *)runtime_packed;
            bias_corrected = runtime_bias_corr;
        } else {
            // L1 allocation failed, fall back to L2
#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 WARNING: L1 alloc failed for runtime packing, using L2\n");
#endif
            if (runtime_packed) pi_cl_l1_free(NULL, runtime_packed, weight_size);
            if (runtime_bias_corr) pi_cl_l1_free(NULL, runtime_bias_corr, bias_size);
            runtime_packed = (uint8_t *)pi_l2_malloc(weight_size);
            runtime_bias_corr = (int32_t *)pi_l2_malloc(bias_size);
            if (runtime_packed && runtime_bias_corr) {
                allocated_weight_size = weight_size;
                memcpy(runtime_packed, ctx->weight_l2, weight_size);
                asm volatile("fence" ::: "memory");
                ne16_pack_weights_inplace_with_bias(
                    (int8_t *)runtime_packed, runtime_bias_corr, (const int32_t *)ctx->bias_l2,
                    params->in_features, params->out_features);
                asm volatile("fence" ::: "memory");
#ifndef MINIMAL_OUTPUT
                printf("CL: NE16 SLOW: Runtime packed %zu bytes in L2 for %s\n", weight_size, layer->name);
#endif
                weights_packed = (const int8_t *)runtime_packed;
                bias_corrected = runtime_bias_corr;
            }
        }
    }

    uint8_t *copied_weights = used_persistent_l1 ? NULL : runtime_packed;  // Skip cleanup for persistent
    int32_t *copied_bias_corr = used_persistent_l1 ? NULL : runtime_bias_corr;  // Skip cleanup for persistent
    int copied_weights_in_l1 = weights_in_l1;  // Track for cleanup
    size_t copied_weights_size = allocated_weight_size;  // Track allocated size for correct cleanup

    if (!weights_packed || !bias_corrected) {
        printf("CL: ERROR - %s: null NE16 packed weights or bias\n", layer->name);
        return;
    }

    // Allocate scratch buffers for NE16 execution.
    //
    // Default (hw_requant_enabled=0): SW requantization path
    //   Layout (pipelined): [input_u8_ping | input_u8_pong | output_s32_scratch]
    //
    // Optional (hw_requant_enabled=1): NE16 OUTQUANT path
    //   Layout (pipelined): [input_u8_ping | input_u8_pong | scale | scale_shift]
    //   Layout (serial):    [input_u8 | (optional output_s8_tile) | scale | scale_shift]

    const int tile_tokens = params->tile_tokens > 0 ? params->tile_tokens : params->num_tokens;
    const int total_tokens = params->batch * params->num_tokens;
    const int out_stride = params->out_stride > 0 ? params->out_stride : params->out_features;

    const size_t single_input_size = (size_t)tile_tokens * (size_t)params->in_features;
    const size_t double_input_size = 2 * single_input_size;  // Ping + pong

    uint8_t *input_u8_ping = NULL;
    uint8_t *input_u8_pong = NULL;
    int32_t *output_s32_scratch = NULL;         /* SW requant path */
    int8_t *output_s8_tile_scratch = NULL;      /* HW requant path, only when out_stride != out_features */
    uint8_t *outquant_scale = NULL;             /* HW requant path */
    uint8_t *outquant_scale_shift = NULL;       /* HW requant path */
    int32_t *outquant_bias = NULL;              /* HW requant path: ScaleBias in NE16 domain (int32) */

    size_t output_scratch_size = 0;
    size_t output_s8_scratch_size = 0;
    size_t outquant_size = 0;
    size_t outquant_bias_size = 0;

    // Try pipelined execution first (requires more scratch but faster)
    int use_pipelined = 0;
    int used_l1_scratch = 0;

alloc_scratch:
    // Reset scratch pointers/sizes in case we fall back from HW → SW allocation.
    input_u8_ping = NULL;
    input_u8_pong = NULL;
    output_s32_scratch = NULL;
    output_s8_tile_scratch = NULL;
    outquant_scale = NULL;
    outquant_scale_shift = NULL;
    outquant_bias = NULL;
    output_scratch_size = 0;
    output_s8_scratch_size = 0;
    outquant_size = 0;
    outquant_bias_size = 0;
    use_pipelined = 0;
    used_l1_scratch = 0;

    if (!hw_requant_enabled) {
        /* ===== SW requantization path (unchanged) ===== */
        output_scratch_size = (size_t)tile_tokens * (size_t)params->out_features * sizeof(int32_t);
        const size_t pipelined_total_size = double_input_size + output_scratch_size;
        const size_t serial_total_size = single_input_size + output_scratch_size;

        if (ctx->l1_buffer && ctx->l1_buffer_size >= pipelined_total_size) {
            input_u8_ping = (uint8_t *)ctx->l1_buffer;
            input_u8_pong = input_u8_ping + single_input_size;
            output_s32_scratch = (int32_t *)(ctx->l1_buffer + double_input_size);
            used_l1_scratch = 1;
            use_pipelined = 1;
#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 PIPELINED using L1 scratch: ping=%p pong=%p acc=%p (total=%zu)\n",
                   (void*)input_u8_ping, (void*)input_u8_pong, (void*)output_s32_scratch, pipelined_total_size);
#endif
        } else if (ctx->l1_buffer && ctx->l1_buffer_size >= serial_total_size) {
            input_u8_ping = (uint8_t *)ctx->l1_buffer;
            input_u8_pong = NULL;
            output_s32_scratch = (int32_t *)(ctx->l1_buffer + single_input_size);
            used_l1_scratch = 1;
            use_pipelined = 0;
#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 SERIAL using L1 scratch: input_u8=%p acc_s32=%p (pipelining disabled)\n",
                   (void*)input_u8_ping, (void*)output_s32_scratch);
#endif
        } else {
            input_u8_ping = (uint8_t *)pi_l2_malloc(single_input_size);
            input_u8_pong = (uint8_t *)pi_l2_malloc(single_input_size);
            output_s32_scratch = (int32_t *)pi_l2_malloc(output_scratch_size);

            if (input_u8_ping && input_u8_pong && output_s32_scratch) {
                use_pipelined = 1;
#ifndef MINIMAL_OUTPUT
                printf("CL: NE16 PIPELINED using L2 scratch: ping=%p pong=%p acc=%p\n",
                       (void*)input_u8_ping, (void*)input_u8_pong, (void*)output_s32_scratch);
#endif
            } else if (input_u8_ping && output_s32_scratch) {
                if (input_u8_pong) pi_l2_free(input_u8_pong, single_input_size);
                input_u8_pong = NULL;
                use_pipelined = 0;
#ifndef MINIMAL_OUTPUT
                printf("CL: NE16 SERIAL using L2 scratch (pong alloc failed)\n");
#endif
            } else {
                printf("CL: ERROR - %s: failed to allocate NE16 scratch buffers\n", layer->name);
                if (input_u8_ping) pi_l2_free(input_u8_ping, single_input_size);
                if (input_u8_pong) pi_l2_free(input_u8_pong, single_input_size);
                if (output_s32_scratch) pi_l2_free(output_s32_scratch, output_scratch_size);
                return;
            }
        }
    } else {
        /* ===== HW requantization (NE16 OUTQUANT) ===== */
        outquant_size = (size_t)params->out_features;
        outquant_bias_size = outquant_size * sizeof(int32_t);
        // GVSOC models NE16 streamout as L1-only, so we always stream into an L1 tile buffer and then
        // scatter/copy into L2. This also matches the common HW usage pattern for cluster peripherals.
        output_s8_scratch_size = (size_t)tile_tokens * (size_t)params->out_features;

        const size_t pipelined_total_size =
            double_input_size + output_s8_scratch_size + 2 * outquant_size + outquant_bias_size;
        const size_t serial_total_size =
            single_input_size + output_s8_scratch_size + 2 * outquant_size + outquant_bias_size;

        if (ctx->l1_buffer && ctx->l1_buffer_size >= pipelined_total_size) {
            input_u8_ping = (uint8_t *)ctx->l1_buffer;
            input_u8_pong = input_u8_ping + single_input_size;
            uint8_t *cursor = (uint8_t *)ctx->l1_buffer + double_input_size;
            output_s8_tile_scratch = (int8_t *)cursor;
            cursor += output_s8_scratch_size;
            outquant_scale = cursor;
            cursor += outquant_size;
            outquant_scale_shift = cursor;
            cursor += outquant_size;

            uintptr_t c = (uintptr_t)cursor;
            c = (c + 3u) & ~(uintptr_t)3u;
            outquant_bias = (int32_t *)c;
            used_l1_scratch = 1;
            use_pipelined = 1;
#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 HW-REQUANT PIPELINED using L1 scratch: ping=%p pong=%p scale=%p shift=%p (total=%zu)\n",
                   (void*)input_u8_ping, (void*)input_u8_pong,
                   (void*)outquant_scale, (void*)outquant_scale_shift, pipelined_total_size);
#endif
        } else if (ctx->l1_buffer && ctx->l1_buffer_size >= serial_total_size) {
            input_u8_ping = (uint8_t *)ctx->l1_buffer;
            input_u8_pong = NULL;

            uint8_t *cursor = (uint8_t *)ctx->l1_buffer + single_input_size;
            output_s8_tile_scratch = (int8_t *)cursor;
            cursor += output_s8_scratch_size;
            outquant_scale = cursor;
            cursor += outquant_size;
            outquant_scale_shift = cursor;
            cursor += outquant_size;

            uintptr_t c = (uintptr_t)cursor;
            c = (c + 3u) & ~(uintptr_t)3u;
            outquant_bias = (int32_t *)c;

            used_l1_scratch = 1;
            use_pipelined = 0;
#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 HW-REQUANT SERIAL using L1 scratch: input_u8=%p out_s8_tile=%p scale=%p shift=%p\n",
                   (void*)input_u8_ping, (void*)output_s8_tile_scratch,
                   (void*)outquant_scale, (void*)outquant_scale_shift);
#endif
        } else {
            // HW outquant needs L1-resident scale/shift/bias and (on gvsoc) L1 streamout.
            // If we can't fit in the provided L1 scratch, fall back to the golden-exact SW path.
#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 HW-REQUANT: insufficient L1 scratch (%zu bytes needed, have %zu) - falling back to SW requant\n",
                   serial_total_size, ctx->l1_buffer_size);
#endif
            hw_requant_enabled = 0;
            goto alloc_scratch;
        }

        if (hw_requant_enabled) {
            /* Fill outquant scale arrays: use per-channel if available, else scalar fallback */
            if (hw_scale && hw_scale_shift) {
                /* Per-channel scale arrays from codegen (reference-compatible) */
                memcpy(outquant_scale, hw_scale, outquant_size);
                memcpy(outquant_scale_shift, hw_scale_shift, outquant_size);
            } else {
                /* Fallback: compute single scalar scale and broadcast */
                uint8_t qbias = 0;
                uint8_t qnorm = 0;
                const float combined_scale = params->scale_input * params->scale_weight / params->scale_output;
                ne16_compute_outquant_scale(combined_scale, &qbias, &qnorm);
                memset(outquant_scale, qbias, outquant_size);
                memset(outquant_scale_shift, qnorm, outquant_size);
            }

            // IMPORTANT: NE16 norm/quant expects `ScaleBias` in the scaled domain (added after multiply).
            // Our offline bias is `bias_corr` (to be added before multiply). Convert into a dedicated buffer:
            //   scale_bias[o] = bias_corr[o] * qbias[o]  (+ rounding offset)
            if (!outquant_bias) {
                printf("CL: ERROR - %s: missing outquant_bias scratch for HW requant\n", layer->name);
                return;
            }
            for (size_t i = 0; i < outquant_size; i++) {
                int32_t b = ne16_scale_bias_wrap_i32(bias_corrected[i], outquant_scale[i]);
                const uint8_t sh = outquant_scale_shift[i];
                if (sh > 0 && sh < 32) {
                    // Reference rounding: (x + 2^(sh-1)) >> sh
                    b = ne16_add_wrap_i32(b, 1u << (sh - 1));
                }
                outquant_bias[i] = b;
            }
            asm volatile("fence iorw, iorw" ::: "memory");
        }
    }

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    // Execute NE16 linear layer
	    if (hw_requant_enabled) {
	        ne16_linear_int8_packed_hw_requant(
	            ctx->input_buffer_l2,
	            weights_packed,
	            outquant_bias,
	            outquant_scale,
	            outquant_scale_shift,
	            ctx->output_buffer_l2,
	            total_tokens,
	            params->in_features,
	            params->out_features,
	            out_stride,
	            tile_tokens,
	            input_u8_ping,
	            input_u8_pong,
	            output_s8_tile_scratch
	        );
    } else if (use_pipelined && input_u8_pong) {
        ne16_linear_int8_pipelined(
            ctx->input_buffer_l2,
            weights_packed,
            bias_corrected,
            /*scale=*/NULL,
            /*scale_shift=*/NULL,
            ctx->output_buffer_l2,
            total_tokens,
            params->in_features,
            params->out_features,
            out_stride,
            params->scale_input,
            params->scale_weight,
            params->scale_output,
            tile_tokens,
            input_u8_ping,
            input_u8_pong,
            output_s32_scratch,
            /*use_hw_requant=*/0
        );
    } else {
        ne16_linear_int8_packed(
            ctx->input_buffer_l2,
            weights_packed,
            bias_corrected,
            ctx->output_buffer_l2,
            total_tokens,
            params->in_features,
            params->out_features,
            out_stride,
            params->scale_input,
            params->scale_weight,
            params->scale_output,
            tile_tokens,
            input_u8_ping,
            output_s32_scratch
        );
    }

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

    // Free L2 scratch if we allocated it
    if (!used_l1_scratch) {
        pi_l2_free(input_u8_ping, single_input_size);
        if (input_u8_pong) pi_l2_free(input_u8_pong, single_input_size);
        if (output_s32_scratch) pi_l2_free(output_s32_scratch, output_scratch_size);
        if (output_s8_tile_scratch) pi_l2_free(output_s8_tile_scratch, output_s8_scratch_size);
        if (outquant_scale) pi_l2_free(outquant_scale, outquant_size);
        if (outquant_scale_shift) pi_l2_free(outquant_scale_shift, outquant_size);
        if (outquant_bias) pi_l2_free(outquant_bias, outquant_bias_size);
    }

    // Free copied weights and bias correction buffers if we allocated them
    if (copied_weights) {
        if (copied_weights_in_l1) {
            pi_cl_l1_free(NULL, copied_weights, copied_weights_size);
        } else {
            pi_l2_free(copied_weights, copied_weights_size);
        }
    }
    if (copied_bias_corr) {
        if (copied_weights_in_l1) {
            pi_cl_l1_free(NULL, copied_bias_corr, bias_size);
        } else {
            pi_l2_free(copied_bias_corr, bias_size);
        }
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: %s NE16 linear complete\n", layer->name);
#endif
}

void execute_conv2d_1x1_ne16(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.conv2d_ne16) *params = &layer->params.conv2d_ne16;

    /* Clear NE16 state between layers to avoid contamination */
    ne16_soft_clear_all();

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing conv2d_1x1_ne16 (in=%dx%dx%d, out_ch=%d)\n",
           layer->name, params->in_h, params->in_w, params->in_channels, params->out_channels);
#endif

    // 1x1 conv is treated as linear: flatten spatial to tokens
    // tokens = in_h * in_w, in_features = in_channels, out_features = out_channels

    if (!ctx->input_buffer_l2 || !ctx->output_buffer_l2) {
        printf("CL: ERROR - %s: null input/output buffer\n", layer->name);
        return;
    }

    // Get pre-packed weights from context (runtime) or fallback to LayerSpec (compile-time)
    const int8_t *weights_packed = ctx->ne16_weights_packed ? ctx->ne16_weights_packed : params->weights_packed;
    const int32_t *bias_corrected = ctx->ne16_bias_corrected ? ctx->ne16_bias_corrected : params->bias_corrected;

    // CRITICAL: NE16 on gvsoc only reads weights correctly from L1, not L2!
    // Use same FAST PATH logic as linear kernel
    uint8_t *runtime_packed = NULL;
    int32_t *runtime_bias_corr = NULL;
    const size_t bias_size = (size_t)params->out_channels * sizeof(int32_t);
    int weights_in_l1 = 0;
    size_t allocated_weight_size = 0;

    // Calculate packed weight size (padded to 16-byte Ki groups)
    const int nb_ki = (params->in_channels + 15) / 16;
    const size_t packed_size = (size_t)params->out_channels * (size_t)nb_ki * 16;

    // FAST PATH: Pre-packed weights available from offline codegen
    // Use async DMA from L2 to L1 for better performance
    int used_persistent_l1 = 0;  // Track if we used persistent buffers
    if (weights_packed && bias_corrected) {
#ifndef MINIMAL_OUTPUT
        printf("CL: NE16 Conv1x1 FAST PATH: Using pre-packed weights for %s\n", layer->name);
#endif
        // Try persistent L1 buffers first (no alloc/free overhead)
        if (ctx->ne16_weight_l1 && ctx->ne16_weight_l1_size >= packed_size &&
            ctx->ne16_bias_l1 && ctx->ne16_bias_l1_size >= bias_size) {
            runtime_packed = ctx->ne16_weight_l1;
            runtime_bias_corr = ctx->ne16_bias_l1;
            weights_in_l1 = 1;
            used_persistent_l1 = 1;
            allocated_weight_size = 0;

            pi_cl_dma_copy_t dma_weights, dma_bias;

            dma_weights.ext = (uint32_t)weights_packed;
            dma_weights.loc = (uint32_t)runtime_packed;
            dma_weights.size = packed_size;
            dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_weights.merge = 0;
            pi_cl_dma_memcpy(&dma_weights);

            dma_bias.ext = (uint32_t)bias_corrected;
            dma_bias.loc = (uint32_t)runtime_bias_corr;
            dma_bias.size = bias_size;
            dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_bias.merge = 0;
            pi_cl_dma_memcpy(&dma_bias);

            pi_cl_dma_wait(&dma_weights);
            pi_cl_dma_wait(&dma_bias);

#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 Conv1x1 PERSISTENT L1: DMA'd %zu+%zu bytes for %s\n",
                   packed_size, bias_size, layer->name);
#endif
            weights_packed = (const int8_t *)runtime_packed;
            bias_corrected = runtime_bias_corr;
        } else {
            // Fallback: per-layer L1 allocation
            runtime_packed = (uint8_t *)pi_cl_l1_malloc(NULL, packed_size);
            runtime_bias_corr = (int32_t *)pi_cl_l1_malloc(NULL, bias_size);
            if (runtime_packed && runtime_bias_corr) {
                weights_in_l1 = 1;
                allocated_weight_size = packed_size;

                pi_cl_dma_copy_t dma_weights, dma_bias;

                dma_weights.ext = (uint32_t)weights_packed;
                dma_weights.loc = (uint32_t)runtime_packed;
                dma_weights.size = packed_size;
                dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
                dma_weights.merge = 0;
                pi_cl_dma_memcpy(&dma_weights);

                dma_bias.ext = (uint32_t)bias_corrected;
                dma_bias.loc = (uint32_t)runtime_bias_corr;
                dma_bias.size = bias_size;
                dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
                dma_bias.merge = 0;
                pi_cl_dma_memcpy(&dma_bias);

                pi_cl_dma_wait(&dma_weights);
                pi_cl_dma_wait(&dma_bias);

#ifndef MINIMAL_OUTPUT
                printf("CL: NE16 Conv1x1 PER-LAYER L1: DMA'd %zu+%zu bytes for %s\n",
                       packed_size, bias_size, layer->name);
#endif
                weights_packed = (const int8_t *)runtime_packed;
                bias_corrected = runtime_bias_corr;
            } else {
                // L1 allocation failed, fall back to L2
#ifndef MINIMAL_OUTPUT
                printf("CL: NE16 Conv1x1 WARNING: L1 alloc failed, using L2\n");
#endif
                if (runtime_packed) pi_cl_l1_free(NULL, runtime_packed, packed_size);
                if (runtime_bias_corr) pi_cl_l1_free(NULL, runtime_bias_corr, bias_size);
                runtime_packed = (uint8_t *)pi_l2_malloc(packed_size);
                runtime_bias_corr = (int32_t *)pi_l2_malloc(bias_size);
                if (runtime_packed && runtime_bias_corr) {
                    allocated_weight_size = packed_size;
                    memcpy(runtime_packed, weights_packed, packed_size);
                    memcpy(runtime_bias_corr, bias_corrected, bias_size);
                    asm volatile("fence" ::: "memory");
                    weights_packed = (const int8_t *)runtime_packed;
                    bias_corrected = runtime_bias_corr;
                }
            }
        }
    }

    uint8_t *copied_weights = used_persistent_l1 ? NULL : runtime_packed;
    int32_t *copied_bias_corr = used_persistent_l1 ? NULL : runtime_bias_corr;
    int copied_weights_in_l1 = weights_in_l1;
    size_t copied_weights_size = allocated_weight_size;

    if (!weights_packed || !bias_corrected) {
        printf("CL: ERROR - %s: null NE16 packed weights or bias\n", layer->name);
        return;
    }

    const int total_tokens = params->batch * params->in_h * params->in_w;
    const int tile_tokens = 64;  // Default tile size
    const size_t input_scratch_size = (size_t)tile_tokens * (size_t)params->in_channels;
    const size_t output_scratch_size = (size_t)tile_tokens * (size_t)params->out_channels * sizeof(int32_t);

    uint8_t *input_u8_scratch = NULL;
    int32_t *output_s32_scratch = NULL;

    int used_l1_scratch = 0;
    if (ctx->l1_buffer && ctx->l1_buffer_size >= input_scratch_size + output_scratch_size) {
        input_u8_scratch = (uint8_t *)ctx->l1_buffer;
        output_s32_scratch = (int32_t *)(ctx->l1_buffer + input_scratch_size);
        used_l1_scratch = 1;
    } else {
        input_u8_scratch = (uint8_t *)pi_l2_malloc(input_scratch_size);
        output_s32_scratch = (int32_t *)pi_l2_malloc(output_scratch_size);
        if (!input_u8_scratch || !output_s32_scratch) {
            printf("CL: ERROR - %s: failed to allocate NE16 scratch buffers\n", layer->name);
            if (input_u8_scratch) pi_l2_free(input_u8_scratch, input_scratch_size);
            if (output_s32_scratch) pi_l2_free(output_s32_scratch, output_scratch_size);
            return;
        }
    }

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

    ne16_linear_int8_packed(
        ctx->input_buffer_l2,
        weights_packed,
        bias_corrected,
        ctx->output_buffer_l2,
        total_tokens,
        params->in_channels,
        params->out_channels,
        params->out_channels,
        params->scale_input,
        params->scale_weight,
        params->scale_output,
        tile_tokens,
        input_u8_scratch,
        output_s32_scratch
    );

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

    // Cleanup scratch buffers
    if (!used_l1_scratch) {
        pi_l2_free(input_u8_scratch, input_scratch_size);
        pi_l2_free(output_s32_scratch, output_scratch_size);
    }

    // Cleanup copied weights
    if (copied_weights) {
        if (copied_weights_in_l1) {
            pi_cl_l1_free(NULL, copied_weights, copied_weights_size);
        } else {
            pi_l2_free(copied_weights, copied_weights_size);
        }
    }
    if (copied_bias_corr) {
        if (copied_weights_in_l1) {
            pi_cl_l1_free(NULL, copied_bias_corr, bias_size);
        } else {
            pi_l2_free(copied_bias_corr, bias_size);
        }
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: %s NE16 conv2d_1x1 complete\n", layer->name);
#endif
}

// Worker arguments for 3x3 NE16 post-processing
typedef struct {
    const int32_t *acc_s32;     // [out_h, out_w, out_ch] INT32 accumulator
    const int32_t *bias_corr;   // [out_ch] bias correction
    int8_t *out_s8;             // [out_h, out_w, out_ch] INT8 output
    int out_h, out_w;
    int out_channels;
    float combined_scale;
} ne16_conv3x3_post_args_t;

static void ne16_conv3x3_postprocess_worker(void *args) {
    const ne16_conv3x3_post_args_t *a = (const ne16_conv3x3_post_args_t *)args;
    const int core_id = pi_core_id();
    const int total_pixels = a->out_h * a->out_w;
    const int chunk = (total_pixels + NUM_CORES - 1) / NUM_CORES;
    const int start_p = core_id * chunk;
    const int end_p = (start_p + chunk > total_pixels) ? total_pixels : (start_p + chunk);

    for (int p = start_p; p < end_p; p++) {
        const int32_t *acc_row = a->acc_s32 + (size_t)p * (size_t)a->out_channels;
        int8_t *out_row = a->out_s8 + (size_t)p * (size_t)a->out_channels;
        for (int c = 0; c < a->out_channels; c++) {
            int32_t acc = acc_row[c] + a->bias_corr[c];
            float val_fp32 = (float)acc * a->combined_scale;
            int32_t q = qround(val_fp32);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_row[c] = (int8_t)q;
        }
    }
}

// Scatter [out_h*out_w, tile_out_channels] INT32 tile into a full [out_h*out_w, full_out_channels] buffer.
typedef struct {
    const int32_t *src_tile;
    int32_t *dst_full;
    int total_pixels;
    int tile_out_channels;
    int full_out_channels;
    int ko_start;
} ne16_conv3x3_scatter_args_t;

static void ne16_conv3x3_scatter_s32_worker(void *args) {
    const ne16_conv3x3_scatter_args_t *a = (const ne16_conv3x3_scatter_args_t *)args;
    const int core_id = pi_core_id();
    const int chunk = (a->total_pixels + NUM_CORES - 1) / NUM_CORES;
    const int start_p = core_id * chunk;
    const int end_p = (start_p + chunk > a->total_pixels) ? a->total_pixels : (start_p + chunk);

    for (int p = start_p; p < end_p; p++) {
        memcpy(a->dst_full + (size_t)p * (size_t)a->full_out_channels + (size_t)a->ko_start,
               a->src_tile + (size_t)p * (size_t)a->tile_out_channels,
               (size_t)a->tile_out_channels * sizeof(int32_t));
    }
}

// Worker arguments for S8→U8 input conversion with padding
typedef struct {
    const int8_t *in_s8;
    uint8_t *out_u8;
    int in_h, in_w, in_ch;
    int pad_h, pad_w;
    int padded_h, padded_w;
} ne16_conv3x3_pad_args_t;

static void ne16_conv3x3_pad_input_worker(void *args) {
    const ne16_conv3x3_pad_args_t *a = (const ne16_conv3x3_pad_args_t *)args;
    const int core_id = pi_core_id();

    // Each core handles a subset of output rows (padded rows)
    const int total_rows = a->padded_h;
    const int chunk = (total_rows + NUM_CORES - 1) / NUM_CORES;
    const int start_row = core_id * chunk;
    const int end_row = (start_row + chunk > total_rows) ? total_rows : (start_row + chunk);

    const int row_size = a->padded_w * a->in_ch;
    const uint8_t pad_val = 128;  // 0 in signed domain

    for (int ph = start_row; ph < end_row; ph++) {
        uint8_t *out_row = a->out_u8 + (size_t)ph * row_size;
        const int src_h = ph - a->pad_h;

        if (src_h < 0 || src_h >= a->in_h) {
            // Entire row is padding
            for (int i = 0; i < row_size; i++) {
                out_row[i] = pad_val;
            }
        } else {
            // Row has padding on left/right, data in center
            const int8_t *in_row = a->in_s8 + (size_t)src_h * a->in_w * a->in_ch;

            // Left padding
            for (int pw = 0; pw < a->pad_w; pw++) {
                for (int c = 0; c < a->in_ch; c++) {
                    out_row[pw * a->in_ch + c] = pad_val;
                }
            }

            // Center (actual data, converted to U8)
            for (int w = 0; w < a->in_w; w++) {
                for (int c = 0; c < a->in_ch; c++) {
                    int idx_out = (a->pad_w + w) * a->in_ch + c;
                    int idx_in = w * a->in_ch + c;
                    out_row[idx_out] = (uint8_t)((int)in_row[idx_in] + 128);
                }
            }

            // Right padding
            for (int pw = a->pad_w + a->in_w; pw < a->padded_w; pw++) {
                for (int c = 0; c < a->in_ch; c++) {
                    out_row[pw * a->in_ch + c] = pad_val;
                }
            }
        }
    }
}

void execute_conv2d_3x3_ne16(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.conv2d_ne16) *params = &layer->params.conv2d_ne16;

    // Clear NE16 state between layers
    ne16_soft_clear_all();

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing conv2d_3x3_ne16 (in=%dx%dx%d, out_ch=%d, pad=%d,%d)\n",
           layer->name, params->in_h, params->in_w, params->in_channels,
           params->out_channels, params->pad_h, params->pad_w);
#endif

    if (!ctx->input_buffer_l2 || !ctx->output_buffer_l2) {
        printf("CL: ERROR - %s: null input/output buffer\n", layer->name);
        return;
    }

    // Get pre-packed weights from context or LayerSpec
    const int8_t *weights_packed = ctx->ne16_weights_packed ? ctx->ne16_weights_packed : params->weights_packed;
    const int32_t *bias_corrected = ctx->ne16_bias_corrected ? ctx->ne16_bias_corrected : params->bias_corrected;

    if (!weights_packed || !bias_corrected) {
        printf("CL: ERROR - %s: null NE16 packed weights or bias\n", layer->name);
        return;
    }

    // Input dimensions - we pre-pad the buffer so NE16 reads valid memory at all positions
    // The pre-padded buffer has (H+2*pad_h) x (W+2*pad_w) x C elements
    const int orig_h = params->in_h;
    const int orig_w = params->in_w;
    const int orig_input_elements = orig_h * orig_w * params->in_channels;
    const int padded_h = orig_h + 2 * params->pad_h;
    const int padded_w = orig_w + 2 * params->pad_w;
    const int padded_elements = padded_h * padded_w * params->in_channels;

    // Output dimensions for 3x3 stride 1 on the padded input
    // out_dim = padded_dim - 2 = in_dim + 2*pad - 2 = in_dim (for pad=1)
    const int out_h = padded_h - 2;
    const int out_w = padded_w - 2;
    const int out_elements = out_h * out_w * params->out_channels;

    // Calculate packed weight size for 3x3: Ko * ceil(Ki/16) * 16 * 9 bytes
    const int nb_ki = (params->in_channels + 15) / 16;
    const size_t packed_size = (size_t)params->out_channels * (size_t)nb_ki * 16 * 9;
    const size_t bias_size = (size_t)params->out_channels * sizeof(int32_t);

    // Allocate scratch buffers (pre-padded U8 input, INT32 output)
    const size_t input_u8_size = (size_t)padded_elements * sizeof(uint8_t);
    const size_t output_s32_size = (size_t)out_elements * sizeof(int32_t);

    // IMPORTANT: NE16 on GVSOC requires ALL buffers (input, weights, output) in L1.
    // We MUST use L1 for input_u8 and output_s32.
    uint8_t *input_u8 = NULL;
    int32_t *output_s32 = NULL;
    int used_l1_scratch = 0;
    int input_u8_allocated = 0;
    int output_s32_allocated = 0;

    if (ctx->l1_buffer && ctx->l1_buffer_size >= input_u8_size + output_s32_size) {
        input_u8 = (uint8_t *)ctx->l1_buffer;
        output_s32 = (int32_t *)(ctx->l1_buffer + input_u8_size);
        used_l1_scratch = 1;
    } else {
        // Try L1 allocation - NE16 requires L1 buffers
        input_u8 = (uint8_t *)pi_cl_l1_malloc(NULL, input_u8_size);
        output_s32 = (int32_t *)pi_cl_l1_malloc(NULL, output_s32_size);
        if (!input_u8 || !output_s32) {
            printf("CL: ERROR - %s: failed to allocate NE16 3x3 L1 buffers (need %u + %u bytes)\n",
                   layer->name, (unsigned)input_u8_size, (unsigned)output_s32_size);
            if (input_u8) pi_cl_l1_free(NULL, input_u8, input_u8_size);
            if (output_s32) pi_cl_l1_free(NULL, output_s32, output_s32_size);
            return;
        }
        input_u8_allocated = 1;
        output_s32_allocated = 1;
    }

    // Copy weights to L1 for fast NE16 access
    uint8_t *weights_l1 = NULL;
    int32_t *bias_l1 = NULL;
    int weights_in_l1 = 0;

    // Try persistent L1 buffers first
    if (ctx->ne16_weight_l1 && ctx->ne16_weight_l1_size >= packed_size &&
        ctx->ne16_bias_l1 && ctx->ne16_bias_l1_size >= bias_size) {
        weights_l1 = ctx->ne16_weight_l1;
        bias_l1 = ctx->ne16_bias_l1;
        weights_in_l1 = 1;

        pi_cl_dma_copy_t dma_weights, dma_bias;
        dma_weights.ext = (uint32_t)weights_packed;
        dma_weights.loc = (uint32_t)weights_l1;
        dma_weights.size = packed_size;
        dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
        dma_weights.merge = 0;
        pi_cl_dma_memcpy(&dma_weights);

        dma_bias.ext = (uint32_t)bias_corrected;
        dma_bias.loc = (uint32_t)bias_l1;
        dma_bias.size = bias_size;
        dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
        dma_bias.merge = 0;
        pi_cl_dma_memcpy(&dma_bias);

        pi_cl_dma_wait(&dma_weights);
        pi_cl_dma_wait(&dma_bias);

        weights_packed = (const int8_t *)weights_l1;
        bias_corrected = bias_l1;
    } else {
        // Try per-layer L1 allocation
        weights_l1 = (uint8_t *)pi_cl_l1_malloc(NULL, packed_size);
        bias_l1 = (int32_t *)pi_cl_l1_malloc(NULL, bias_size);
        if (weights_l1 && bias_l1) {
            weights_in_l1 = 1;

            pi_cl_dma_copy_t dma_weights, dma_bias;
            dma_weights.ext = (uint32_t)weights_packed;
            dma_weights.loc = (uint32_t)weights_l1;
            dma_weights.size = packed_size;
            dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_weights.merge = 0;
            pi_cl_dma_memcpy(&dma_weights);

            dma_bias.ext = (uint32_t)bias_corrected;
            dma_bias.loc = (uint32_t)bias_l1;
            dma_bias.size = bias_size;
            dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_bias.merge = 0;
            pi_cl_dma_memcpy(&dma_bias);

            pi_cl_dma_wait(&dma_weights);
            pi_cl_dma_wait(&dma_bias);

            weights_packed = (const int8_t *)weights_l1;
            bias_corrected = bias_l1;
        } else {
            // L1 alloc failed, use L2 weights directly
            if (weights_l1) pi_cl_l1_free(NULL, weights_l1, packed_size);
            if (bias_l1) pi_cl_l1_free(NULL, bias_l1, bias_size);
            weights_l1 = NULL;
            bias_l1 = NULL;
#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 Conv3x3 WARNING: L1 alloc failed, using L2 weights\n");
#endif
        }
    }

#ifdef ENABLE_PERF_COUNTERS
    perf_compute_start();
#endif

#ifndef MINIMAL_OUTPUT
    // Debug: print original S8 input BEFORE conversion
    printf("CL: %s original S8 input (first 20): ", layer->name);
    const int8_t *orig_s8_ptr = ctx->input_buffer_l2;
    for (int i = 0; i < 20 && i < params->in_h * params->in_w * params->in_channels; i++) {
        printf("%d ", (int)orig_s8_ptr[i]);
    }
    printf("\n");
#endif

    // Step 1: Convert input from S8 to U8 WITH pre-padding
    // We pre-pad the buffer so NE16 reads valid memory at all positions.
    // The pad value is 128 (which represents 0 in the signed domain).
    const uint8_t pad_val = 128;

    // First, fill entire buffer with pad_val (for the padding regions)
    for (int i = 0; i < padded_elements; i++) {
        input_u8[i] = pad_val;
    }

    // Then copy the actual data into the center (S8->U8 conversion)
    // Layout is HWC, so we need to copy row by row
    {
        const int8_t *in_s8 = ctx->input_buffer_l2;
        const int C = params->in_channels;
        const int pad_h_val = params->pad_h;
        const int pad_w_val = params->pad_w;

        for (int h = 0; h < orig_h; h++) {
            for (int w = 0; w < orig_w; w++) {
                const int src_offset = (h * orig_w + w) * C;
                const int dst_offset = ((h + pad_h_val) * padded_w + (w + pad_w_val)) * C;
                for (int c = 0; c < C; c++) {
                    input_u8[dst_offset + c] = (uint8_t)((int)in_s8[src_offset + c] + 128);
                }
            }
        }
    }

    // Memory barrier to ensure all writes are complete before NE16 reads
    asm volatile("fence" ::: "memory");

#ifndef MINIMAL_OUTPUT
    printf("CL: %s S8->U8 pre-pad conversion: orig=%dx%d padded=%dx%d C=%d\n",
           layer->name, orig_h, orig_w, padded_h, padded_w, params->in_channels);
    printf("CL: %s U8 input (first 10): ", layer->name);
    for (int i = 0; i < 10 && i < padded_elements; i++) {
        printf("%d ", (int)input_u8[i]);
    }
    printf("\n");
    // Also print some from the center to show actual data
    printf("CL: %s U8 input center (row %d): ", layer->name, params->pad_h);
    const int center_start = (params->pad_h * padded_w + params->pad_w) * params->in_channels;
    for (int i = 0; i < 10 && i < padded_elements - center_start; i++) {
        printf("%d ", (int)input_u8[center_start + i]);
    }
    printf("\n");
#endif

    // Zero-initialize output buffer
    for (int i = 0; i < out_elements; i++) {
        output_s32[i] = 0;
    }

    // Step 2: Run NE16 3x3 convolution on the pre-padded buffer
    // Pass the PADDED dimensions and set padding=0 (since we pre-padded).
    //
    // IMPORTANT (GVSOC correctness): If packed weights don't fit in L1, NE16 can behave incorrectly
    // when reading weights directly from L2. In that case, we fall back to Ko-tiling (DMA weights
    // tiles to L1, run NE16 per tile, and scatter the INT32 accumulators into the full output).
    ne16_init();

    if (weights_in_l1) {
        ne16_conv3x3_u8_u8_to_s32(
            input_u8,
            (const uint8_t *)weights_packed,
            output_s32,
            padded_w,        // Padded input width
            padded_h,        // Padded input height
            params->in_channels,
            params->out_channels,
            0,               // No additional padding (we pre-padded)
            0,               // No additional padding (we pre-padded)
            NE16_WEIGHT_OFFSET  // -128
        );
    } else {
#ifndef MINIMAL_OUTPUT
        printf("CL: NE16 3x3: weights too large for L1 (%zu bytes), using Ko-tiling\n", packed_size);
#endif
        const size_t weight_per_ko = (size_t)nb_ki * 144u;  // bytes per output channel (3x3 packed)

        // Target ~64KB weight tiles (fits typical L1 budgets); keep tile_ko a multiple of 8.
        const size_t target_l1_tile_bytes = 65536u;
        int tile_ko = (int)(target_l1_tile_bytes / weight_per_ko);
        if (tile_ko > params->out_channels) tile_ko = params->out_channels;
        if (tile_ko >= 8) {
            tile_ko = (tile_ko / 8) * 8;
        }
        if (tile_ko < 8 && params->out_channels >= 8) {
            tile_ko = 8;
        }
        if (tile_ko < 1) {
            printf("CL: ERROR - %s: invalid tile_ko=%d for NE16 3x3 tiling\n", layer->name, tile_ko);
            goto cleanup;
        }

        // Allocate L1 tile buffer for packed weights.
        size_t tile_weight_size = (size_t)tile_ko * weight_per_ko;
        uint8_t *weights_tile_l1 = NULL;
        int weights_tile_allocated = 0;

        if (ctx->ne16_weight_l1 && ctx->ne16_weight_l1_size >= tile_weight_size) {
            weights_tile_l1 = ctx->ne16_weight_l1;
        } else {
            // If L1 alloc fails, reduce tile_ko until it fits.
            while (!weights_tile_l1 && tile_ko >= 8) {
                tile_weight_size = (size_t)tile_ko * weight_per_ko;
                weights_tile_l1 = (uint8_t *)pi_cl_l1_malloc(NULL, tile_weight_size);
                if (!weights_tile_l1) {
                    tile_ko -= 8;
                }
            }
            if (weights_tile_l1) {
                weights_tile_allocated = 1;
            }
        }

        if (!weights_tile_l1) {
            printf("CL: ERROR - %s: failed to allocate L1 weight tile buffer for NE16 3x3\n", layer->name);
            goto cleanup;
        }

        // Allocate a temporary tile output buffer (INT32) for NE16 streamout.
        // IMPORTANT: NE16 on GVSOC requires output to be in L1, not L2.
        const int total_pixels = out_h * out_w;
        const size_t output_s32_tile_size = (size_t)total_pixels * (size_t)tile_ko * sizeof(int32_t);
        int32_t *output_s32_tile = (int32_t *)pi_cl_l1_malloc(NULL, output_s32_tile_size);
        int output_tile_allocated = 0;
        if (!output_s32_tile) {
            // Try reducing tile_ko to fit both weights and output in L1
            int reduced_tile_ko = tile_ko;
            while (!output_s32_tile && reduced_tile_ko >= 8) {
                reduced_tile_ko -= 8;
                size_t reduced_out_size = (size_t)total_pixels * (size_t)reduced_tile_ko * sizeof(int32_t);
                output_s32_tile = (int32_t *)pi_cl_l1_malloc(NULL, reduced_out_size);
                if (output_s32_tile) {
                    // Also reallocate weights tile for smaller size
                    if (weights_tile_allocated) {
                        pi_cl_l1_free(NULL, weights_tile_l1, tile_weight_size);
                    }
                    tile_ko = reduced_tile_ko;
                    tile_weight_size = (size_t)tile_ko * weight_per_ko;
                    weights_tile_l1 = (uint8_t *)pi_cl_l1_malloc(NULL, tile_weight_size);
                    if (!weights_tile_l1) {
                        pi_cl_l1_free(NULL, output_s32_tile, reduced_out_size);
                        output_s32_tile = NULL;
                        continue;
                    }
                    weights_tile_allocated = 1;
                    output_tile_allocated = 1;
                }
            }
        } else {
            output_tile_allocated = 1;
        }
        if (!output_s32_tile) {
            printf("CL: ERROR - %s: failed to allocate L1 output tile buffer for NE16 3x3\n", layer->name);
            if (weights_tile_allocated) {
                pi_cl_l1_free(NULL, weights_tile_l1, tile_weight_size);
            }
            goto cleanup;
        }
        // Recalculate tile output size after potential reduction
        const size_t actual_output_s32_tile_size = (size_t)total_pixels * (size_t)tile_ko * sizeof(int32_t);

        // Run NE16 per Ko tile.
        for (int ko_start = 0; ko_start < params->out_channels; ko_start += tile_ko) {
            int current_tile_ko = params->out_channels - ko_start;
            if (current_tile_ko > tile_ko) current_tile_ko = tile_ko;

            const size_t current_weight_size = (size_t)current_tile_ko * weight_per_ko;

            // DMA weights tile into L1
            pi_cl_dma_copy_t dma_weights;
            dma_weights.ext = (uint32_t)((const uint8_t *)weights_packed + (size_t)ko_start * weight_per_ko);
            dma_weights.loc = (uint32_t)weights_tile_l1;
            dma_weights.size = current_weight_size;
            dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_weights.merge = 0;
            pi_cl_dma_memcpy(&dma_weights);
            pi_cl_dma_wait(&dma_weights);

            // Run NE16 for this tile (writes contiguous [out_h*out_w, tile_ko] accumulators)
            ne16_conv3x3_u8_u8_to_s32(
                input_u8,
                weights_tile_l1,
                output_s32_tile,
                padded_w,
                padded_h,
                params->in_channels,
                current_tile_ko,
                0,
                0,
                NE16_WEIGHT_OFFSET
            );

            // Scatter tile accumulators into the full accumulator buffer (HWC)
            ne16_conv3x3_scatter_args_t scatter_args = {
                .src_tile = output_s32_tile,
                .dst_full = output_s32,
                .total_pixels = total_pixels,
                .tile_out_channels = current_tile_ko,
                .full_out_channels = params->out_channels,
                .ko_start = ko_start,
            };
            pi_cl_team_fork(NUM_CORES, ne16_conv3x3_scatter_s32_worker, &scatter_args);
        }

        if (output_tile_allocated) {
            pi_cl_l1_free(NULL, output_s32_tile, actual_output_s32_tile_size);
        }
        if (weights_tile_allocated) {
            pi_cl_l1_free(NULL, weights_tile_l1, tile_weight_size);
        }
    }

#ifndef MINIMAL_OUTPUT
    // Debug: print NE16 raw accumulator output (first 10 values)
    printf("CL: %s NE16 s32 output (first 10): ", layer->name);
    for (int i = 0; i < 10 && i < out_elements; i++) {
        printf("%d ", (int)output_s32[i]);
    }
    printf("\n");
    printf("CL: %s bias_corr (first 5): ", layer->name);
    for (int c = 0; c < 5 && c < params->out_channels; c++) {
        printf("%d ", (int)bias_corrected[c]);
    }
    printf("\n");
#endif

    // Step 3: Post-process: apply bias and requantize to INT8 (parallel)
    const float combined_scale = params->scale_input * params->scale_weight / params->scale_output;
#ifndef MINIMAL_OUTPUT
    printf("CL: %s scales: in=%f wt=%f out=%f combined=%f\n",
           layer->name, params->scale_input, params->scale_weight, params->scale_output, combined_scale);
    // Debug: compute first output value manually
    int32_t acc0 = output_s32[0] + bias_corrected[0];
    float val0 = (float)acc0 * combined_scale;
    printf("CL: %s debug: acc[0]=%d + bias[0]=%d = %d -> *%f = %f -> clip to %d\n",
           layer->name, (int)output_s32[0], (int)bias_corrected[0], acc0, combined_scale, val0,
           (int)(val0 > 127 ? 127 : (val0 < -128 ? -128 : (int)val0)));
    // Debug: Compute what SW path would get for first output pixel
    // SW: acc_sw = sum(input_s8 * weight_s8) = NE16_acc - 128*sum(weights)
    // Since bias_corr = bias - 128*sum(weights), we have:
    // 128*sum(weights) = bias - bias_corr
    // Let's use the original ctx->input_buffer_l2 and stored weights to compute SW result
    const int8_t *orig_weights = (const int8_t *)ctx->weight_l2;  // Original S8 weights
    // Debug: print packed weights pointer and first few bytes
    printf("CL: %s packed_weights ptr=%p first bytes: ", layer->name, (void *)weights_packed);
    for (int i = 0; i < 10; i++) {
        printf("%02x ", weights_packed[i]);
    }
    printf("\n");
    // Also check the original weights if available
    if (ctx->weight_l2) {
        printf("CL: %s orig weight_l2 ptr=%p first bytes: ", layer->name, (void *)ctx->weight_l2);
        for (int i = 0; i < 10; i++) {
            printf("%02x ", ((uint8_t *)ctx->weight_l2)[i]);
        }
        printf("\n");
    }
#endif
    ne16_conv3x3_post_args_t post_args = {
        .acc_s32 = output_s32,
        .bias_corr = bias_corrected,
        .out_s8 = ctx->output_buffer_l2,
        .out_h = out_h,
        .out_w = out_w,
        .out_channels = params->out_channels,
        .combined_scale = combined_scale
    };
    pi_cl_team_fork(NUM_CORES, ne16_conv3x3_postprocess_worker, &post_args);

#ifndef MINIMAL_OUTPUT
    // Debug: check what was actually written to output
    printf("CL: %s actual output (first 10): ", layer->name);
    for (int i = 0; i < 10 && i < out_elements; i++) {
        printf("%d ", (int)ctx->output_buffer_l2[i]);
    }
    printf("\n");
#endif

#ifdef ENABLE_PERF_COUNTERS
    if (ctx->perf_counter) {
        ctx->perf_counter->compute_cycles += perf_compute_end();
    }
#endif

cleanup:
    // Release scratch buffers allocated in L1.
    if (!used_l1_scratch) {
        if (input_u8_allocated) {
            pi_cl_l1_free(NULL, input_u8, input_u8_size);
        }
        if (output_s32_allocated) {
            pi_cl_l1_free(NULL, output_s32, output_s32_size);
        }
    }

    // Cleanup L1 weights if we allocated them (not persistent)
    if (weights_in_l1 && weights_l1 && ctx->ne16_weight_l1 != weights_l1) {
        pi_cl_l1_free(NULL, weights_l1, packed_size);
        pi_cl_l1_free(NULL, bias_l1, bias_size);
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: %s NE16 conv2d_3x3 complete (out=%dx%dx%d)\n",
           layer->name, out_h, out_w, params->out_channels);
#endif
}

#ifdef ARES_NE16_DEPTHWISE
void execute_conv2d_3x3_dw_ne16(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.conv2d_ne16) *params = &layer->params.conv2d_ne16;

    // Clear NE16 state between layers
    ne16_soft_clear_all();

    // For depthwise, in_channels == out_channels
    const int channels = params->in_channels;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing conv2d_3x3_dw_ne16 (in=%dx%dx%d, pad=%d,%d)\n",
           layer->name, params->in_h, params->in_w, channels,
           params->pad_h, params->pad_w);
#endif

    if (!ctx->input_buffer_l2 || !ctx->output_buffer_l2) {
        printf("CL: ERROR - %s: null input/output buffer\n", layer->name);
        return;
    }

    // Get pre-packed depthwise weights from context or LayerSpec
    const int8_t *weights_packed = ctx->ne16_weights_packed ? ctx->ne16_weights_packed : params->weights_packed;
    const int32_t *bias_corrected = ctx->ne16_bias_corrected ? ctx->ne16_bias_corrected : params->bias_corrected;

    if (!weights_packed || !bias_corrected) {
        printf("CL: ERROR - %s: null NE16 packed depthwise weights or bias\n", layer->name);
        return;
    }

    // Input dimensions with pre-padding
    const int orig_h = params->in_h;
    const int orig_w = params->in_w;
    const int padded_h = orig_h + 2 * params->pad_h;
    const int padded_w = orig_w + 2 * params->pad_w;
    const int padded_elements = padded_h * padded_w * channels;

    // Output dimensions for 3x3 stride 1 on padded input
    const int out_h = padded_h - 2;
    const int out_w = padded_w - 2;
    const int out_elements = out_h * out_w * channels;

    // Depthwise packed weight size: ceil(channels/16) * 144 bytes
    const int nb_k = (channels + 15) / 16;
    const size_t packed_size = (size_t)nb_k * 8 * 3 * 3 * 2;  // nb_k * 144
    const size_t bias_size = (size_t)channels * sizeof(int32_t);

    // Allocate scratch buffers (pre-padded U8 input, INT32 output)
    const size_t input_u8_size = (size_t)padded_elements * sizeof(uint8_t);
    const size_t output_s32_size = (size_t)out_elements * sizeof(int32_t);

    uint8_t *input_u8 = NULL;
    int32_t *output_s32 = NULL;
    int used_l1_scratch = 0;
    int input_u8_allocated = 0;
    int output_s32_allocated = 0;

    if (ctx->l1_buffer && ctx->l1_buffer_size >= input_u8_size + output_s32_size) {
        input_u8 = (uint8_t *)ctx->l1_buffer;
        output_s32 = (int32_t *)(ctx->l1_buffer + input_u8_size);
        used_l1_scratch = 1;
    } else {
        input_u8 = (uint8_t *)pi_cl_l1_malloc(NULL, input_u8_size);
        output_s32 = (int32_t *)pi_cl_l1_malloc(NULL, output_s32_size);
        if (!input_u8 || !output_s32) {
            printf("CL: ERROR - %s: failed to allocate NE16 DW L1 buffers\n", layer->name);
            if (input_u8) pi_cl_l1_free(NULL, input_u8, input_u8_size);
            if (output_s32) pi_cl_l1_free(NULL, output_s32, output_s32_size);
            return;
        }
        input_u8_allocated = 1;
        output_s32_allocated = 1;
    }

    // Copy weights to L1
    uint8_t *weights_l1 = NULL;
    int32_t *bias_l1 = NULL;
    int weights_in_l1 = 0;

    if (ctx->ne16_weight_l1 && ctx->ne16_weight_l1_size >= packed_size &&
        ctx->ne16_bias_l1 && ctx->ne16_bias_l1_size >= bias_size) {
        weights_l1 = ctx->ne16_weight_l1;
        bias_l1 = ctx->ne16_bias_l1;
        weights_in_l1 = 1;

        pi_cl_dma_copy_t dma_weights, dma_bias;
        dma_weights.ext = (uint32_t)weights_packed;
        dma_weights.loc = (uint32_t)weights_l1;
        dma_weights.size = packed_size;
        dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
        dma_weights.merge = 0;
        pi_cl_dma_memcpy(&dma_weights);

        dma_bias.ext = (uint32_t)bias_corrected;
        dma_bias.loc = (uint32_t)bias_l1;
        dma_bias.size = bias_size;
        dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
        dma_bias.merge = 0;
        pi_cl_dma_memcpy(&dma_bias);

        pi_cl_dma_wait(&dma_weights);
        pi_cl_dma_wait(&dma_bias);

        weights_packed = (const int8_t *)weights_l1;
        bias_corrected = bias_l1;
    } else {
        weights_l1 = (uint8_t *)pi_cl_l1_malloc(NULL, packed_size);
        bias_l1 = (int32_t *)pi_cl_l1_malloc(NULL, bias_size);
        if (weights_l1 && bias_l1) {
            weights_in_l1 = 1;

            pi_cl_dma_copy_t dma_weights, dma_bias;
            dma_weights.ext = (uint32_t)weights_packed;
            dma_weights.loc = (uint32_t)weights_l1;
            dma_weights.size = packed_size;
            dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_weights.merge = 0;
            pi_cl_dma_memcpy(&dma_weights);

            dma_bias.ext = (uint32_t)bias_corrected;
            dma_bias.loc = (uint32_t)bias_l1;
            dma_bias.size = bias_size;
            dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
            dma_bias.merge = 0;
            pi_cl_dma_memcpy(&dma_bias);

            pi_cl_dma_wait(&dma_weights);
            pi_cl_dma_wait(&dma_bias);

            weights_packed = (const int8_t *)weights_l1;
            bias_corrected = bias_l1;
        } else {
            if (weights_l1) pi_cl_l1_free(NULL, weights_l1, packed_size);
            if (bias_l1) pi_cl_l1_free(NULL, bias_l1, bias_size);
            weights_l1 = NULL;
            bias_l1 = NULL;
#ifndef MINIMAL_OUTPUT
            printf("CL: NE16 DW WARNING: L1 alloc failed, using L2 weights\n");
#endif
        }
    }

    // Convert input from S8 to U8 with pre-padding
    const uint8_t pad_val = 128;

    // Fill entire buffer with pad_val
    for (int i = 0; i < padded_elements; i++) {
        input_u8[i] = pad_val;
    }

    // Copy actual data into center (S8->U8 conversion)
    {
        const int8_t *in_s8 = ctx->input_buffer_l2;
        const int pad_h_val = params->pad_h;
        const int pad_w_val = params->pad_w;

        for (int h = 0; h < orig_h; h++) {
            for (int w = 0; w < orig_w; w++) {
                const int src_offset = (h * orig_w + w) * channels;
                const int dst_offset = ((h + pad_h_val) * padded_w + (w + pad_w_val)) * channels;
                for (int c = 0; c < channels; c++) {
                    input_u8[dst_offset + c] = (uint8_t)((int)in_s8[src_offset + c] + 128);
                }
            }
        }
    }

    asm volatile("fence" ::: "memory");

    // Zero-initialize output buffer
    for (int i = 0; i < out_elements; i++) {
        output_s32[i] = 0;
    }

    // Run NE16 depthwise 3x3 convolution
    ne16_init();

    if (weights_in_l1) {
        ne16_conv3x3_dw_u8_u8_to_s32(
            input_u8,
            (const uint8_t *)weights_packed,
            output_s32,
            padded_w,
            padded_h,
            channels,
            0,  // No additional padding (pre-padded)
            0,
            NE16_WEIGHT_OFFSET
        );
    } else {
        // Fallback: use L2 weights directly (not ideal for GVSOC)
        ne16_conv3x3_dw_u8_u8_to_s32(
            input_u8,
            (const uint8_t *)weights_packed,
            output_s32,
            padded_w,
            padded_h,
            channels,
            0,
            0,
            NE16_WEIGHT_OFFSET
        );
    }

    // Postprocess: apply bias and requantize INT32 -> INT8
    const float scale_in = params->scale_input;
    const float scale_w = params->scale_weight;
    const float scale_out = params->scale_output;
    const float combined_scale = (scale_in * scale_w) / scale_out;

    int8_t *out_s8 = ctx->output_buffer_l2;
    for (int i = 0; i < out_elements; i++) {
        int32_t acc = output_s32[i];
        acc += bias_corrected[i % channels];  // Bias per channel
        float val_fp = (float)acc * combined_scale;
        int32_t q = (int32_t)roundf(val_fp);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out_s8[i] = (int8_t)q;
    }

    // Cleanup
    if (!used_l1_scratch) {
        if (input_u8_allocated) {
            pi_cl_l1_free(NULL, input_u8, input_u8_size);
        }
        if (output_s32_allocated) {
            pi_cl_l1_free(NULL, output_s32, output_s32_size);
        }
    }

    if (weights_in_l1 && weights_l1 && ctx->ne16_weight_l1 != weights_l1) {
        pi_cl_l1_free(NULL, weights_l1, packed_size);
        pi_cl_l1_free(NULL, bias_l1, bias_size);
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: %s NE16 conv2d_3x3_dw complete (out=%dx%dx%d)\n",
           layer->name, out_h, out_w, channels);
#endif
}

/**
 * NE16 Depthwise 3x3 with Spatial Tiling
 *
 * For large activations that don't fit in L1, we tile along the height dimension.
 * Each tile processes tile_h_out output rows, requiring tile_h_in input rows (with halo).
 *
 * Pipeline:
 *   PROLOGUE: Load weights+bias (once), load first input tile
 *   MAIN LOOP:
 *     - Run NE16 on current input -> current output (S32)
 *     - Store previous output tile (S32->S8 requant) to L2
 *     - Async load next input tile (with S8->U8 conversion and padding)
 *     - Swap double buffers
 *   EPILOGUE: Store last output tile
 */

// Arguments for parallel tile loading worker
typedef struct {
    uint8_t *dst;
    const int8_t *src_base;
    int tile_y;
    int tile_h_out;
    int tile_h_in;
    int orig_h;
    int orig_w;
    int padded_w;
    int channels;
    int pad_h;
    int pad_w;
    uint8_t pad_val;
    size_t input_tile_bytes;
} ne16_dw_tile_load_args_t;

// Arguments for parallel requantization worker
typedef struct {
    int8_t *dst_base;
    const int32_t *src;
    const int32_t *bias;
    int tile_y;
    int tile_h_out;
    int actual_h;
    int out_w;
    int channels;
    float combined_scale;
} ne16_dw_tile_store_args_t;

// Parallel worker for loading input tile with S8->U8 conversion (SIMD optimized)
static void ne16_dw_tile_load_worker(void *arg) {
    ne16_dw_tile_load_args_t *a = (ne16_dw_tile_load_args_t *)arg;
    int core_id = pi_core_id();
    int num_cores = pi_cl_cluster_nb_cores();

    int out_y_start = a->tile_y * a->tile_h_out;
    int in_y_start = out_y_start - a->pad_h;

    // For SIMD S8->U8: XOR with 0x80 (flip sign bit) is equivalent to +128
    // Pack 4 bytes of 0x80 for vectorized operation
    const uint32_t xor_mask = 0x80808080u;

    // Each core handles a subset of rows
    for (int h = core_id; h < a->tile_h_in; h += num_cores) {
        int src_y = in_y_start + h;
        int row_bytes = a->padded_w * a->channels;
        uint8_t *dst_row = a->dst + h * row_bytes;

        if (src_y < 0 || src_y >= a->orig_h) {
            // Padding row - use memset for speed (pad_val=128)
            memset(dst_row, a->pad_val, row_bytes);
        } else {
            // Valid row - copy with S8->U8 conversion using SIMD
            int left_pad_bytes = a->pad_w * a->channels;
            int center_bytes = a->orig_w * a->channels;
            int right_pad_bytes = a->pad_w * a->channels;

            // Left padding (memset)
            memset(dst_row, a->pad_val, left_pad_bytes);

            // Center: SIMD S8->U8 conversion (XOR with 0x80)
            const int8_t *src_ptr = a->src_base + src_y * a->orig_w * a->channels;
            uint8_t *dst_ptr = dst_row + left_pad_bytes;

            // Process 4 bytes at a time with SIMD
            int simd_count = center_bytes >> 2;
            const uint32_t *src32 = (const uint32_t *)src_ptr;
            uint32_t *dst32 = (uint32_t *)dst_ptr;
            for (int i = 0; i < simd_count; i++) {
                dst32[i] = src32[i] ^ xor_mask;
            }
            // Handle remainder
            int rem_start = simd_count << 2;
            for (int i = rem_start; i < center_bytes; i++) {
                dst_ptr[i] = (uint8_t)((int)src_ptr[i] + 128);
            }

            // Right padding (memset)
            memset(dst_row + left_pad_bytes + center_bytes, a->pad_val, right_pad_bytes);
        }
    }
    pi_cl_team_barrier();
}

// Parallel worker for storing output tile with S32->S8 requantization
static void ne16_dw_tile_store_worker(void *arg) {
    ne16_dw_tile_store_args_t *a = (ne16_dw_tile_store_args_t *)arg;
    int core_id = pi_core_id();
    int num_cores = pi_cl_cluster_nb_cores();

    int out_y_start = a->tile_y * a->tile_h_out;
    int total_elements = a->actual_h * a->out_w * a->channels;

    const int channels = a->channels;
    const int out_w = a->out_w;
    const float scale = a->combined_scale;

    // Each core handles strided elements
    for (int i = core_id; i < total_elements; i += num_cores) {
        int c = i % channels;
        int w = (i / channels) % out_w;
        int h = i / (channels * out_w);

        int32_t acc = a->src[i] + a->bias[c];
        int32_t q = qround((float)acc * scale);
        if (q > 127) q = 127;
        if (q < -128) q = -128;

        int out_idx = ((out_y_start + h) * out_w + w) * channels + c;
        a->dst_base[out_idx] = (int8_t)q;
    }
    pi_cl_team_barrier();
}

void execute_conv2d_3x3_dw_ne16_tiled(const LayerSpec *layer, layer_runtime_ctx_t *ctx) {
    const typeof(layer->params.conv2d_ne16) *params = &layer->params.conv2d_ne16;

    // Clear NE16 state
    ne16_soft_clear_all();

    const int channels = params->in_channels;
    const int orig_h = params->in_h;
    const int orig_w = params->in_w;
    const int pad_h = params->pad_h;
    const int pad_w = params->pad_w;
    const int padded_w = orig_w + 2 * pad_w;
    const int out_h = orig_h;  // Same spatial for stride=1, pad=1
    const int out_w = orig_w;

    // Check if hardware requant is available
    const uint8_t *hw_scale_l2 = ctx->ne16_hw_scale ? ctx->ne16_hw_scale : params->hw_scale;
    const uint8_t *hw_scale_shift_l2 = ctx->ne16_hw_scale_shift ? ctx->ne16_hw_scale_shift : params->hw_scale_shift;
    const int use_hw_requant = (params->use_hw_requant || ctx->ne16_use_hw_requant) &&
                               hw_scale_l2 != NULL &&
                               hw_scale_shift_l2 != NULL;

    // Tiling parameters
    const int num_tiles = params->ne16_dw_num_tiles;
    const int tile_h_out = params->ne16_dw_tile_h_out;
    const int tile_h_in = params->ne16_dw_tile_h_in;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s executing NE16 DW 3x3 TILED (in=%dx%dx%d, tiles=%d, tile_h=%d, hw_requant=%d)\n",
           layer->name, orig_h, orig_w, channels, num_tiles, tile_h_out, use_hw_requant);
#endif

    if (!ctx->input_buffer_l2 || !ctx->output_buffer_l2) {
        printf("CL: ERROR - %s: null input/output buffer\n", layer->name);
        return;
    }

    // Get weights from context or LayerSpec
    const int8_t *weights_packed_l2 = ctx->ne16_weights_packed ? ctx->ne16_weights_packed : params->weights_packed;
    const int32_t *bias_corrected_l2 = ctx->ne16_bias_corrected ? ctx->ne16_bias_corrected : params->bias_corrected;

    if (!weights_packed_l2 || !bias_corrected_l2) {
        printf("CL: ERROR - %s: null NE16 packed weights or bias\n", layer->name);
        return;
    }

    // Calculate buffer sizes
    // With HW requant: output is INT8 (1 byte), without: INT32 (4 bytes)
    const size_t input_tile_bytes = (size_t)tile_h_in * padded_w * channels;
    const size_t output_tile_bytes = use_hw_requant ?
        ((size_t)tile_h_out * out_w * channels) :
        ((size_t)tile_h_out * out_w * channels * sizeof(int32_t));
    const int nb_k = (channels + 15) / 16;
    const size_t weight_bytes = (size_t)nb_k * 8 * 3 * 3 * 2;
    const size_t bias_bytes = (size_t)channels * sizeof(int32_t);
    const size_t scale_bytes = use_hw_requant ? (size_t)channels : 0;

    // Total L1 requirement: 2*(input + output) + weights + bias + scales (if HW requant)
    const size_t total_l1_needed = 2 * (input_tile_bytes + output_tile_bytes) +
                                   weight_bytes + bias_bytes + 2 * scale_bytes;

    // Allocate L1 buffer
    uint8_t *l1_base = NULL;
    int l1_allocated = 0;

    if (ctx->l1_buffer && ctx->l1_buffer_size >= total_l1_needed) {
        l1_base = (uint8_t *)ctx->l1_buffer;
    } else {
        l1_base = (uint8_t *)pi_cl_l1_malloc(NULL, total_l1_needed);
        if (!l1_base) {
            printf("CL: ERROR - %s: failed to allocate L1 (%zu bytes)\n", layer->name, total_l1_needed);
            return;
        }
        l1_allocated = 1;
    }

    // Setup buffer layout
    // [input_A][output_A][input_B][output_B][weights][bias][scale][scale_shift]
    size_t single_io = input_tile_bytes + output_tile_bytes;
    uint8_t *input_a = l1_base;
    uint8_t *output_a_u8 = l1_base + input_tile_bytes;  // For HW requant (INT8 output)
    int32_t *output_a_s32 = (int32_t *)(l1_base + input_tile_bytes);  // For SW requant (INT32 output)
    uint8_t *input_b = l1_base + single_io;
    uint8_t *output_b_u8 = l1_base + single_io + input_tile_bytes;
    int32_t *output_b_s32 = (int32_t *)(l1_base + single_io + input_tile_bytes);
    uint8_t *weights_l1 = l1_base + 2 * single_io;
    int32_t *bias_l1 = (int32_t *)(l1_base + 2 * single_io + weight_bytes);
    uint8_t *scale_l1 = use_hw_requant ? (l1_base + 2 * single_io + weight_bytes + bias_bytes) : NULL;
    uint8_t *scale_shift_l1 = use_hw_requant ? (scale_l1 + scale_bytes) : NULL;

    int current_idx = 0;

    // PROLOGUE: Load weights, bias, and scales to L1
    pi_cl_dma_copy_t dma_weights, dma_bias;
    dma_weights.ext = (uint32_t)weights_packed_l2;
    dma_weights.loc = (uint32_t)weights_l1;
    dma_weights.size = weight_bytes;
    dma_weights.dir = PI_CL_DMA_DIR_EXT2LOC;
    dma_weights.merge = 0;
    pi_cl_dma_memcpy(&dma_weights);

    dma_bias.ext = (uint32_t)bias_corrected_l2;
    dma_bias.loc = (uint32_t)bias_l1;
    dma_bias.size = bias_bytes;
    dma_bias.dir = PI_CL_DMA_DIR_EXT2LOC;
    dma_bias.merge = 0;
    pi_cl_dma_memcpy(&dma_bias);

    pi_cl_dma_wait(&dma_weights);
    pi_cl_dma_wait(&dma_bias);

    // Load HW requant scales if available
    if (use_hw_requant) {
        pi_cl_dma_copy_t dma_scale, dma_shift;
        dma_scale.ext = (uint32_t)hw_scale_l2;
        dma_scale.loc = (uint32_t)scale_l1;
        dma_scale.size = scale_bytes;
        dma_scale.dir = PI_CL_DMA_DIR_EXT2LOC;
        dma_scale.merge = 0;
        pi_cl_dma_memcpy(&dma_scale);

        dma_shift.ext = (uint32_t)hw_scale_shift_l2;
        dma_shift.loc = (uint32_t)scale_shift_l1;
        dma_shift.size = scale_bytes;
        dma_shift.dir = PI_CL_DMA_DIR_EXT2LOC;
        dma_shift.merge = 0;
        pi_cl_dma_memcpy(&dma_shift);

        pi_cl_dma_wait(&dma_scale);
        pi_cl_dma_wait(&dma_shift);
    }

    // Combined scale for SW requantization (only used if !use_hw_requant)
    const float scale_in = params->scale_input;
    const float scale_w = params->scale_weight;
    const float scale_out = params->scale_output;
    const float combined_scale = (scale_in * scale_w) / scale_out;

    const uint8_t pad_val = 128;  // 0 in signed domain
    const int8_t *in_s8 = ctx->input_buffer_l2;
    int8_t *out_s8 = ctx->output_buffer_l2;

    // Initialize NE16
    ne16_init();

    // Setup args for parallel workers
    ne16_dw_tile_load_args_t load_args = {
        .src_base = in_s8,
        .tile_h_out = tile_h_out,
        .tile_h_in = tile_h_in,
        .orig_h = orig_h,
        .orig_w = orig_w,
        .padded_w = padded_w,
        .channels = channels,
        .pad_h = pad_h,
        .pad_w = pad_w,
        .pad_val = pad_val,
        .input_tile_bytes = input_tile_bytes
    };

    ne16_dw_tile_store_args_t store_args = {
        .dst_base = out_s8,
        .bias = bias_l1,
        .tile_h_out = tile_h_out,
        .out_w = out_w,
        .channels = channels,
        .combined_scale = combined_scale
    };

    // PROLOGUE: Load first input tile (parallel)
    uint8_t *cur_input = (current_idx == 0) ? input_a : input_b;

    load_args.dst = cur_input;
    load_args.tile_y = 0;
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), ne16_dw_tile_load_worker, &load_args);

    // DMA handle for HW requant output
    pi_cl_dma_copy_t dma_out;

    // MAIN LOOP
    for (int tile_y = 0; tile_y < num_tiles; tile_y++) {
        // Get current buffers
        cur_input = (current_idx == 0) ? input_a : input_b;

        // Calculate actual output height for this tile (last tile may be smaller)
        int out_y_start = tile_y * tile_h_out;
        int actual_tile_h_out = (out_y_start + tile_h_out <= out_h) ? tile_h_out : (out_h - out_y_start);

        if (use_hw_requant) {
            // HW REQUANT PATH: NE16 outputs INT8 directly
            int8_t *cur_output_s8 = (current_idx == 0) ? (int8_t *)output_a_u8 : (int8_t *)output_b_u8;

            // Run NE16 depthwise 3x3 with hardware requantization
            ne16_conv3x3_dw_u8_u8_to_s8(
                cur_input,
                weights_l1,
                bias_l1,
                scale_l1,
                scale_shift_l1,
                cur_output_s8,
                padded_w,
                tile_h_in,
                channels,
                0, 0,  // No additional padding (pre-padded)
                NE16_WEIGHT_OFFSET
            );

            // DMA previous output tile to L2 (if not first tile)
            if (tile_y > 0) {
                int prev_idx = 1 - current_idx;
                int8_t *prev_output_s8 = (prev_idx == 0) ? (int8_t *)output_a_u8 : (int8_t *)output_b_u8;
                int prev_tile_y = tile_y - 1;
                int prev_out_y_start = prev_tile_y * tile_h_out;
                int prev_actual_h = (prev_out_y_start + tile_h_out <= out_h) ? tile_h_out : (out_h - prev_out_y_start);
                size_t prev_out_bytes = (size_t)prev_actual_h * out_w * channels;

                dma_out.ext = (uint32_t)(out_s8 + prev_out_y_start * out_w * channels);
                dma_out.loc = (uint32_t)prev_output_s8;
                dma_out.size = prev_out_bytes;
                dma_out.dir = PI_CL_DMA_DIR_LOC2EXT;
                dma_out.merge = 0;
                pi_cl_dma_memcpy(&dma_out);
                pi_cl_dma_wait(&dma_out);
            }
        } else {
            // SW REQUANT PATH: NE16 outputs INT32, CPU does requantization
            int32_t *cur_output_s32 = (current_idx == 0) ? output_a_s32 : output_b_s32;

            // Zero output buffer before NE16 call
            memset(cur_output_s32, 0, actual_tile_h_out * out_w * channels * sizeof(int32_t));

            // Run NE16 depthwise 3x3 (INT32 output)
            ne16_conv3x3_dw_u8_u8_to_s32(
                cur_input,
                weights_l1,
                cur_output_s32,
                padded_w,
                tile_h_in,
                channels,
                0, 0,  // No additional padding (pre-padded)
                NE16_WEIGHT_OFFSET
            );

            // Store previous output tile with SW requant (if not first tile)
            if (tile_y > 0) {
                int prev_idx = 1 - current_idx;
                int32_t *prev_output_s32 = (prev_idx == 0) ? output_a_s32 : output_b_s32;
                int prev_tile_y = tile_y - 1;
                int prev_out_y_start = prev_tile_y * tile_h_out;
                int prev_actual_h = (prev_out_y_start + tile_h_out <= out_h) ? tile_h_out : (out_h - prev_out_y_start);

                store_args.src = prev_output_s32;
                store_args.tile_y = prev_tile_y;
                store_args.actual_h = prev_actual_h;
                pi_cl_team_fork(pi_cl_cluster_nb_cores(), ne16_dw_tile_store_worker, &store_args);
            }
        }

        // Prefetch next input tile (if not last tile) - parallel
        if (tile_y < num_tiles - 1) {
            int next_idx = 1 - current_idx;
            uint8_t *next_input = (next_idx == 0) ? input_a : input_b;

            load_args.dst = next_input;
            load_args.tile_y = tile_y + 1;
            pi_cl_team_fork(pi_cl_cluster_nb_cores(), ne16_dw_tile_load_worker, &load_args);
        }

        // Swap double buffers
        current_idx = 1 - current_idx;
    }

    // EPILOGUE: Store last output tile
    int last_tile_y = num_tiles - 1;
    int last_idx = 1 - current_idx;  // After last swap, previous is the last computed
    int last_out_y_start = last_tile_y * tile_h_out;
    int last_actual_h = (last_out_y_start + tile_h_out <= out_h) ? tile_h_out : (out_h - last_out_y_start);

    if (use_hw_requant) {
        // DMA last tile to L2
        int8_t *last_output_s8 = (last_idx == 0) ? (int8_t *)output_a_u8 : (int8_t *)output_b_u8;
        size_t last_out_bytes = (size_t)last_actual_h * out_w * channels;

        dma_out.ext = (uint32_t)(out_s8 + last_out_y_start * out_w * channels);
        dma_out.loc = (uint32_t)last_output_s8;
        dma_out.size = last_out_bytes;
        dma_out.dir = PI_CL_DMA_DIR_LOC2EXT;
        dma_out.merge = 0;
        pi_cl_dma_memcpy(&dma_out);
        pi_cl_dma_wait(&dma_out);
    } else {
        // SW requant last tile
        int32_t *last_output_s32 = (last_idx == 0) ? output_a_s32 : output_b_s32;
        store_args.src = last_output_s32;
        store_args.tile_y = last_tile_y;
        store_args.actual_h = last_actual_h;
        pi_cl_team_fork(pi_cl_cluster_nb_cores(), ne16_dw_tile_store_worker, &store_args);
    }

    // Cleanup
    if (l1_allocated) {
        pi_cl_l1_free(NULL, l1_base, total_l1_needed);
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: %s NE16 DW 3x3 TILED complete (out=%dx%dx%d)\n",
           layer->name, out_h, out_w, channels);
#endif
}
#endif /* ARES_NE16_DEPTHWISE */

#endif /* ARES_USE_NE16 */

/* --- Golden Validation Helper --- */

void validate_layer_output(const char *layer_name, const int8_t *output,
                           const int8_t *golden, size_t size) {
#ifndef DISABLE_GOLDEN_VALIDATION
    int mismatches = 0;
    int max_diff = 0;
    int64_t sum_diff = 0;

    for (size_t i = 0; i < size; i++) {
        int diff = abs((int)output[i] - (int)golden[i]);
        if (diff > 0) {
            mismatches++;
            sum_diff += diff;
            if (diff > max_diff) max_diff = diff;
        }
    }

    float error_rate = 100.0f * mismatches / size;
    float mean_diff = mismatches > 0 ? (float)sum_diff / mismatches : 0.0f;

#ifndef MINIMAL_OUTPUT
    printf("CL: %s validation: %.2f%% error (mismatches=%d, max_diff=%d, mean_diff=%.2f)\n",
           layer_name, error_rate, mismatches, max_diff, mean_diff);
#endif
#endif
}
