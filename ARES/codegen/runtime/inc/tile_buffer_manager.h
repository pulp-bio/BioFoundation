/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/* Double-buffer (ping-pong) management for L1 tiled execution. */
/*
 * Tile Buffer Manager - Double-Buffer Abstraction
 * 
 * Provides explicit buffer management for L1 double-buffering with ping-pong
 * pattern. Supports multiple buffer layouts (Conv2D, Linear, GlobalAvgPool).
 * 
 * Key principles:
 * - Explicit ownership: Buffer pointers are always explicit, never hidden
 * - No hidden state: Current buffer selection is visible via current_idx
 * - Type-safe: Separate functions for input/output/weight buffers
 */

#ifndef TILE_BUFFER_MANAGER_H
#define TILE_BUFFER_MANAGER_H

#include <stdint.h>
#include <stddef.h>

/*
 * Buffer manager for operations with separate input/output buffers
 * Layout: [input_A + output_A | input_B + output_B]
 * Used by: Conv2D, MaxPool, AvgPool
 */
typedef struct {
    int8_t *input_a;      // Buffer A input pointer
    int8_t *output_a;     // Buffer A output pointer
    int8_t *input_b;      // Buffer B input pointer
    int8_t *output_b;     // Buffer B output pointer
    int current_idx;      // Current buffer index (0=A, 1=B)
} tile_buffer_mgr_t;

/*
 * Buffer manager for Linear operations with shared input
 * Layout: [input (shared) | output_A + weights_A | output_B + weights_B]
 * Used by: Linear INT8, Linear FP32
 */
typedef struct {
    int8_t *input;        // Shared input buffer (used by all tiles)
    int8_t *output_a;     // Buffer A output pointer
    int8_t *weights_a;    // Buffer A weights pointer
    int8_t *output_b;     // Buffer B output pointer
    int8_t *weights_b;    // Buffer B weights pointer
    int current_idx;      // Current buffer index (0=A, 1=B)
} tile_buffer_linear_mgr_t;

/*
 * Buffer manager for GlobalAvgPool with double input, shared partial sums
 * Layout: [input_A | input_B | partial_sums (shared)]
 * Used by: GlobalAvgPool
 */
typedef struct {
    int8_t *input_a;      // Buffer A input pointer
    int8_t *input_b;      // Buffer B input pointer
    int32_t *partial_sums; // Shared partial sums (accumulates across all tiles)
    int current_idx;      // Current buffer index (0=A, 1=B)
} tile_buffer_global_avgpool_mgr_t;

/*
 * Buffer manager for Conv2D with L1 weight caching
 * Layout: [input_A + output_A + weights_A | input_B + output_B + weights_B]
 * Used by: Conv2D with weight tiling enabled
 *
 * This enables weights to be cached in L1 alongside input/output tiles,
 * reducing L2 access latency (10x) to L1 latency (1x) for weight reads.
 * Weights are tiled by output channel dimension.
 */
typedef struct {
    int8_t *input_a;      // Buffer A input tile pointer
    int8_t *output_a;     // Buffer A output tile pointer
    int8_t *weights_a;    // Buffer A weight tile pointer
    int8_t *input_b;      // Buffer B input tile pointer
    int8_t *output_b;     // Buffer B output tile pointer
    int8_t *weights_b;    // Buffer B weight tile pointer
    int current_idx;      // Current buffer index (0=A, 1=B)
} tile_buffer_conv2d_weight_mgr_t;

/*
 * Initialize buffer manager for Conv2D/MaxPool/AvgPool
 * 
 * @param mgr          Buffer manager to initialize
 * @param l1_base      Base L1 buffer pointer
 * @param input_size   Size of input buffer in bytes
 * @param output_size  Size of output buffer in bytes
 */
static inline void tile_buffer_init(
    tile_buffer_mgr_t *mgr,
    int8_t *l1_base,
    size_t input_size,
    size_t output_size)
{
    size_t single_buffer_size = input_size + output_size;
    
    mgr->input_a = l1_base;
    mgr->output_a = l1_base + input_size;
    mgr->input_b = l1_base + single_buffer_size;
    mgr->output_b = l1_base + single_buffer_size + input_size;
    mgr->current_idx = 0;
}

/*
 * Initialize buffer manager for Linear operations
 * 
 * @param mgr          Buffer manager to initialize
 * @param l1_base      Base L1 buffer pointer
 * @param input_size   Size of shared input buffer in bytes
 * @param output_size  Size of output buffer in bytes
 * @param weight_size  Size of weight buffer in bytes
 */
static inline void tile_buffer_linear_init(
    tile_buffer_linear_mgr_t *mgr,
    int8_t *l1_base,
    size_t input_size,
    size_t output_size,
    size_t weight_size)
{
    mgr->input = l1_base;
    mgr->output_a = l1_base + input_size;
    mgr->weights_a = mgr->output_a + output_size;
    mgr->output_b = mgr->weights_a + weight_size;
    mgr->weights_b = mgr->output_b + output_size;
    mgr->current_idx = 0;
}

/*
 * Initialize buffer manager for GlobalAvgPool
 * 
 * @param mgr          Buffer manager to initialize
 * @param l1_base      Base L1 buffer pointer
 * @param input_size   Size of input buffer in bytes
 * @param partial_sum_size Size of partial sum buffer in bytes
 */
static inline void tile_buffer_global_avgpool_init(
    tile_buffer_global_avgpool_mgr_t *mgr,
    int8_t *l1_base,
    size_t input_size,
    size_t partial_sum_size)
{
    mgr->input_a = l1_base;
    mgr->input_b = l1_base + input_size;
    mgr->partial_sums = (int32_t *)(l1_base + 2 * input_size);
    mgr->current_idx = 0;
}

/*
 * Initialize buffer manager for Conv2D with L1 weight caching
 *
 * @param mgr          Buffer manager to initialize
 * @param l1_base      Base L1 buffer pointer
 * @param input_size   Size of input tile buffer in bytes
 * @param output_size  Size of output tile buffer in bytes
 * @param weight_size  Size of weight tile buffer in bytes
 *
 * Memory layout (double-buffered):
 * [input_A][output_A][weights_A][input_B][output_B][weights_B]
 */
static inline void tile_buffer_conv2d_weight_init(
    tile_buffer_conv2d_weight_mgr_t *mgr,
    int8_t *l1_base,
    size_t input_size,
    size_t output_size,
    size_t weight_size)
{
    size_t single_buffer_size = input_size + output_size + weight_size;

    // Buffer A
    mgr->input_a = l1_base;
    mgr->output_a = l1_base + input_size;
    mgr->weights_a = l1_base + input_size + output_size;

    // Buffer B
    mgr->input_b = l1_base + single_buffer_size;
    mgr->output_b = l1_base + single_buffer_size + input_size;
    mgr->weights_b = l1_base + single_buffer_size + input_size + output_size;

    mgr->current_idx = 0;
}

/*
 * Get current input buffer pointer
 */
static inline int8_t* tile_buffer_get_input(const tile_buffer_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_a : mgr->input_b;
}

/*
 * Get current output buffer pointer
 */
static inline int8_t* tile_buffer_get_output(const tile_buffer_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_a : mgr->output_b;
}

/*
 * Get next input buffer pointer (for prefetching)
 */
static inline int8_t* tile_buffer_get_next_input(const tile_buffer_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_b : mgr->input_a;
}

/*
 * Get next output buffer pointer (for prefetching)
 */
static inline int8_t* tile_buffer_get_next_output(const tile_buffer_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_b : mgr->output_a;
}

/*
 * Get current input buffer pointer for GlobalAvgPool
 */
static inline int8_t* tile_buffer_global_avgpool_get_input(const tile_buffer_global_avgpool_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_a : mgr->input_b;
}

/*
 * Get current output buffer pointer for Linear
 */
static inline int8_t* tile_buffer_linear_get_output(const tile_buffer_linear_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_a : mgr->output_b;
}

/*
 * Get current weight buffer pointer for Linear
 */
static inline int8_t* tile_buffer_linear_get_weights(const tile_buffer_linear_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->weights_a : mgr->weights_b;
}

/*
 * Get next output buffer pointer for Linear (for prefetching)
 */
static inline int8_t* tile_buffer_linear_get_next_output(const tile_buffer_linear_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_b : mgr->output_a;
}

/*
 * Get next weight buffer pointer for Linear (for prefetching)
 */
static inline int8_t* tile_buffer_linear_get_next_weights(const tile_buffer_linear_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->weights_b : mgr->weights_a;
}

/*
 * Get next input buffer pointer for GlobalAvgPool (for prefetching)
 */
static inline int8_t* tile_buffer_global_avgpool_get_next_input(const tile_buffer_global_avgpool_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_b : mgr->input_a;
}

/*
 * Swap buffers (A ↔ B)
 * Call after processing each tile to toggle between buffers
 */
static inline void tile_buffer_swap(tile_buffer_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

static inline void tile_buffer_linear_swap(tile_buffer_linear_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

static inline void tile_buffer_global_avgpool_swap(tile_buffer_global_avgpool_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

/*
 * Conv2D with weight caching - Buffer accessors
 */

/* Get current input buffer pointer */
static inline int8_t* tile_buffer_conv2d_weight_get_input(const tile_buffer_conv2d_weight_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_a : mgr->input_b;
}

/* Get current output buffer pointer */
static inline int8_t* tile_buffer_conv2d_weight_get_output(const tile_buffer_conv2d_weight_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_a : mgr->output_b;
}

/* Get current weight buffer pointer */
static inline int8_t* tile_buffer_conv2d_weight_get_weights(const tile_buffer_conv2d_weight_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->weights_a : mgr->weights_b;
}

/* Get next input buffer pointer (for prefetching) */
static inline int8_t* tile_buffer_conv2d_weight_get_next_input(const tile_buffer_conv2d_weight_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_b : mgr->input_a;
}

/* Get next output buffer pointer (for prefetching) */
static inline int8_t* tile_buffer_conv2d_weight_get_next_output(const tile_buffer_conv2d_weight_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_b : mgr->output_a;
}

/* Get next weight buffer pointer (for prefetching) */
static inline int8_t* tile_buffer_conv2d_weight_get_next_weights(const tile_buffer_conv2d_weight_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->weights_b : mgr->weights_a;
}

/* Swap buffers (A <-> B) */
static inline void tile_buffer_conv2d_weight_swap(tile_buffer_conv2d_weight_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

/*
 * ---
 * Conv2D Triple-Weight Buffer Manager (3-buffer weight pipeline)
 * ---
 *
 * Extends the double-buffered Conv2D weight manager with a THIRD weight buffer
 * to eliminate the blocking wait on the first weight tile load.
 *
 * Standard double-buffer timeline:
 *   LOAD W0 (blocking) -> COMPUTE W0 + LOAD W1 -> COMPUTE W1 + LOAD W2 -> ...
 *   ^-- Cores IDLE here!
 *
 * Triple-buffer timeline:
 *   LOAD W0 + LOAD W1 (async) -> wait W0 -> COMPUTE W0 + LOAD W2 -> wait W1 -> COMPUTE W1 + LOAD W3 -> ...
 *   ^-- No blocking wait, cores start immediately with loaded data
 *
 * Memory layout (triple-buffered weights, double-buffered IO):
 *   [input_A][output_A][input_B][output_B][weights_0][weights_1][weights_2]
 *
 * Weight buffer roles (rotate each iteration):
 *   - compute_idx: Buffer being used for current computation
 *   - ready_idx:   Buffer with completed DMA, ready for next compute
 *   - loading_idx: Buffer with DMA in progress (prefetch for iteration+2)
 */
typedef struct {
    int8_t *input_a;       // Buffer A input tile pointer
    int8_t *output_a;      // Buffer A output tile pointer
    int8_t *input_b;       // Buffer B input tile pointer
    int8_t *output_b;      // Buffer B output tile pointer
    int8_t *weights[3];    // Triple-buffered weight pointers
    int compute_idx;       // Index of weight buffer being computed (0, 1, or 2)
    int io_idx;            // Current I/O buffer index (0=A, 1=B)
} tile_buffer_conv2d_triple_weight_mgr_t;

/*
 * Initialize buffer manager for Conv2D with triple-buffered weights
 *
 * @param mgr          Buffer manager to initialize
 * @param l1_base      Base L1 buffer pointer
 * @param input_size   Size of input tile buffer in bytes
 * @param output_size  Size of output tile buffer in bytes
 * @param weight_size  Size of weight tile buffer in bytes
 *
 * Memory layout:
 * [input_A][output_A][input_B][output_B][weights_0][weights_1][weights_2]
 */
static inline void tile_buffer_conv2d_triple_weight_init(
    tile_buffer_conv2d_triple_weight_mgr_t *mgr,
    int8_t *l1_base,
    size_t input_size,
    size_t output_size,
    size_t weight_size)
{
    // Double-buffered I/O
    mgr->input_a = l1_base;
    mgr->output_a = l1_base + input_size;
    mgr->input_b = l1_base + input_size + output_size;
    mgr->output_b = l1_base + 2 * input_size + output_size;

    // Triple-buffered weights (after I/O buffers)
    int8_t *weight_base = l1_base + 2 * (input_size + output_size);
    mgr->weights[0] = weight_base;
    mgr->weights[1] = weight_base + weight_size;
    mgr->weights[2] = weight_base + 2 * weight_size;

    mgr->compute_idx = 0;
    mgr->io_idx = 0;
}

/* Get current input buffer pointer */
static inline int8_t* tile_buffer_conv2d_triple_weight_get_input(
    const tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    return (mgr->io_idx == 0) ? mgr->input_a : mgr->input_b;
}

/* Get current output buffer pointer */
static inline int8_t* tile_buffer_conv2d_triple_weight_get_output(
    const tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    return (mgr->io_idx == 0) ? mgr->output_a : mgr->output_b;
}

/* Get next input buffer pointer (for prefetching) */
static inline int8_t* tile_buffer_conv2d_triple_weight_get_next_input(
    const tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    return (mgr->io_idx == 0) ? mgr->input_b : mgr->input_a;
}

/* Get next output buffer pointer (for prefetching) */
static inline int8_t* tile_buffer_conv2d_triple_weight_get_next_output(
    const tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    return (mgr->io_idx == 0) ? mgr->output_b : mgr->output_a;
}

/* Get weight buffer for current computation */
static inline int8_t* tile_buffer_conv2d_triple_weight_get_compute(
    const tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    return mgr->weights[mgr->compute_idx];
}

/* Get weight buffer that's ready (DMA completed, for next iteration) */
static inline int8_t* tile_buffer_conv2d_triple_weight_get_ready(
    const tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    return mgr->weights[(mgr->compute_idx + 1) % 3];
}

/* Get weight buffer for loading (DMA target for prefetch) */
static inline int8_t* tile_buffer_conv2d_triple_weight_get_loading(
    const tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    return mgr->weights[(mgr->compute_idx + 2) % 3];
}

/* Advance weight buffers: compute -> ready -> loading -> compute */
static inline void tile_buffer_conv2d_triple_weight_advance(
    tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    mgr->compute_idx = (mgr->compute_idx + 1) % 3;
}

/* Swap I/O buffers (A <-> B) */
static inline void tile_buffer_conv2d_triple_weight_swap_io(
    tile_buffer_conv2d_triple_weight_mgr_t *mgr) {
    mgr->io_idx = 1 - mgr->io_idx;
}

/*
 * ---
 * Element-wise Operation Buffer Managers (ReLU, GELU, Requantize, Add, etc.)
 * ---
 */

/*
 * Buffer manager for in-place element-wise operations (ReLU, GELU)
 * Layout: [buffer_A | buffer_B]
 * Used by: ReLU, GELU, Requantize (in-place)
 *
 * These operations read from and write to the same buffer.
 * Double-buffering allows DMA and compute to overlap.
 */
typedef struct {
    int8_t *buffer_a;     // Buffer A pointer
    int8_t *buffer_b;     // Buffer B pointer
    int current_idx;      // Current buffer index (0=A, 1=B)
} tile_buffer_elementwise_mgr_t;

/*
 * Buffer manager for binary element-wise operations (Add)
 * Layout: [input1_A + input2_A | input1_B + input2_B]
 * Used by: Add (output written in-place to input1)
 *
 * Add operation reads two inputs and writes output (can be in-place to input1).
 */
typedef struct {
    int8_t *input1_a;     // Buffer A input1 pointer
    int8_t *input2_a;     // Buffer A input2 pointer
    int8_t *input1_b;     // Buffer B input1 pointer
    int8_t *input2_b;     // Buffer B input2 pointer
    int current_idx;      // Current buffer index (0=A, 1=B)
} tile_buffer_add_mgr_t;

/*
 * Initialize buffer manager for in-place element-wise operations
 *
 * @param mgr          Buffer manager to initialize
 * @param l1_base      Base L1 buffer pointer
 * @param tile_size    Size of each buffer in bytes
 */
static inline void tile_buffer_elementwise_init(
    tile_buffer_elementwise_mgr_t *mgr,
    int8_t *l1_base,
    size_t tile_size)
{
    mgr->buffer_a = l1_base;
    mgr->buffer_b = l1_base + tile_size;
    mgr->current_idx = 0;
}

/*
 * Initialize buffer manager for Add operation
 *
 * @param mgr          Buffer manager to initialize
 * @param l1_base      Base L1 buffer pointer
 * @param tile_size    Size of each input buffer in bytes
 */
static inline void tile_buffer_add_init(
    tile_buffer_add_mgr_t *mgr,
    int8_t *l1_base,
    size_t tile_size)
{
    // Layout: [input1_A][input2_A][input1_B][input2_B]
    mgr->input1_a = l1_base;
    mgr->input2_a = l1_base + tile_size;
    mgr->input1_b = l1_base + 2 * tile_size;
    mgr->input2_b = l1_base + 3 * tile_size;
    mgr->current_idx = 0;
}

/*
 * Get current buffer pointer for element-wise operations
 */
static inline int8_t* tile_buffer_elementwise_get(const tile_buffer_elementwise_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->buffer_a : mgr->buffer_b;
}

/*
 * Get next buffer pointer for element-wise operations (for prefetching)
 */
static inline int8_t* tile_buffer_elementwise_get_next(const tile_buffer_elementwise_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->buffer_b : mgr->buffer_a;
}

/*
 * Get current input1 buffer pointer for Add
 */
static inline int8_t* tile_buffer_add_get_input1(const tile_buffer_add_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input1_a : mgr->input1_b;
}

/*
 * Get current input2 buffer pointer for Add
 */
static inline int8_t* tile_buffer_add_get_input2(const tile_buffer_add_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input2_a : mgr->input2_b;
}

/*
 * Get next input1 buffer pointer for Add (for prefetching)
 */
static inline int8_t* tile_buffer_add_get_next_input1(const tile_buffer_add_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input1_b : mgr->input1_a;
}

/*
 * Get next input2 buffer pointer for Add (for prefetching)
 */
static inline int8_t* tile_buffer_add_get_next_input2(const tile_buffer_add_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input2_b : mgr->input2_a;
}

/*
 * Swap buffers for element-wise operations
 */
static inline void tile_buffer_elementwise_swap(tile_buffer_elementwise_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

/*
 * Swap buffers for Add operation
 */
static inline void tile_buffer_add_swap(tile_buffer_add_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

/*
 * ---
 * Concat Operation Buffer Manager
 * ---
 */

/*
 * Buffer manager for Concat operation
 * Layout: [data_A | data_B] where each data section contains spatial tile x all channels
 * Used by: Concat (channel-wise concatenation)
 *
 * Concat tiles spatially (HxW chunks), processing all channels per tile.
 * The buffer contains interleaved input tiles followed by output space.
 */
typedef struct {
    int8_t *buffer_a;     // Buffer A pointer (full spatial tile x total_channels)
    int8_t *buffer_b;     // Buffer B pointer (full spatial tile x total_channels)
    int spatial_tile_size; // Size of spatial tile (elements, not bytes)
    int total_channels;    // Sum of all input channels (= output channels)
    int current_idx;       // Current buffer index (0=A, 1=B)
} tile_buffer_concat_mgr_t;

/*
 * Initialize buffer manager for Concat operation
 *
 * @param mgr              Buffer manager to initialize
 * @param l1_base          Base L1 buffer pointer
 * @param spatial_tile_size Size of spatial tile (HxW chunk elements)
 * @param total_channels   Sum of all input channels
 */
static inline void tile_buffer_concat_init(
    tile_buffer_concat_mgr_t *mgr,
    int8_t *l1_base,
    size_t spatial_tile_size,
    size_t total_channels)
{
    size_t single_buffer_size = spatial_tile_size * total_channels;

    mgr->buffer_a = l1_base;
    mgr->buffer_b = l1_base + single_buffer_size;
    mgr->spatial_tile_size = spatial_tile_size;
    mgr->total_channels = total_channels;
    mgr->current_idx = 0;
}

/*
 * Get current buffer pointer for Concat
 */
static inline int8_t* tile_buffer_concat_get(const tile_buffer_concat_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->buffer_a : mgr->buffer_b;
}

/*
 * Get pointer to specific input's data within current Concat tile buffer
 *
 * @param mgr           Buffer manager
 * @param input_idx     Index of the input (0, 1, 2, ...)
 * @param channel_offset Cumulative channel offset for this input
 */
static inline int8_t* tile_buffer_concat_get_input(
    const tile_buffer_concat_mgr_t *mgr,
    int channel_offset)
{
    int8_t *base = (mgr->current_idx == 0) ? mgr->buffer_a : mgr->buffer_b;
    // Data layout: for each spatial position, channels are contiguous
    // So input_i data starts at base + channel_offset (for NCHW layout with spatial tiling)
    return base + channel_offset * mgr->spatial_tile_size;
}

/*
 * Swap buffers for Concat
 */
static inline void tile_buffer_concat_swap(tile_buffer_concat_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

/*
 * ---
 * LayerNorm Operation Buffer Manager
 * ---
 */

/*
 * Buffer manager for LayerNorm operation
 * Layout: [input_A + output_A | input_B + output_B | weight (gamma) | bias (beta)]
 * Used by: LayerNorm
 *
 * LayerNorm processes tokens in batches. Each token is normalized_dim elements.
 * Weights (gamma, beta) are loaded once and shared across all token batches.
 * Data buffers are double-buffered for DMA/compute overlap.
 */
typedef struct {
    int8_t *input_a;      // Buffer A input pointer
    int8_t *output_a;     // Buffer A output pointer
    int8_t *input_b;      // Buffer B input pointer
    int8_t *output_b;     // Buffer B output pointer
    float *weight;        // Shared weight (gamma) pointer
    float *bias;          // Shared bias (beta) pointer
    int current_idx;      // Current buffer index (0=A, 1=B)
} tile_buffer_layernorm_mgr_t;

/*
 * Initialize buffer manager for LayerNorm operation
 *
 * @param mgr              Buffer manager to initialize
 * @param l1_base          Base L1 buffer pointer
 * @param data_tile_size   Size of data tile in bytes (tokens_per_batch x normalized_dim)
 * @param normalized_dim   Dimension of weights (gamma/beta length)
 */
static inline void tile_buffer_layernorm_init(
    tile_buffer_layernorm_mgr_t *mgr,
    int8_t *l1_base,
    size_t data_tile_size,
    size_t normalized_dim)
{
    // Layout: [input_A][output_A][input_B][output_B][weight][bias]
    size_t single_data_size = 2 * data_tile_size;  // input + output per buffer
    size_t weight_size = normalized_dim * sizeof(float);

    mgr->input_a = l1_base;
    mgr->output_a = l1_base + data_tile_size;
    mgr->input_b = l1_base + single_data_size;
    mgr->output_b = l1_base + single_data_size + data_tile_size;
    mgr->weight = (float*)(l1_base + 2 * single_data_size);
    mgr->bias = (float*)(l1_base + 2 * single_data_size + weight_size);
    mgr->current_idx = 0;
}

/*
 * Get current input buffer pointer for LayerNorm
 */
static inline int8_t* tile_buffer_layernorm_get_input(const tile_buffer_layernorm_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_a : mgr->input_b;
}

/*
 * Get current output buffer pointer for LayerNorm
 */
static inline int8_t* tile_buffer_layernorm_get_output(const tile_buffer_layernorm_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_a : mgr->output_b;
}

/*
 * Get weight pointer for LayerNorm (shared across all tiles)
 */
static inline float* tile_buffer_layernorm_get_weight(const tile_buffer_layernorm_mgr_t *mgr) {
    return mgr->weight;
}

/*
 * Get bias pointer for LayerNorm (shared across all tiles)
 */
static inline float* tile_buffer_layernorm_get_bias(const tile_buffer_layernorm_mgr_t *mgr) {
    return mgr->bias;
}

/*
 * Swap buffers for LayerNorm
 */
static inline void tile_buffer_layernorm_swap(tile_buffer_layernorm_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

/*
 * ---
 * Transpose_2d Operation Buffer Manager
 * ---
 */

/*
 * Buffer manager for Transpose_2d operation
 * Layout: [input_A + output_A | input_B + output_B]
 * Used by: Transpose_2d ([B, D1, D2] -> [B, D2, D1])
 *
 * Transpose_2d tiles along D2 dimension. Each tile processes tile_d2 columns.
 * Input tile: [D1 x tile_d2] (contiguous in D2)
 * Output tile: [tile_d2 x D1] (strided write to output)
 */
typedef struct {
    int8_t *input_a;      // Buffer A input pointer
    int8_t *output_a;     // Buffer A output pointer
    int8_t *input_b;      // Buffer B input pointer
    int8_t *output_b;     // Buffer B output pointer
    int dim1;             // First dimension (D1)
    int tile_d2;          // Tile size in D2 dimension
    int current_idx;      // Current buffer index (0=A, 1=B)
} tile_buffer_transpose2d_mgr_t;

/*
 * Initialize buffer manager for Transpose_2d operation
 *
 * @param mgr          Buffer manager to initialize
 * @param l1_base      Base L1 buffer pointer
 * @param dim1         First dimension (D1)
 * @param tile_d2      Tile size in D2 dimension
 */
static inline void tile_buffer_transpose2d_init(
    tile_buffer_transpose2d_mgr_t *mgr,
    int8_t *l1_base,
    size_t dim1,
    size_t tile_d2)
{
    // Input tile: D1 x tile_d2
    // Output tile: tile_d2 x D1 (same size, different layout)
    size_t tile_size = dim1 * tile_d2;
    size_t single_buffer_size = 2 * tile_size;  // input + output

    mgr->input_a = l1_base;
    mgr->output_a = l1_base + tile_size;
    mgr->input_b = l1_base + single_buffer_size;
    mgr->output_b = l1_base + single_buffer_size + tile_size;
    mgr->dim1 = dim1;
    mgr->tile_d2 = tile_d2;
    mgr->current_idx = 0;
}

/*
 * Get current input buffer pointer for Transpose_2d
 */
static inline int8_t* tile_buffer_transpose2d_get_input(const tile_buffer_transpose2d_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_a : mgr->input_b;
}

/*
 * Get current output buffer pointer for Transpose_2d
 */
static inline int8_t* tile_buffer_transpose2d_get_output(const tile_buffer_transpose2d_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_a : mgr->output_b;
}

/*
 * Get next input buffer pointer for Transpose_2d (for prefetching)
 */
static inline int8_t* tile_buffer_transpose2d_get_next_input(const tile_buffer_transpose2d_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_b : mgr->input_a;
}

/*
 * Get next output buffer pointer for Transpose_2d (for prefetching)
 */
static inline int8_t* tile_buffer_transpose2d_get_next_output(const tile_buffer_transpose2d_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_b : mgr->output_a;
}

/*
 * Swap buffers for Transpose_2d
 */
static inline void tile_buffer_transpose2d_swap(tile_buffer_transpose2d_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

/*
 * ---
 * NE16 Depthwise 3x3 Spatial Tiling Buffer Manager
 * ---
 *
 * Buffer manager for NE16 depthwise 3x3 convolution with spatial (height) tiling.
 *
 * NE16 depthwise produces INT32 output that must be requantized to INT8 in SW.
 * When the full activation doesn't fit in L1, we tile along the height dimension.
 *
 * Memory layout (double-buffered input/output, shared weights):
 *   [input_A (U8)][output_A (S32)][input_B (U8)][output_B (S32)][weights][bias]
 *
 * Input tiles include halo for the 3x3 kernel (tile_h_in = tile_h_out + 2).
 * Weights and bias are loaded once and shared across all tiles.
 */
typedef struct {
    uint8_t *input_a;       // Buffer A: padded U8 input tile
    int32_t *output_a;      // Buffer A: INT32 output tile
    uint8_t *input_b;       // Buffer B: padded U8 input tile
    int32_t *output_b;      // Buffer B: INT32 output tile
    uint8_t *weights;       // Shared: packed depthwise weights
    int32_t *bias;          // Shared: corrected INT32 bias
    int current_idx;        // Current buffer index (0=A, 1=B)
} tile_buffer_ne16_dw_mgr_t;

/*
 * Initialize buffer manager for NE16 depthwise spatial tiling
 *
 * @param mgr              Buffer manager to initialize
 * @param l1_base          Base L1 buffer pointer
 * @param input_tile_size  Size of input tile in bytes (U8)
 * @param output_tile_size Size of output tile in bytes (S32)
 * @param weight_size      Size of packed weights in bytes
 * @param bias_size        Size of bias in bytes
 *
 * Memory layout:
 * [input_A][output_A][input_B][output_B][weights][bias]
 */
static inline void tile_buffer_ne16_dw_init(
    tile_buffer_ne16_dw_mgr_t *mgr,
    int8_t *l1_base,
    size_t input_tile_size,
    size_t output_tile_size,
    size_t weight_size,
    size_t bias_size)
{
    size_t single_io_size = input_tile_size + output_tile_size;

    // Buffer A
    mgr->input_a = (uint8_t *)l1_base;
    mgr->output_a = (int32_t *)(l1_base + input_tile_size);

    // Buffer B
    mgr->input_b = (uint8_t *)(l1_base + single_io_size);
    mgr->output_b = (int32_t *)(l1_base + single_io_size + input_tile_size);

    // Shared weights and bias (after double-buffered I/O)
    mgr->weights = (uint8_t *)(l1_base + 2 * single_io_size);
    mgr->bias = (int32_t *)(l1_base + 2 * single_io_size + weight_size);

    mgr->current_idx = 0;
}

/* Get current input buffer pointer */
static inline uint8_t* tile_buffer_ne16_dw_get_input(const tile_buffer_ne16_dw_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_a : mgr->input_b;
}

/* Get current output buffer pointer */
static inline int32_t* tile_buffer_ne16_dw_get_output(const tile_buffer_ne16_dw_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_a : mgr->output_b;
}

/* Get next input buffer pointer (for prefetching) */
static inline uint8_t* tile_buffer_ne16_dw_get_next_input(const tile_buffer_ne16_dw_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->input_b : mgr->input_a;
}

/* Get next output buffer pointer (for prefetching) */
static inline int32_t* tile_buffer_ne16_dw_get_next_output(const tile_buffer_ne16_dw_mgr_t *mgr) {
    return (mgr->current_idx == 0) ? mgr->output_b : mgr->output_a;
}

/* Get previous (now available for store) output buffer pointer */
static inline int32_t* tile_buffer_ne16_dw_get_prev_output(const tile_buffer_ne16_dw_mgr_t *mgr) {
    // After swap, previous output is the "next" buffer
    return (mgr->current_idx == 0) ? mgr->output_b : mgr->output_a;
}

/* Swap buffers (A <-> B) */
static inline void tile_buffer_ne16_dw_swap(tile_buffer_ne16_dw_mgr_t *mgr) {
    mgr->current_idx = 1 - mgr->current_idx;
}

#endif // TILE_BUFFER_MANAGER_H
