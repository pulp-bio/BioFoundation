/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// DMA pipeline interface for tiled L3→L2→L1 data movement.

#ifndef NETWORK_DMA_PIPELINE_H
#define NETWORK_DMA_PIPELINE_H

#include <stdint.h>
#include <stddef.h>
#include "pmsis.h"
#include "bsp/ram.h"

#ifdef ENABLE_PERF_COUNTERS
#include "network_kernels.h"
#endif

// Data Layout for tensor operations
typedef enum {
    LAYOUT_CHW = 0,   // Channel-Height-Width (default, PyTorch-style)
    LAYOUT_HWC = 1    // Height-Width-Channel (SIMD-friendly for small channels)
} TensorLayout;

// Generic Tile Mover types
typedef enum { MEM_LOC_L3 = 0, MEM_LOC_L2 = 1, MEM_LOC_L1 = 2 } dma_mem_loc_t;

typedef struct {
    void *base_addr;
    uint32_t width;        // Changed from uint16_t to support L3 transfers >65KB
    uint32_t height;       // Changed from uint16_t for consistency
    uint32_t channels;     // Changed from uint16_t for consistency
    uint32_t stride_row;
    uint32_t stride_channel;
    dma_mem_loc_t loc;
    uint8_t is_async;
    pi_cl_dma_copy_t *dma_cmd;
    pi_cl_ram_req_t *ram_cmd;
    struct pi_device *ram_dev; // HyperRAM Device Handle
} dma_layout_t;

void execute_dma_transfer(dma_layout_t *src, dma_layout_t *dst);

// MHSA Pooling Modes
#define MHSA_POOL_NONE 0
#define MHSA_POOL_MEAN 1
#define MHSA_POOL_FLAT 2

// --- Pipeline Configurations ---

typedef struct {
    const char *layer_name;
    int8_t *input_buffer_l2;
    int8_t *output_buffer_l2;
    int8_t *weight_l2;
    int32_t *bias_l2;
    int8_t *l1_buffer;
    size_t l1_buffer_size;
    int in_h, in_w, in_ch;
    int out_h, out_w, out_ch;
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int groups;                   // 1=standard conv, >1=grouped/depthwise conv
    int tile_h, tile_w;
    int tile_h_halo, tile_w_halo;
    int num_tiles, num_tiles_h, num_tiles_w;
    size_t l1_input_size, l1_output_size;
    int out_tile_h, out_tile_w;
    float scale_input, scale_weight, scale_output;
    int fusion_relu;
    int fusion_quant;
    float quant_scale_in;
    float quant_scale_out;

    // Conv+MaxPool Fusion: fuse pooling into conv for L1 efficiency
    int fusion_maxpool;           // True if MaxPool is fused after conv+relu
    int pool_kernel_h, pool_kernel_w;  // Pool kernel size
    int pool_stride_h, pool_stride_w;  // Pool stride
    int pool_out_h, pool_out_w;        // Final output dimensions after pool
    int8_t *fused_output_buffer_l2;    // Pool's output buffer (different from conv output)

#ifdef ENABLE_PERF_COUNTERS
    layer_perf_t *perf_counter;
#endif
    struct pi_device *cluster_dev;
    struct pi_device *ram_dev; // [NEW] HyperRAM Device
    int8_t *golden_buffer;
    size_t golden_size;
    int8_t *compare_buffer;

    // L3 Tiling
    int l3_tiling_enabled;
    void *l3_input_addr;
    void *l3_output_addr;
    int l3_tile_h;
    int l3_tile_h_halo;
    int num_l3_tiles;

    // L1 weight caching
    int weight_tiling_enabled;    // True if weights should be cached in L1
    int tile_out_ch;              // Output channels per weight tile
    int num_out_ch_tiles;         // Number of output channel tiles
    size_t l1_weight_size;        // Weight tile size in bytes

    // Triple-buffer weight pipeline
    // Eliminates blocking wait on first weight load by using 3 weight buffers
    // Only enabled when num_out_ch_tiles >= 3 (need enough tiles to pipeline)
    int triple_buffer_weights;    // True if using 3-buffer weight pipeline

    // Data Layout (CHW vs HWC)
    TensorLayout layout;          // LAYOUT_CHW (default) or LAYOUT_HWC
} conv2d_pipeline_config_t;

typedef struct {
    const char *layer_name;
    int8_t *input_buffer_l2;
    int8_t *output_buffer_l2;
    int8_t *weight_l2;
    int32_t *bias_l2;
    int8_t *l1_buffer;
    size_t l1_buffer_size;
    int in_features;
    int out_features;
    int batch_tokens;  // Number of tokens to process (for 3D linear: seq_len)
    int tile_out_features;
    int num_tiles;

    // K-dimension tiling (input feature tiling)
    int tile_in_features;     // Input features per K-tile (0 or in_features = no K-tiling)
    int num_k_tiles;          // Number of input feature tiles
    int k_tiling_enabled;     // Flag: 1 if K-tiling is active

    // M-dimension tiling (batch/token tiling for 3D linear)
    int tile_batch_tokens;    // Tokens per M-tile (0 or batch_tokens = no M-tiling)
    int num_m_tiles;          // Number of token tiles
    int m_tiling_enabled;     // Flag: 1 if M-tiling is active

    size_t l1_input_size, l1_output_size, l1_weight_size;
    float scale_input, scale_weight, scale_output;
    int fusion_relu;
    int fusion_quant;
    float relu_output_scale;  // ReLU's advertised output scale (for intermediate rescale)
    float quant_scale_in;
    float quant_scale_out;
#ifdef ENABLE_PERF_COUNTERS
    layer_perf_t *perf_counter;
#endif
    int8_t *golden_buffer;
    size_t golden_size;
    int8_t *compare_buffer;
    struct pi_device *ram_dev;

    // L3 Tiling
    int l3_tiling_enabled;
    void *l3_weight_addr;
    void *l3_output_addr;
    void *l3_bias_addr;
    int l3_tile_out_features;
    int num_l3_tiles;
} linear_int8_pipeline_config_t;

typedef struct {
    const char *layer_name;
    int8_t *input_buffer_l2;
    float *output_buffer_l2;
    int8_t *weight_l2;
    float *bias_l2;
    int8_t *l1_buffer;
    size_t l1_buffer_size;
    int in_features;
    int out_features;
    int tile_out_features;
    int num_tiles;
    size_t l1_input_size, l1_output_size, l1_weight_size;
    float scale_input, scale_weight;
#ifdef ENABLE_PERF_COUNTERS
    layer_perf_t *perf_counter;
#endif
    struct pi_device *ram_dev;

    // L3 Tiling
    int l3_tiling_enabled;
    void *l3_weight_addr;
    void *l3_output_addr;
    void *l3_bias_addr;
    int l3_tile_out_features;
    int num_l3_tiles;
} linear_fp32_pipeline_config_t;

typedef struct {
    const char *layer_name;
    int8_t *input_buffer_l2;
    int8_t *q_buffer_l2;
    int8_t *k_buffer_l2;
    int8_t *v_buffer_l2;
    int8_t *output_buffer_l2;
    int8_t *l1_buffer;
    size_t l1_buffer_size;
    
    // Projection Weights (L2 source)
    int8_t *q_weight_l2; int32_t *q_bias_l2;
    int8_t *k_weight_l2; int32_t *k_bias_l2;
    int8_t *v_weight_l2; int32_t *v_bias_l2;
    int8_t *out_weight_l2; int32_t *out_bias_l2;

    // L1 Weight Caching (for projection optimization)
    int l1_weight_caching_enabled;  // True if projection weights fit in L1
    size_t l1_proj_weight_size;     // Size of one projection weight matrix (embed_dim * embed_dim)

    int seq_len;
    int num_heads;
    int n_kv_heads;           // Number of KV heads for GQA (0 or == num_heads means standard MHA)
    int head_dim;
    int embed_dim;
    int tile_q;
    int num_tiles;
    size_t persistent_bytes;
    size_t tile_bytes;
    uint8_t pool_mode;
    uint8_t use_fp32_projections;

    // Scales
    float scale_input;
    float scale_q_weight, scale_k_weight, scale_v_weight, scale_out_weight;
    float scale_q, scale_k, scale_v, scale_output;
    float softmax_scale;

    // i-Softmax LUT for bit-exact matching (NULL = use fast_exp FP32 path)
    const int16_t *softmax_lut;

    // RoPE (Rotary Position Embeddings) tables (Q15). Applied to Q/K after projection when enabled.
    int use_rope;                 // 1 = apply RoPE, 0 = disabled
    const int16_t *rope_cos_q15;  // [seq_len, head_dim/2]
    const int16_t *rope_sin_q15;  // [seq_len, head_dim/2]
    int rope_pos_offset;          // Position offset for RoPE (for autoregressive with KV cache)

    // KV Cache for autoregressive generation (Llama-style)
    int kv_cache_enabled;         // 1 = use KV cache, 0 = standard attention
    int8_t *kv_cache_k;           // Cached K: [n_kv_heads, max_seq_len, head_dim]
    int8_t *kv_cache_v;           // Cached V: [n_kv_heads, max_seq_len, head_dim]
    float kv_cache_scale_k;       // Scale for cached K values
    float kv_cache_scale_v;       // Scale for cached V values
    int kv_cache_pos;             // Current position in cache (number of tokens so far)
    int kv_cache_max_seq_len;     // Maximum sequence length for cache

    // Fully-integer softmax configuration (LUT-based)
    int use_integer_softmax;        // 1 = LUT-based iSoftmax, 0 = FP32 fast_exp
    int32_t requant_mul;            // QK requantization multiplier (for shift=24)
    int32_t requant_shift;          // QK requantization shift (typically 24)
    int32_t isoftmax_coeffA;        // Polynomial coefficient A
    int32_t isoftmax_coeffB;        // Polynomial coefficient B
    int32_t isoftmax_coeffC;        // Polynomial coefficient C
    int32_t isoftmax_log2;          // ln(2) in fixed point for range reduction
    uint32_t isoftmax_n_levels;     // Output levels (256 for UINT8)

    // Head-contiguous optimization (bulk DMA per head)
    int use_head_contiguous;        // 1 = permute Q/K/V to [num_heads, seq_len, head_dim]
    int use_inplace_permute;        // 1 = use output buffer as scratch for in-place permutation
    int8_t *q_permuted_l2;          // Permuted Q buffer [num_heads, seq_len, head_dim]
    int8_t *k_permuted_l2;          // Permuted K buffer [num_heads, seq_len, head_dim]
    int8_t *v_permuted_l2;          // Permuted V buffer [num_heads, seq_len, head_dim]
    int8_t *m_permuted_l2;          // Permuted output buffer [num_heads, seq_len, head_dim]
    int8_t *permute_scratch_l2;     // Scratch buffer for in-place permutation (= output buffer)

    // Inner loop cycle tracking (QK, Softmax, AV breakdown)
    unsigned int inner_qk_cycles;       // Accumulated QK kernel cycles
    unsigned int inner_softmax_cycles;  // Accumulated Softmax kernel cycles
    unsigned int inner_av_cycles;       // Accumulated AV kernel cycles

#ifdef ENABLE_PERF_COUNTERS
    layer_perf_t *perf_counter;
#endif
    int8_t *golden_buffer;
    size_t golden_size;
    int8_t *compare_buffer;
    struct pi_device *ram_dev; // [NEW] HyperRAM Device

    // L3 Tiling
    int l3_tiling_enabled;
    void *l3_input_addr;
    void *l3_output_addr;
    int l3_seq_len;
    int num_l3_tiles;
} mhsa_pipeline_config_t;

void conv2d_tiled_l1_pipeline(conv2d_pipeline_config_t *cfg);
void linear_int8_tiled_l1_pipeline(linear_int8_pipeline_config_t *cfg);
void linear_fp32_tiled_l1_pipeline(linear_fp32_pipeline_config_t *cfg);
void mhsa_tiled_l1_pipeline(mhsa_pipeline_config_t *cfg);

// KV Cache operations for autoregressive generation (GQA/Llama-style)
void mhsa_kv_cache_store(
    mhsa_pipeline_config_t *cfg,
    const int8_t *k_projected,
    const int8_t *v_projected,
    int cache_pos
);
int mhsa_kv_cache_retrieve(
    mhsa_pipeline_config_t *cfg,
    int8_t *k_out,
    int8_t *v_out,
    int cache_pos
);

// ---
// Standardized L3 Streaming Helpers
// ---
//
// These helpers provide a consistent API for L3<->L2 DMA transfers.
//
// Usage Example (double-buffered weight streaming):
//
//   l3_double_buffer_t db;
//   l3_stream_req_t req;
//   l3_double_buffer_init(&db, ping_buf, pong_buf, chunk_size);
//
//   // Prefetch first chunk
//   l3_stream_async_start(&req, ram_dev, l3_double_buffer_active(&db), l3_weights, chunk_size);
//   l3_stream_async_wait(&req);
//
//   for (int i = 0; i < num_chunks; i++) {
//       void *compute_buf = l3_double_buffer_active(&db);
//       void *prefetch_buf = l3_double_buffer_inactive(&db);
//
//       // Start prefetch of next chunk while computing current
//       if (i + 1 < num_chunks) {
//           l3_stream_async_start(&req, ram_dev, prefetch_buf,
//                                  l3_weights + (i+1)*chunk_size, chunk_size);
//       }
//
//       // Process current chunk
//       compute_on_data(compute_buf);
//
//       // Wait for prefetch before swapping
//       if (i + 1 < num_chunks) {
//           l3_stream_async_wait(&req);
//       }
//       l3_double_buffer_swap(&db);
//   }
//

typedef enum {
    DMA_STREAM_DIR_L3_TO_L2 = 0,
    DMA_STREAM_DIR_L2_TO_L3 = 1
} dma_stream_direction_t;

typedef enum {
    DMA_WAIT_POLICY_PER_TRANSFER = 0,
    DMA_WAIT_POLICY_DIRECTION = 1,
    DMA_WAIT_POLICY_BARRIER = 2
} dma_wait_policy_t;

typedef struct {
    struct pi_device *ram_dev;
    void *l3_addr;
    void *l2_addr;
    size_t bytes;
    dma_stream_direction_t direction;
} dma_async_request_t;

// Double buffer state for ping-pong L3 streaming
typedef struct {
    void *ping;              // First buffer (L2)
    void *pong;              // Second buffer (L2)
    size_t buffer_size;      // Size of each buffer in bytes
    int active_buffer;       // 0 = ping active, 1 = pong active
} l3_double_buffer_t;

// Async L3 streaming request handle
typedef struct {
    pi_cl_ram_req_t req;     // DMA request handle
    void *dst;               // Destination buffer (L2 for reads, L3 for writes)
    void *src;               // Source address (L3 for reads, L2 for writes)
    size_t bytes;            // Transfer size
    dma_stream_direction_t direction;
    int pending;             // 1 = transfer in progress, 0 = complete/idle
} l3_stream_req_t;

typedef struct {
    l3_stream_req_t transfer;
    pi_cl_dma_copy_t l2_copy;
    int l2_pending;
    int valid;
} dma_async_future_t;

// Initialize double buffer structure
static inline void l3_double_buffer_init(l3_double_buffer_t *db, void *ping, void *pong, size_t size) {
    if (db == NULL) return;
    db->ping = ping;
    db->pong = pong;
    db->buffer_size = size;
    db->active_buffer = 0;
}

// Get the currently active buffer
static inline void *l3_double_buffer_active(l3_double_buffer_t *db) {
    if (db == NULL) return NULL;
    return db->active_buffer ? db->pong : db->ping;
}

// Get the inactive buffer (for prefetching)
static inline void *l3_double_buffer_inactive(l3_double_buffer_t *db) {
    if (db == NULL) return NULL;
    return db->active_buffer ? db->ping : db->pong;
}

// Swap active/inactive buffers
static inline void l3_double_buffer_swap(l3_double_buffer_t *db) {
    if (db == NULL) return;
    db->active_buffer = !db->active_buffer;
}

// Start async L3->L2 transfer (non-blocking)
static inline void l3_stream_async_start(l3_stream_req_t *handle, struct pi_device *ram_dev,
                                          void *l2_dst, void *l3_src, size_t bytes) {
    if (handle == NULL) return;
    handle->dst = l2_dst;
    handle->src = l3_src;
    handle->bytes = bytes;
    handle->direction = DMA_STREAM_DIR_L3_TO_L2;
    handle->pending = 0;
    if (bytes == 0 || ram_dev == NULL) return;
    handle->pending = 1;
    pi_cl_ram_read(ram_dev, (uint32_t)l3_src, l2_dst, bytes, &handle->req);
}

// Wait for async L3->L2 transfer to complete (blocking)
static inline void l3_stream_async_wait(l3_stream_req_t *handle) {
    if (handle == NULL) return;
    if (handle->pending && handle->bytes > 0) {
        pi_cl_ram_read_wait(&handle->req);
        handle->pending = 0;
    }
}

// Synchronous L3->L2 transfer (blocking convenience wrapper)
static inline void l3_stream_sync(struct pi_device *ram_dev, void *l2_dst, void *l3_src, size_t bytes) {
    l3_stream_req_t req;
    l3_stream_async_start(&req, ram_dev, l2_dst, l3_src, bytes);
    l3_stream_async_wait(&req);
}

// Start async L2->L3 write (non-blocking)
static inline void l3_store_async_start(l3_stream_req_t *handle, struct pi_device *ram_dev,
                                         void *l3_dst, void *l2_src, size_t bytes) {
    if (handle == NULL) return;
    handle->dst = l3_dst;
    handle->src = l2_src;
    handle->bytes = bytes;
    handle->direction = DMA_STREAM_DIR_L2_TO_L3;
    handle->pending = 0;
    if (bytes == 0 || ram_dev == NULL) return;
    handle->pending = 1;
    pi_cl_ram_write(ram_dev, (uint32_t)l3_dst, l2_src, bytes, &handle->req);
}

// Wait for async L2->L3 write to complete (blocking)
static inline void l3_store_async_wait(l3_stream_req_t *handle) {
    if (handle == NULL) return;
    if (handle->pending && handle->bytes > 0) {
        pi_cl_ram_write_wait(&handle->req);
        handle->pending = 0;
    }
}

// Synchronous L2->L3 write (blocking convenience wrapper)
static inline void l3_store_sync(struct pi_device *ram_dev, void *l3_dst, void *l2_src, size_t bytes) {
    l3_stream_req_t req;
    l3_store_async_start(&req, ram_dev, l3_dst, l2_src, bytes);
    l3_store_async_wait(&req);
}

// ---
// DMA async contract scaffold (request/future/wait-policy)
// ---

static inline void dma_async_request_reset(dma_async_request_t *request) {
    if (request == NULL) return;
    request->ram_dev = NULL;
    request->l3_addr = NULL;
    request->l2_addr = NULL;
    request->bytes = 0;
    request->direction = DMA_STREAM_DIR_L3_TO_L2;
}

static inline void dma_async_future_reset(dma_async_future_t *future) {
    if (future == NULL) return;
    future->transfer.dst = NULL;
    future->transfer.src = NULL;
    future->transfer.bytes = 0;
    future->transfer.direction = DMA_STREAM_DIR_L3_TO_L2;
    future->transfer.pending = 0;
    future->l2_copy.size = 0;
    future->l2_pending = 0;
    future->valid = 0;
}

static inline int dma_async_request_make_l3_read(
    dma_async_request_t *request,
    struct pi_device *ram_dev,
    void *l2_dst,
    void *l3_src,
    size_t bytes
) {
    if (request == NULL) return -1;
    dma_async_request_reset(request);
    request->ram_dev = ram_dev;
    request->l3_addr = l3_src;
    request->l2_addr = l2_dst;
    request->bytes = bytes;
    request->direction = DMA_STREAM_DIR_L3_TO_L2;
    if (bytes == 0) return 0;
    if (ram_dev == NULL || l3_src == NULL || l2_dst == NULL) return -1;
    return 0;
}

static inline int dma_async_request_make_l3_write(
    dma_async_request_t *request,
    struct pi_device *ram_dev,
    void *l3_dst,
    void *l2_src,
    size_t bytes
) {
    if (request == NULL) return -1;
    dma_async_request_reset(request);
    request->ram_dev = ram_dev;
    request->l3_addr = l3_dst;
    request->l2_addr = l2_src;
    request->bytes = bytes;
    request->direction = DMA_STREAM_DIR_L2_TO_L3;
    if (bytes == 0) return 0;
    if (ram_dev == NULL || l3_dst == NULL || l2_src == NULL) return -1;
    return 0;
}

static inline int dma_async_submit(dma_async_future_t *future, const dma_async_request_t *request) {
    if (future == NULL || request == NULL) return -1;
    if (request->bytes > 0 && (request->ram_dev == NULL || request->l3_addr == NULL || request->l2_addr == NULL)) {
        return -1;
    }

    dma_async_future_reset(future);
    future->valid = 1;

    if (request->direction == DMA_STREAM_DIR_L3_TO_L2) {
        l3_stream_async_start(&future->transfer, request->ram_dev, request->l2_addr, request->l3_addr, request->bytes);
    } else {
        l3_store_async_start(&future->transfer, request->ram_dev, request->l3_addr, request->l2_addr, request->bytes);
    }
    return 0;
}

static inline int dma_async_wait(
    dma_async_future_t *future,
    dma_wait_policy_t wait_policy,
    dma_stream_direction_t direction_filter
) {
    if (future == NULL || !future->valid) return -1;

    if (future->l2_pending) {
        pi_cl_dma_wait(&future->l2_copy);
        future->l2_pending = 0;
        return 0;
    }

    if (wait_policy == DMA_WAIT_POLICY_DIRECTION && future->transfer.direction != direction_filter) {
        return 0;
    }

    if (future->transfer.direction == DMA_STREAM_DIR_L3_TO_L2) {
        l3_stream_async_wait(&future->transfer);
    } else {
        l3_store_async_wait(&future->transfer);
    }
    return 0;
}

static inline int dma_async_wait_barrier(dma_async_future_t *futures, size_t count) {
    if (futures == NULL && count > 0) return -1;
    for (size_t i = 0; i < count; ++i) {
        if (dma_async_wait(&futures[i], DMA_WAIT_POLICY_BARRIER, DMA_STREAM_DIR_L3_TO_L2) != 0) {
            return -1;
        }
    }
    return 0;
}

static inline int dma_async_poll(const dma_async_future_t *future) {
    if (future == NULL || !future->valid) return -1;
    if (future->l2_pending) return 0;
    return future->transfer.pending ? 0 : 1;
}

// Compatibility wrappers for the previous start/wait/sync call shape.
static inline int dma_async_compat_l3_stream_start(
    l3_stream_req_t *handle,
    struct pi_device *ram_dev,
    void *l2_dst,
    void *l3_src,
    size_t bytes
) {
    if (handle == NULL) return -1;
    l3_stream_async_start(handle, ram_dev, l2_dst, l3_src, bytes);
    return 0;
}

static inline int dma_async_compat_l3_stream_wait(l3_stream_req_t *handle) {
    if (handle == NULL) return -1;
    l3_stream_async_wait(handle);
    return 0;
}

static inline int dma_async_compat_l3_store_start(
    l3_stream_req_t *handle,
    struct pi_device *ram_dev,
    void *l3_dst,
    void *l2_src,
    size_t bytes
) {
    if (handle == NULL) return -1;
    l3_store_async_start(handle, ram_dev, l3_dst, l2_src, bytes);
    return 0;
}

static inline int dma_async_compat_l3_store_wait(l3_stream_req_t *handle) {
    if (handle == NULL) return -1;
    l3_store_async_wait(handle);
    return 0;
}

static inline int dma_async_compat_l2_copy_start(
    dma_async_future_t *future,
    const pi_cl_dma_copy_t *copy
) {
    if (future == NULL || copy == NULL) return -1;
    dma_async_future_reset(future);
    future->valid = 1;
    future->l2_copy = *copy;
    if (future->l2_copy.size == 0) return 0;
    pi_cl_dma_memcpy(&future->l2_copy);
    future->l2_pending = 1;
    return 0;
}

static inline int dma_async_compat_l2_copy_wait(dma_async_future_t *future) {
    if (future == NULL || !future->valid) return -1;
    if (future->l2_pending) {
        pi_cl_dma_wait(&future->l2_copy);
        future->l2_pending = 0;
    }
    return 0;
}

static inline int dma_async_compat_l2_copy_sync(const pi_cl_dma_copy_t *copy) {
    dma_async_future_t future;
    if (dma_async_compat_l2_copy_start(&future, copy) != 0) return -1;
    return dma_async_compat_l2_copy_wait(&future);
}

#endif // NETWORK_DMA_PIPELINE_H
