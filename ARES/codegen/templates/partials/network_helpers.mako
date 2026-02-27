/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/* ---
 * Helper Functions (validation, L1 fusion kernels)
 * --- */

% if mamba_block_entries:
<%
    # Check if chunking is enabled
    mamba_needs_chunking = mamba_slab_sizes.get('needs_chunking', False)
    in_proj_num_chunks = mamba_slab_sizes.get('in_proj_num_chunks', 1)
    out_proj_num_chunks = mamba_slab_sizes.get('out_proj_num_chunks', 1)
    in_proj_chunk_out_features = mamba_slab_sizes.get('in_proj_chunk_out_features', 0)
    out_proj_chunk_out_features = mamba_slab_sizes.get('out_proj_chunk_out_features', 0)
    chunk_d_model = mamba_slab_sizes.get('d_model', 0)
    chunk_d_inner = mamba_slab_sizes.get('d_inner', 0)
%>
// --- Mamba L3 Streaming Helper ---
% if mamba_needs_chunking:
// CHUNKING MODE: Large projections don't fit in L2
// in_proj: ${in_proj_num_chunks} chunks, out_proj: ${out_proj_num_chunks} chunks
// Load small weights only (projections loaded chunk-by-chunk during execution)
static void mamba_stream_small_weights(
    network_cl_args_t *a,
    void *conv1d_weight_l3, int conv1d_weight_bytes,
    void *conv1d_bias_l3, int conv1d_bias_bytes,
    void *silu_lut_l3, int silu_lut_bytes,
    void *silu_gate_lut_q13_l3, int silu_gate_lut_q13_bytes,
    void *softplus_lut_l3, int softplus_lut_bytes,
    void *exp_lut_l3, int exp_lut_bytes,
    void *x_proj_l3, int x_proj_bytes,
    void *dt_proj_l3, int dt_proj_bytes,
    void *dt_proj_bias_l3, int dt_proj_bias_bytes,
    void *A_q15_l3, int A_q15_bytes,
    void *D_q15_l3, int D_q15_bytes
) {
    // Load small weights from L3 to the shared slab (NOT in_proj/out_proj)
    cl_ram_read(a->mamba_cur_conv1d_weight, conv1d_weight_l3, conv1d_weight_bytes);
    cl_ram_read(a->mamba_cur_conv1d_bias, conv1d_bias_l3, conv1d_bias_bytes);
    cl_ram_read(a->mamba_cur_silu_lut, silu_lut_l3, silu_lut_bytes);
    cl_ram_read(a->mamba_cur_silu_gate_lut_q13, silu_gate_lut_q13_l3, silu_gate_lut_q13_bytes);
    cl_ram_read(a->mamba_cur_softplus_lut, softplus_lut_l3, softplus_lut_bytes);
    cl_ram_read(a->mamba_cur_exp_lut, exp_lut_l3, exp_lut_bytes);
    cl_ram_read(a->mamba_cur_x_proj_weight, x_proj_l3, x_proj_bytes);
    cl_ram_read(a->mamba_cur_dt_proj_weight, dt_proj_l3, dt_proj_bytes);
    cl_ram_read(a->mamba_cur_dt_proj_bias_q16_16, dt_proj_bias_l3, dt_proj_bias_bytes);
    cl_ram_read(a->mamba_cur_A_q15, A_q15_l3, A_q15_bytes);
    cl_ram_read(a->mamba_cur_D_q15, D_q15_l3, D_q15_bytes);
}

// Async prefetch handle for direction-level small weights pipelining
// Allows prefetching next direction's small weights during current direction's out_proj
#define NUM_SMALL_WEIGHT_BUFFERS 11
typedef struct {
    pi_cl_ram_req_t reqs[NUM_SMALL_WEIGHT_BUFFERS];
    int bytes[NUM_SMALL_WEIGHT_BUFFERS];
    int pending;
} small_weights_prefetch_t;

// Start async prefetch of all small weights for next direction (non-blocking)
// Call this during out_proj of current direction to overlap DMA with compute
static void mamba_stream_small_weights_async_start(
    small_weights_prefetch_t *handle,
    network_cl_args_t *a,
    void *conv1d_weight_l3, int conv1d_weight_bytes,
    void *conv1d_bias_l3, int conv1d_bias_bytes,
    void *silu_lut_l3, int silu_lut_bytes,
    void *silu_gate_lut_q13_l3, int silu_gate_lut_q13_bytes,
    void *softplus_lut_l3, int softplus_lut_bytes,
    void *exp_lut_l3, int exp_lut_bytes,
    void *x_proj_l3, int x_proj_bytes,
    void *dt_proj_l3, int dt_proj_bytes,
    void *dt_proj_bias_l3, int dt_proj_bias_bytes,
    void *A_q15_l3, int A_q15_bytes,
    void *D_q15_l3, int D_q15_bytes
) {
    struct pi_device *ram_ptr = get_ram_ptr();
    handle->pending = 1;

    // Store sizes for validation
    handle->bytes[0] = conv1d_weight_bytes;
    handle->bytes[1] = conv1d_bias_bytes;
    handle->bytes[2] = silu_lut_bytes;
    handle->bytes[3] = silu_gate_lut_q13_bytes;
    handle->bytes[4] = softplus_lut_bytes;
    handle->bytes[5] = exp_lut_bytes;
    handle->bytes[6] = x_proj_bytes;
    handle->bytes[7] = dt_proj_bytes;
    handle->bytes[8] = dt_proj_bias_bytes;
    handle->bytes[9] = A_q15_bytes;
    handle->bytes[10] = D_q15_bytes;

    // Initiate all DMA transfers (non-blocking)
    // These will run in parallel with compute on the current direction
    pi_cl_ram_read(ram_ptr, (uint32_t)conv1d_weight_l3, a->mamba_cur_conv1d_weight, conv1d_weight_bytes, &handle->reqs[0]);
    pi_cl_ram_read(ram_ptr, (uint32_t)conv1d_bias_l3, a->mamba_cur_conv1d_bias, conv1d_bias_bytes, &handle->reqs[1]);
    pi_cl_ram_read(ram_ptr, (uint32_t)silu_lut_l3, a->mamba_cur_silu_lut, silu_lut_bytes, &handle->reqs[2]);
    pi_cl_ram_read(ram_ptr, (uint32_t)silu_gate_lut_q13_l3, a->mamba_cur_silu_gate_lut_q13, silu_gate_lut_q13_bytes, &handle->reqs[3]);
    pi_cl_ram_read(ram_ptr, (uint32_t)softplus_lut_l3, a->mamba_cur_softplus_lut, softplus_lut_bytes, &handle->reqs[4]);
    pi_cl_ram_read(ram_ptr, (uint32_t)exp_lut_l3, a->mamba_cur_exp_lut, exp_lut_bytes, &handle->reqs[5]);
    pi_cl_ram_read(ram_ptr, (uint32_t)x_proj_l3, a->mamba_cur_x_proj_weight, x_proj_bytes, &handle->reqs[6]);
    pi_cl_ram_read(ram_ptr, (uint32_t)dt_proj_l3, a->mamba_cur_dt_proj_weight, dt_proj_bytes, &handle->reqs[7]);
    pi_cl_ram_read(ram_ptr, (uint32_t)dt_proj_bias_l3, a->mamba_cur_dt_proj_bias_q16_16, dt_proj_bias_bytes, &handle->reqs[8]);
    pi_cl_ram_read(ram_ptr, (uint32_t)A_q15_l3, a->mamba_cur_A_q15, A_q15_bytes, &handle->reqs[9]);
    pi_cl_ram_read(ram_ptr, (uint32_t)D_q15_l3, a->mamba_cur_D_q15, D_q15_bytes, &handle->reqs[10]);
}

// Wait for all async small weight prefetches to complete (blocking)
// Call this before using the weights in the next direction
static void mamba_stream_small_weights_async_wait(small_weights_prefetch_t *handle) {
    if (!handle->pending) return;

    // Wait for all DMA transfers to complete
    for (int i = 0; i < NUM_SMALL_WEIGHT_BUFFERS; i++) {
        if (handle->bytes[i] > 0) {
            pi_cl_ram_read_wait(&handle->reqs[i]);
        }
    }
    handle->pending = 0;
}

// Load a chunk of in_proj weights: weights[chunk_start:chunk_end, :] -> dst buffer
// in_proj shape: [2*d_inner, d_model] (output_features, input_features)
// Returns: bytes transferred
static int mamba_stream_in_proj_chunk_to(
    int8_t *dst,            // Destination buffer (ping or pong)
    void *in_proj_l3,       // Full in_proj weights in L3
    int chunk_idx,          // Which chunk (0-indexed)
    int chunk_out_features, // Max output features per chunk
    int total_out_features, // Total output features (2*d_inner)
    int in_features         // Input features (d_model)
) {
    int chunk_start = chunk_idx * chunk_out_features;
    int chunk_end = chunk_start + chunk_out_features;
    if (chunk_end > total_out_features) chunk_end = total_out_features;
    int this_chunk_size = chunk_end - chunk_start;
    int bytes = this_chunk_size * in_features;
    void *src = (int8_t *)in_proj_l3 + chunk_start * in_features;
    cl_ram_read(dst, src, bytes);
    return bytes;
}

// Convenience wrapper that uses the default destination buffer
static void mamba_stream_in_proj_chunk(
    network_cl_args_t *a,
    void *in_proj_l3,       // Full in_proj weights in L3
    int chunk_idx,          // Which chunk (0-indexed)
    int chunk_out_features, // Max output features per chunk
    int total_out_features, // Total output features (2*d_inner)
    int in_features         // Input features (d_model)
) {
    mamba_stream_in_proj_chunk_to(a->mamba_cur_in_proj_weight,
                                   in_proj_l3, chunk_idx, chunk_out_features,
                                   total_out_features, in_features);
}

// Async prefetch handle for double-buffered chunk streaming
// Uses pi_cl_ram_read() for TRUE async L3->L2 DMA
typedef struct {
    int8_t *dst;
    void *src;
    int bytes;
    int pending;
    pi_cl_ram_req_t req;  // DMA request handle for async operation
} chunk_prefetch_t;

// Start async prefetch of in_proj chunk (non-blocking)
// This initiates the DMA transfer and returns immediately
static void mamba_stream_in_proj_chunk_async_start(
    chunk_prefetch_t *handle,
    int8_t *dst,            // Destination buffer (ping or pong)
    void *in_proj_l3,       // Full in_proj weights in L3
    int chunk_idx,          // Which chunk (0-indexed)
    int chunk_out_features, // Max output features per chunk
    int total_out_features, // Total output features (2*d_inner)
    int in_features         // Input features (d_model)
) {
    int chunk_start = chunk_idx * chunk_out_features;
    int chunk_end = chunk_start + chunk_out_features;
    if (chunk_end > total_out_features) chunk_end = total_out_features;
    int this_chunk_size = chunk_end - chunk_start;
    handle->bytes = this_chunk_size * in_features;
    handle->src = (int8_t *)in_proj_l3 + chunk_start * in_features;
    handle->dst = dst;
    handle->pending = 1;

    // TRUE ASYNC: Initiate DMA transfer without waiting
    // pi_cl_ram_read() returns immediately, DMA runs in background
    struct pi_device *ram_ptr = get_ram_ptr();
    pi_cl_ram_read(ram_ptr, (uint32_t)handle->src, handle->dst, handle->bytes, &handle->req);
}

// Wait for async prefetch to complete (blocking)
// Call this before using the data in dst buffer
static void mamba_stream_in_proj_chunk_async_wait(chunk_prefetch_t *handle) {
    if (handle->pending && handle->bytes > 0) {
        // Wait for the DMA transfer initiated in async_start to complete
        pi_cl_ram_read_wait(&handle->req);
        handle->pending = 0;
    }
}

// Load a chunk of out_proj weights: weights[chunk_start:chunk_end, :] -> dst buffer
// out_proj shape: [d_model, d_inner] (output_features, input_features)
// Returns: bytes transferred
static int mamba_stream_out_proj_chunk_to(
    int8_t *dst,            // Destination buffer (ping or pong)
    void *out_proj_l3,      // Full out_proj weights in L3
    int chunk_idx,          // Which chunk (0-indexed)
    int chunk_out_features, // Max output features per chunk
    int total_out_features, // Total output features (d_model)
    int in_features         // Input features (d_inner)
) {
    int chunk_start = chunk_idx * chunk_out_features;
    int chunk_end = chunk_start + chunk_out_features;
    if (chunk_end > total_out_features) chunk_end = total_out_features;
    int this_chunk_size = chunk_end - chunk_start;
    int bytes = this_chunk_size * in_features;
    void *src = (int8_t *)out_proj_l3 + chunk_start * in_features;
    cl_ram_read(dst, src, bytes);
    return bytes;
}

// Convenience wrapper that uses the default destination buffer
static void mamba_stream_out_proj_chunk(
    network_cl_args_t *a,
    void *out_proj_l3,      // Full out_proj weights in L3
    int chunk_idx,          // Which chunk (0-indexed)
    int chunk_out_features, // Max output features per chunk
    int total_out_features, // Total output features (d_model)
    int in_features         // Input features (d_inner)
) {
    mamba_stream_out_proj_chunk_to(a->mamba_cur_out_proj_weight,
                                   out_proj_l3, chunk_idx, chunk_out_features,
                                   total_out_features, in_features);
}

// Async prefetch handle for double-buffered out_proj chunk streaming
typedef struct {
    int8_t *dst;
    void *src;
    int bytes;
    int pending;
    pi_cl_ram_req_t req;
} out_proj_prefetch_t;

// Start async prefetch of out_proj chunk (non-blocking)
static void mamba_stream_out_proj_chunk_async_start(
    out_proj_prefetch_t *handle,
    int8_t *dst,            // Destination buffer (ping or pong)
    void *out_proj_l3,      // Full out_proj weights in L3
    int chunk_idx,          // Which chunk (0-indexed)
    int chunk_out_features, // Max output features per chunk
    int total_out_features, // Total output features (d_model)
    int in_features         // Input features (d_inner)
) {
    int chunk_start = chunk_idx * chunk_out_features;
    int chunk_end = chunk_start + chunk_out_features;
    if (chunk_end > total_out_features) chunk_end = total_out_features;
    int this_chunk_size = chunk_end - chunk_start;
    handle->bytes = this_chunk_size * in_features;
    handle->src = (int8_t *)out_proj_l3 + chunk_start * in_features;
    handle->dst = dst;
    handle->pending = 1;

    // TRUE ASYNC: Initiate DMA transfer without waiting
    struct pi_device *ram_ptr = get_ram_ptr();
    pi_cl_ram_read(ram_ptr, (uint32_t)handle->src, handle->dst, handle->bytes, &handle->req);
}

// Wait for async out_proj prefetch to complete (blocking)
static void mamba_stream_out_proj_chunk_async_wait(out_proj_prefetch_t *handle) {
    if (handle->pending && handle->bytes > 0) {
        pi_cl_ram_read_wait(&handle->req);
        handle->pending = 0;
    }
}
% else:
// NON-CHUNKING MODE: All weights fit in L2
// Load weights for a specific Mamba direction from L3 to the shared slab
static void mamba_stream_direction_weights(
    network_cl_args_t *a,
    void *in_proj_l3, int in_proj_bytes,
    void *conv1d_weight_l3, int conv1d_weight_bytes,
    void *conv1d_bias_l3, int conv1d_bias_bytes,
    void *silu_lut_l3, int silu_lut_bytes,
    void *silu_gate_lut_q13_l3, int silu_gate_lut_q13_bytes,
    void *softplus_lut_l3, int softplus_lut_bytes,
    void *exp_lut_l3, int exp_lut_bytes,
    void *x_proj_l3, int x_proj_bytes,
    void *dt_proj_l3, int dt_proj_bytes,
    void *dt_proj_bias_l3, int dt_proj_bias_bytes,
    void *A_q15_l3, int A_q15_bytes,
    void *D_q15_l3, int D_q15_bytes,
    void *out_proj_l3, int out_proj_bytes
) {
    // Load all weights from L3 to the shared slab
    cl_ram_read(a->mamba_cur_in_proj_weight, in_proj_l3, in_proj_bytes);
    cl_ram_read(a->mamba_cur_conv1d_weight, conv1d_weight_l3, conv1d_weight_bytes);
    cl_ram_read(a->mamba_cur_conv1d_bias, conv1d_bias_l3, conv1d_bias_bytes);
    cl_ram_read(a->mamba_cur_silu_lut, silu_lut_l3, silu_lut_bytes);
    cl_ram_read(a->mamba_cur_silu_gate_lut_q13, silu_gate_lut_q13_l3, silu_gate_lut_q13_bytes);
    cl_ram_read(a->mamba_cur_softplus_lut, softplus_lut_l3, softplus_lut_bytes);
    cl_ram_read(a->mamba_cur_exp_lut, exp_lut_l3, exp_lut_bytes);
    cl_ram_read(a->mamba_cur_x_proj_weight, x_proj_l3, x_proj_bytes);
    cl_ram_read(a->mamba_cur_dt_proj_weight, dt_proj_l3, dt_proj_bytes);
    cl_ram_read(a->mamba_cur_dt_proj_bias_q16_16, dt_proj_bias_l3, dt_proj_bias_bytes);
    cl_ram_read(a->mamba_cur_A_q15, A_q15_l3, A_q15_bytes);
    cl_ram_read(a->mamba_cur_D_q15, D_q15_l3, D_q15_bytes);
    cl_ram_read(a->mamba_cur_out_proj_weight, out_proj_l3, out_proj_bytes);
}
% endif
% endif

// --- Helper Functions ---
#ifndef MINIMAL_OUTPUT
void compare_int8_output_impl(const char *layer_name, const int8_t *output, const int8_t *golden, size_t size) {
    if (pi_core_id() != CL_ORCHESTRATOR_CORE_ID) return;
    if (golden == NULL || output == NULL) {
        printf("  %s: SKIPPED validation (output=%p, golden=%p)\n", layer_name, (void *)output, (void *)golden);
        return;
    }

    int mismatches = 0;
    int max_diff = 0;
    long long sum_diff = 0;
    for (size_t i = 0; i < size; i++) {
        int diff = (int)output[i] - (int)golden[i];
        if (diff < 0) diff = -diff;
        if (diff) mismatches++;
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    float mean_diff = size ? ((float)sum_diff / size) : 0.0f;
    float mismatch_percent = size ? (100.0f * mismatches / size) : 0.0f;
    printf("  %s: mismatches=%d/%zu (%.1f%%), max_diff=%d, mean_diff=%.2f\n",
           layer_name, mismatches, size, mismatch_percent, max_diff, mean_diff);

    if (mismatches > 0) {
        size_t limit = (size < 16) ? size : 16;
        printf("    Output: [");
        for (size_t i = 0; i < limit; i++) printf("%d%s", output[i], (i + 1 < limit) ? ", " : "");
        printf("...]\n");
        printf("    Golden: [");
        for (size_t i = 0; i < limit; i++) printf("%d%s", golden[i], (i + 1 < limit) ? ", " : "");
        printf("...]\n");
    }
}

% if use_streamed_golden:
// L3 Streamed Golden: Stream from L3 in chunks and compare
// Chunk size: ${golden_chunk_size} bytes (allows comparing arbitrarily large goldens)
#define GOLDEN_CHUNK_SIZE ${golden_chunk_size}

static void compare_int8_output_streamed(const char *layer_name, const int8_t *output,
                                         void *golden_l3, int8_t *staging, size_t size,
                                         struct pi_device *ram_dev) {
    if (pi_core_id() != CL_ORCHESTRATOR_CORE_ID) return;
    if (golden_l3 == NULL || output == NULL || staging == NULL) {
        printf("  %s: SKIPPED (no golden)\n", layer_name);
        return;
    }

    // Compare in chunks to handle arbitrarily large golden outputs
    size_t total_mismatches = 0;
    int max_diff = 0;
    double sum_diff = 0.0;
    size_t offset = 0;

    while (offset < size) {
        size_t chunk = (size - offset > GOLDEN_CHUNK_SIZE) ? GOLDEN_CHUNK_SIZE : (size - offset);

        // Stream this chunk from L3 to staging buffer
        pi_cl_ram_req_t req;
        pi_cl_ram_read(ram_dev, (uint32_t)golden_l3 + offset, staging, chunk, &req);
        pi_cl_ram_read_wait(&req);

        // Compare this chunk
        for (size_t i = 0; i < chunk; i++) {
            int diff = (int)output[offset + i] - (int)staging[i];
            if (diff < 0) diff = -diff;
            if (diff > 0) {
                total_mismatches++;
                sum_diff += diff;
                if (diff > max_diff) max_diff = diff;
            }
        }
        offset += chunk;
    }

    // Print results
    double pct = (100.0 * total_mismatches) / size;
    double mean_diff = (total_mismatches > 0) ? (sum_diff / total_mismatches) : 0.0;
    printf("  %s: mismatches=%zu/%zu (%.1f%%), max_diff=%d, mean_diff=%.2f\n",
           layer_name, total_mismatches, size, pct, max_diff, mean_diff);
}
#define compare_int8_output(name, output, golden_l3, size) \
    compare_int8_output_streamed(name, output, golden_l3, a->golden_staging, size, a->ram_dev)
% else:
#define compare_int8_output(name, output, golden, size) \
    compare_int8_output_impl(name, output, golden, size)
% endif
#endif // MINIMAL_OUTPUT

// L1 Fusion Kernels
// These operate on data already in L1 (TCDM) with parallel core execution.
// The L2 versions (relu_int8_inplace, requantize_int8_inplace) are in network_kernels.c.
static inline void relu_int8_inplace_l1(int8_t *data, size_t size) {
    size_t i = pi_core_id();
    const size_t step = NUM_CORES;
    for (; i < size; i += step) {
        if (data[i] < 0) data[i] = 0;
    }
}

static inline void requantize_int8_inplace_l1(int8_t *data, size_t size, float scale_in, float scale_out) {
    if (fabsf(scale_in - scale_out) < 1e-12f) return;
    int8_t map_int8[256];
    for (int v = -128; v <= 127; v++) {
        float val_fp32 = (float)v * scale_in;
        int32_t val_int32 = (int32_t)lrintf(val_fp32 / scale_out);
        if (val_int32 > 127) val_int32 = 127;
        if (val_int32 < -128) val_int32 = -128;
        map_int8[(uint32_t)(v + 128)] = (int8_t)val_int32;
    }

    size_t i = pi_core_id();
    const size_t step = NUM_CORES;
    for (; i < size; i += step) {
        data[i] = map_int8[(uint32_t)((int)data[i] + 128)];
    }
}

/* End Helper Functions */
