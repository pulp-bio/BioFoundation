/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// L3→L2 prefetch interface for weight and activation streaming.

#pragma once

#include <stdint.h>
#include <stddef.h>
#include "dma_async_contract.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Prefetch descriptor for simple L3→L2 transfers.
 *
 * Used for batch prefetching of network data (weights, biases, input)
 * where the L2 destination buffer is pre-allocated.
 *
 * USAGE:
 * - Create array of descriptors
 * - Initialize all fields (name, l3_src, l2_dst, bytes, priority)
 * - Pass to prefetch_network_data()
 */
typedef struct {
    const char *name;        // Name for logging (e.g., "conv1_weight")
    void *l3_src;            // Source address in L3 (HyperRAM/Flash)
    void *l2_dst;            // Destination address in L2 (pre-allocated)
    size_t bytes;            // Transfer size in bytes
    int priority;            // 0=high priority (weights/biases), 1=low priority (inputs)
} prefetch_descriptor_t;

/**
 * Prefetch descriptor for golden outputs (with automatic L2 allocation).
 *
 * Used for batch prefetching of golden reference data where the L2
 * destination buffer needs to be allocated first.
 *
 * USAGE:
 * - Create array of descriptors
 * - Initialize all fields (name, l3_src, l2_dst_ptr, bytes, priority)
 * - Pass to prefetch_golden_outputs()
 * - Function allocates L2 buffer and stores pointer in *l2_dst_ptr
 *
 */
typedef struct {
    const char *name;        // Name for logging (e.g., "conv1_golden")
    void *l3_src;            // Source address in L3 (NULL if no golden data)
    void **l2_dst_ptr;       // Pointer to L2 destination pointer (function will malloc and store here)
    size_t bytes;            // Transfer size in bytes (also used for malloc size)
    int priority;            // 0=high priority, 1=low priority (currently unused)
} golden_prefetch_descriptor_t;

/**
 * Batch prefetch network data from L3 to L2.
 *
 * Transfers weights, biases, and input tensors from L3 (HyperRAM/Flash)
 * to L2 (cluster shared memory) in priority order. Uses synchronous
 * blocking transfers (l3_prefetch_sync).
 *
 * EXAMPLE:
 *   prefetch_descriptor_t descs[] = {
 *       {.name = "input", .l3_src = input_l3, .l2_dst = input_l2, .bytes = 784, .priority = 0},
 *       {.name = "conv1_weight", .l3_src = w_l3, .l2_dst = w_l2, .bytes = 144, .priority = 0},
 *   };
 *   prefetch_network_data(descs, 2);
 *
 * @param descriptors Array of prefetch descriptors (pre-allocated L2 destinations)
 * @param num_descriptors Number of descriptors in array
 * @return 0 on success, 0 if nothing to prefetch (NULL descriptors or zero count)
 */
int prefetch_network_data(
    prefetch_descriptor_t *descriptors,
    size_t num_descriptors
);

/**
 * Batch prefetch golden outputs from L3 to L2 (with malloc).
 *
 * Allocates L2 buffers and transfers golden reference data from L3
 * for intermediate layer validation. Skips descriptors with NULL l3_src.
 *
 * EXAMPLE:
 *   int8_t *conv1_golden = NULL;
 *   golden_prefetch_descriptor_t descs[] = {
 *       {.name = "conv1_golden", .l3_src = golden_l3, .l2_dst_ptr = &conv1_golden, .bytes = 12544, .priority = 0},
 *   };
 *   if (prefetch_golden_outputs(descs, 1) == 0) {
 *       // conv1_golden now points to allocated L2 buffer with golden data
 *   }
 *
 * @param descriptors Array of golden prefetch descriptors (L2 will be malloc'd)
 * @param num_descriptors Number of descriptors in array
 * @return 0 on success, -1 on L2 allocation failure, 0 if nothing to prefetch
 */
int prefetch_golden_outputs(
    golden_prefetch_descriptor_t *descriptors,
    size_t num_descriptors
);

// ========== Async Prefetch API (JIT Weight Staging) ==========

/**
 * Async prefetch handle for non-blocking L3→L2 transfers.
 *
 * Used to track in-flight async prefetch operations. Allows computation
 * to overlap with L3→L2 DMA transfers for improved performance.
 *
 * USAGE:
 * 1. Call prefetch_async_start() to initiate transfer (returns immediately)
 * 2. Do other work (computation, other DMAs, etc.)
 * 3. Call prefetch_async_wait() before using prefetched data
 *
 */
typedef struct {
    void *l3_src;            // Source address in L3 (HyperRAM/Flash)
    void *l2_dst;            // Destination address in L2 (pre-allocated)
    size_t bytes;            // Transfer size in bytes
    int pending;             // 1 if transfer in-flight, 0 if complete
    int submitted;           // 1 if request has been submitted to DMA helper
    const char *name;        // Name for logging/debugging
    dma_contract_request_t request;  // Deferred DMA request (contract API)
    dma_contract_future_t future;    // Future handle used at wait/check boundaries
    unsigned int start_cycles;     // Cycle count when prefetch started
    unsigned int transfer_cycles;  // Measured L3→L2 transfer time
} prefetch_async_handle_t;

/**
 * Start async prefetch from L3 to L2 (non-blocking).
 *
 * Registers an L3→L2 DMA request and returns immediately.
 *
 * Contract scaffolding keeps current behavior by deferring submission to
 * prefetch_async_wait(). This preserves existing pipelines while introducing
 * request/future state in the handle.
 *
 * EXAMPLE:
 *   prefetch_async_handle_t handle;
 *   prefetch_async_start(&handle, "fc2_weight", fc2_weight_l3, fc2_weight_l2, 32768);
 *
 *   // Execute current layer (overlaps with prefetch!)
 *   execute_layer_fc1();
 *
 *   // Wait for prefetch before starting next layer
 *   prefetch_async_wait(&handle);
 *   execute_layer_fc2();  // fc2 weights now in L2
 *
 * @param handle Handle to track async transfer (caller-allocated)
 * @param name Name for logging (e.g., "conv2_weight")
 * @param l3_src Source address in L3 memory
 * @param l2_dst Destination address in L2 memory (pre-allocated)
 * @param bytes Number of bytes to transfer
 * @return 0 on success, -1 on error (NULL pointers, zero bytes)
 */
int prefetch_async_start(
    prefetch_async_handle_t *handle,
    const char *name,
    void *l3_src,
    void *l2_dst,
    size_t bytes
);

/**
 * Wait for async prefetch to complete (blocking).
 *
 * Blocks until the L3→L2 transfer initiated by prefetch_async_start()
 * completes. After this call, the data is guaranteed to be in L2.
 *
 * EXAMPLE:
 *   prefetch_async_handle_t handle;
 *   prefetch_async_start(&handle, "weights", weights_l3, weights_l2, 200704);
 *
 *   // Do other work...
 *
 *   prefetch_async_wait(&handle);  // Block until transfer done
 *   // Now safe to use weights_l2
 *
 * @param handle Handle from prefetch_async_start()
 * @return 0 on success, -1 if handle invalid or not pending
 */
int prefetch_async_wait(
    prefetch_async_handle_t *handle
);

/**
 * Check if async prefetch is complete (non-blocking).
 *
 * Tests whether the transfer initiated by prefetch_async_start() has
 * completed. Returns immediately without blocking.
 *
 * EXAMPLE:
 *   prefetch_async_handle_t handle;
 *   prefetch_async_start(&handle, "weights", weights_l3, weights_l2, 200704);
 *
 *   while (!prefetch_async_check(&handle)) {
 *       // Transfer still in progress, do other work
 *   }
 *   // Transfer complete, safe to use weights_l2
 *
 * @param handle Handle from prefetch_async_start()
 * @return 1 if transfer complete, 0 if still in progress, -1 on error
 */
int prefetch_async_check(
    prefetch_async_handle_t *handle
);

#ifdef __cplusplus
}
#endif
