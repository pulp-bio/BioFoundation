/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// L3->L2 prefetch implementation for weight and activation streaming.

#include "network_l3_prefetch.h"
#include "l3_prefetch.h"  // For l3_prefetch_request_t and l3_prefetch_sync()
#include "dma_async_contract.h"
#include "mem.h"
#include <stdio.h>
#include <pmsis.h>

// Batch prefetch network data from L3 to L2
// Prefetches weights, biases, and input tensors in priority order
int prefetch_network_data(
    prefetch_descriptor_t *descriptors,
    size_t num_descriptors
)
{
    if (!descriptors || num_descriptors == 0) {
        return 0;  // Nothing to prefetch
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: L3->L2 Prefetch: Loading network data...\n");
#endif

    for (size_t i = 0; i < num_descriptors; i++) {
        prefetch_descriptor_t *desc = &descriptors[i];

        if (!desc->l3_src || !desc->l2_dst) {
            continue;  // Skip NULL pointers
        }

        l3_prefetch_request_t req = {
            .l3_src = desc->l3_src,
            .l2_dst = desc->l2_dst,
            .bytes = desc->bytes,
            .priority = desc->priority
        };
        l3_prefetch_sync(&req);

#ifndef MINIMAL_OUTPUT
        printf("CL:   [L2 RESIDENT] %s: %zu bytes\n", desc->name, desc->bytes);
#endif
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: L3->L2 Prefetch: Complete (all weights/biases loaded)\n");
#endif
    return 0;
}

// Batch prefetch golden outputs from L3 to L2 (with malloc)
// Allocates L2 buffers and prefetches golden reference data
int prefetch_golden_outputs(
    golden_prefetch_descriptor_t *descriptors,
    size_t num_descriptors
)
{
    if (!descriptors || num_descriptors == 0) {
        return 0;  // Nothing to prefetch
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: L3->L2 Prefetch: Loading golden outputs for validation...\n");
#endif

    for (size_t i = 0; i < num_descriptors; i++) {
        golden_prefetch_descriptor_t *desc = &descriptors[i];

        // Skip if no golden data available
        if (!desc->l3_src || !desc->l2_dst_ptr) {
            continue;
        }

        // Allocate L2 buffer
        void *l2_buffer = pi_l2_malloc(desc->bytes);
        if (!l2_buffer) {
            printf("CL ERR: L2 allocation failed for %s golden buffer (%zu bytes)\n",
                   desc->name, desc->bytes);
            return -1;  // Allocation failure
        }

        // Store allocated pointer
        *desc->l2_dst_ptr = l2_buffer;

        // Prefetch from L3 to L2
        l3_prefetch_request_t req = {
            .l3_src = desc->l3_src,
            .l2_dst = l2_buffer,
            .bytes = desc->bytes,
            .priority = desc->priority
        };
        l3_prefetch_sync(&req);

#ifndef MINIMAL_OUTPUT
        printf("CL:   [L2 RESIDENT] %s: %zu bytes\n", desc->name, desc->bytes);
#endif
    }

#ifndef MINIMAL_OUTPUT
    printf("CL: L3->L2 Prefetch: Intermediate golden outputs loaded\n");
#endif
    return 0;
}

// ========== Async Prefetch Implementation ==========

/**
 * Start async prefetch from L3 to L2 (non-blocking).
 *
 * Registers an L3->L2 transfer request and returns immediately.
 *
 *
 * @param handle Async handle to track transfer
 * @param name Layer name for logging
 * @param l3_src Source address in L3
 * @param l2_dst Destination address in L2 (pre-allocated)
 * @param bytes Transfer size
 * @return 0 on success, -1 on error
 */
int prefetch_async_start(
    prefetch_async_handle_t *handle,
    const char *name,
    void *l3_src,
    void *l2_dst,
    size_t bytes
)
{
    if (!handle || !l3_src || !l2_dst || bytes == 0) {
        return -1;  // Invalid parameters
    }

    struct pi_device *ram_dev = get_ram_ptr();
    if (!ram_dev) {
        return -1;
    }

    // Store transfer parameters
    handle->l3_src = l3_src;
    handle->l2_dst = l2_dst;
    handle->bytes = bytes;
    handle->name = name;
    handle->pending = 1;      // Mark as in-flight
    handle->submitted = 0;    // Deferred submit

    if (dma_contract_request_make_l3_read(&handle->request, ram_dev, l2_dst, l3_src, bytes) != 0) {
        handle->pending = 0;
        return -1;
    }
    dma_contract_future_reset(&handle->future);

#ifdef ENABLE_PERF_COUNTERS
    // Record start time for performance measurement
    handle->start_cycles = pi_perf_read(PI_PERF_CYCLES);
    handle->transfer_cycles = 0;
#endif

    return 0;
}

/**
 * Wait for async prefetch to complete (blocking).
 *
 * Blocks until the L3->L2 transfer completes. After this call,
 * data is guaranteed to be in L2.
 *
 *
 * @param handle Async handle from prefetch_async_start()
 * @return 0 on success, -1 on error
 */
int prefetch_async_wait(
    prefetch_async_handle_t *handle
)
{
    if (!handle || !handle->pending) {
        return -1;  // Invalid handle or already complete
    }

    // Validate pointers and size before RAM access
    if (!handle->l3_src || !handle->l2_dst || handle->bytes == 0) {
        printf("CL ERR: Invalid prefetch for '%s': l3_src=%p, l2_dst=%p, bytes=%zu\n",
               handle->name ? handle->name : "unknown",
               handle->l3_src, handle->l2_dst, handle->bytes);
        handle->pending = 0;
        return -1;
    }

#ifdef ENABLE_PERF_COUNTERS
    // Measure transfer time
    unsigned int transfer_start = pi_perf_read(PI_PERF_CYCLES);
#endif

    if (!handle->submitted) {
        if (dma_contract_submit(&handle->future, &handle->request) != 0) {
            handle->pending = 0;
            return -1;
        }
        handle->submitted = 1;
    }

    if (dma_contract_wait(&handle->future,
                          DMA_WAIT_POLICY_PER_TRANSFER,
                          DMA_STREAM_DIR_L3_TO_L2) != 0) {
        handle->pending = 0;
        return -1;
    }

#ifdef ENABLE_PERF_COUNTERS
    unsigned int transfer_end = pi_perf_read(PI_PERF_CYCLES);

    // Calculate and store transfer time
    handle->transfer_cycles = transfer_end - transfer_start;

    // Report transfer performance

#ifndef MINIMAL_OUTPUT
    printf("CL:   L3->L2 Transfer '%s': %u cycles (%zu bytes, %.2f MB/s @ 370MHz)\n",
           handle->name,
           handle->transfer_cycles,
           handle->bytes,
           (float)handle->bytes / (float)handle->transfer_cycles * 370.0f);

#endif
#endif

    // Mark as complete
    handle->pending = 0;

    return 0;
}

/**
 * Check if async prefetch is complete (non-blocking).
 *
 * Tests whether the transfer is done without blocking.
 *
 *
 * @param handle Async handle from prefetch_async_start()
 * @return 1 if complete, 0 if in progress, -1 on error
 */
int prefetch_async_check(
    prefetch_async_handle_t *handle
)
{
    if (!handle) {
        return -1;  // Invalid handle
    }

    if (handle->pending == 0) {
        return 1;
    }
    if (!handle->submitted) {
        return 0;
    }

    int poll = dma_contract_poll(&handle->future);
    if (poll < 0) {
        return -1;
    }
    if (poll == 1) {
        handle->pending = 0;
    }
    return poll;
}
