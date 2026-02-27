/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/** L3 prefetch request types and synchronization interface. */
/**
 * L3 Prefetch Interface
 *
 * This header defines the interface for explicit L3→L2 data movement.
 * Currently implements blocking prefetch; designed to support future
 * asynchronous pipelining.
 *
 * Buffer Residency Rules:
 * =======================
 *
 * L2 RESIDENT (remains in L2 throughout network execution):
 * - Network weights and biases (loaded once at start)
 * - Intermediate activations (when L2 budget allows)
 * - Input tensor (loaded once at start)
 *
 * L3 EVICTION (would be used if L2 budget exhausted - not currently implemented):
 * - Large intermediate buffers that don't fit in L2
 * - Temporary scratch buffers
 *
 * Current Policy:
 * - All weights/biases: L2 resident (loaded in network_fc_entry)
 * - All activations: L2 resident (allocated and reused throughout)
 * - No L3 eviction (L2 is sufficient for current test networks)
 *
 */

#ifndef L3_PREFETCH_H
#define L3_PREFETCH_H

#include <stddef.h>
#include "mem.h"  // For cl_ram_read

/**
 * L3 Prefetch Request
 *
 * Describes a data movement operation from L3 (external memory) to L2 (cluster memory).
 */
typedef struct {
    void *l3_src;      // Source address in L3 memory
    void *l2_dst;      // Destination address in L2 memory
    size_t bytes;      // Number of bytes to transfer
    int priority;      // Priority level (0=highest, higher=lower priority)
                       // Currently unused; reserved for future scheduler
} l3_prefetch_request_t;

/**
 * Synchronous L3→L2 Prefetch (Blocking)
 *
 * Performs immediate blocking transfer from L3 to L2.
 * This is a wrapper around cl_ram_read() with explicit documentation
 * of data movement for future pipelining.
 *
 * @param req  Prefetch request describing the transfer
 */
static inline void l3_prefetch_sync(l3_prefetch_request_t *req)
{
    if (!req || !req->l3_src || !req->l2_dst || req->bytes == 0) {
        return;
    }

    // Blocking transfer using cl_ram_read
    cl_ram_read(req->l2_dst, req->l3_src, req->bytes);
}

#endif // L3_PREFETCH_H
