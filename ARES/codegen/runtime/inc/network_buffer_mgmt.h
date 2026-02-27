/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// L2 buffer allocation and batch-free utilities.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Buffer descriptor for L2 cleanup.
 *
 * Used for batch freeing of L2 buffers with automatic NULL'ing
 * to prevent double-free and use-after-free bugs.
 *
 * USAGE:
 * - Create array of descriptors
 * - Initialize all fields (name optional, buffer_ptr required, bytes required)
 * - Pass to free_l2_buffers()
 * - After function returns, all *buffer_ptr are NULL
 */
typedef struct {
    const char *name;        // Name for logging (optional, can be NULL)
    void **buffer_ptr;       // Pointer to buffer pointer (will be NULL'd after free)
    size_t bytes;            // Buffer size in bytes (passed to pi_l2_free)
} buffer_descriptor_t;

/**
 * Free multiple L2 buffers in batch.
 *
 * Iterates through descriptor array, frees all non-NULL buffers,
 * and sets buffer pointers to NULL to prevent double-free.
 *
 * EXAMPLE:
 *   int8_t *conv1_golden = <allocated buffer>;
 *   int8_t *conv2_golden = <allocated buffer>;
 *   buffer_descriptor_t descs[] = {
 *       {.name = "conv1_golden", .buffer_ptr = (void **)&conv1_golden, .bytes = 12544},
 *       {.name = "conv2_golden", .buffer_ptr = (void **)&conv2_golden, .bytes = 6272},
 *   };
 *   free_l2_buffers(descs, 2);
 *   // conv1_golden and conv2_golden are now NULL
 *
 * SAFETY:
 * - Skips NULL buffer_ptr (defensive programming)
 * - Skips already-freed buffers (*buffer_ptr == NULL)
 * - NULLs pointer after free to prevent double-free
 *
 * @param descriptors Array of buffer descriptors
 * @param num_descriptors Number of descriptors in array
 */
void free_l2_buffers(
    buffer_descriptor_t *descriptors,
    size_t num_descriptors
);

#ifdef __cplusplus
}
#endif
