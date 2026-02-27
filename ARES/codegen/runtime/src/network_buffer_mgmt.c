/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// L2 buffer allocation and batch-free implementation.

#include "network_buffer_mgmt.h"
#include "mem.h"
#include <stdio.h>

// Free multiple L2 buffers in batch
// Frees all non-NULL buffers and sets pointers to NULL
void free_l2_buffers(
    buffer_descriptor_t *descriptors,
    size_t num_descriptors
)
{
    if (!descriptors) {
        return;
    }

    for (size_t i = 0; i < num_descriptors; i++) {
        buffer_descriptor_t *desc = &descriptors[i];

        if (!desc->buffer_ptr || !(*desc->buffer_ptr)) {
            continue;  // Skip NULL or already freed buffers
        }

        // Free the buffer
        pi_l2_free(*desc->buffer_ptr, desc->bytes);

        // NULL the pointer to prevent double-free
        *desc->buffer_ptr = NULL;
    }
}
