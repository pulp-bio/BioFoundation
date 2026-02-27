/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * simple_executor.h - Data-driven executor for simple ops
 *
 * Provides table-driven execution of simple operations, keeping
 * generated network code compact and readable.
 */

#ifndef SIMPLE_EXECUTOR_H
#define SIMPLE_EXECUTOR_H

#include "layer_descriptors.h"
#include <stdint.h>

/**
 * Execute all simple layers in sequence.
 *
 * @param layers     Array of SimpleLayerSpec descriptors
 * @param num_layers Number of layers in the array
 * @param buffers    Buffer slot table mapping indices to L2 offsets
 * @param l2_arena   Base pointer to L2 memory arena
 */
void execute_simple_layers(
    const SimpleLayerSpec* layers,
    int num_layers,
    const BufferSlot* buffers,
    int8_t* l2_arena
);

/**
 * Execute a single simple layer by index.
 * Used for interleaved execution when simple and complex ops
 * must execute in a specific order.
 *
 * @param layers     Array of SimpleLayerSpec descriptors
 * @param layer_idx  Index of the layer to execute
 * @param buffers    Buffer slot table mapping indices to L2 offsets
 * @param l2_arena   Base pointer to L2 memory arena
 */
void execute_simple_layer_by_index(
    const SimpleLayerSpec* layers,
    int layer_idx,
    const BufferSlot* buffers,
    int8_t* l2_arena
);

#endif // SIMPLE_EXECUTOR_H
