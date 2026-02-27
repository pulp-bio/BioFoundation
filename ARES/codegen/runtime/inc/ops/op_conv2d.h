/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Conv2D Tile Workers Header for ARES Runtime
 *
 * Tile workers for L2 fallback and pipelined Conv2D execution.
 * Type definitions are in network_kernels.h.
 */

#ifndef ARES_OPS_CONV2D_H
#define ARES_OPS_CONV2D_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Conv2D tile worker for L2 fallback */
void conv2d_tile_worker(void *arg);

/* Conv2D tile worker with ReLU/requant fusion */
void conv2d_tile_worker_with_fusion(void *arg);

#ifdef __cplusplus
}
#endif

#endif /* ARES_OPS_CONV2D_H */
