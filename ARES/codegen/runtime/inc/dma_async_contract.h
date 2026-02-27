/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared runtime header for standardized DMA async contract
// Part of ARES GAP9 code generation pipeline

#ifndef DMA_ASYNC_CONTRACT_H
#define DMA_ASYNC_CONTRACT_H

#include <stddef.h>
#include "network_dma_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

// Contract aliases keep API names explicit while preserving existing
// request/future object layouts used by runtime call sites.
typedef dma_stream_direction_t dma_contract_direction_t;
typedef dma_wait_policy_t dma_contract_wait_policy_t;
typedef dma_async_request_t dma_contract_request_t;
typedef dma_async_future_t dma_contract_future_t;

void dma_contract_request_reset(dma_contract_request_t *request);
void dma_contract_future_reset(dma_contract_future_t *future);

int dma_contract_request_make_l3_read(
    dma_contract_request_t *request,
    struct pi_device *ram_dev,
    void *l2_dst,
    void *l3_src,
    size_t bytes
);

int dma_contract_request_make_l3_write(
    dma_contract_request_t *request,
    struct pi_device *ram_dev,
    void *l3_dst,
    void *l2_src,
    size_t bytes
);

int dma_contract_submit(
    dma_contract_future_t *future,
    const dma_contract_request_t *request
);

int dma_contract_wait(
    dma_contract_future_t *future,
    dma_contract_wait_policy_t wait_policy,
    dma_contract_direction_t direction_filter
);

int dma_contract_wait_barrier(dma_contract_future_t *futures, size_t count);
int dma_contract_poll(const dma_contract_future_t *future);

int dma_contract_l2_copy_start(
    dma_contract_future_t *future,
    const pi_cl_dma_copy_t *copy
);

int dma_contract_l2_copy_wait(dma_contract_future_t *future);
int dma_contract_l2_copy_sync(const pi_cl_dma_copy_t *copy);

#ifdef __cplusplus
}
#endif

#endif // DMA_ASYNC_CONTRACT_H
