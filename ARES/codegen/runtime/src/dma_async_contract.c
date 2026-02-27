/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared runtime implementation for standardized DMA async contract
// Part of ARES GAP9 code generation pipeline

#include "dma_async_contract.h"

void dma_contract_request_reset(dma_contract_request_t *request) {
    dma_async_request_reset(request);
}

void dma_contract_future_reset(dma_contract_future_t *future) {
    dma_async_future_reset(future);
}

int dma_contract_request_make_l3_read(
    dma_contract_request_t *request,
    struct pi_device *ram_dev,
    void *l2_dst,
    void *l3_src,
    size_t bytes
) {
    return dma_async_request_make_l3_read(request, ram_dev, l2_dst, l3_src, bytes);
}

int dma_contract_request_make_l3_write(
    dma_contract_request_t *request,
    struct pi_device *ram_dev,
    void *l3_dst,
    void *l2_src,
    size_t bytes
) {
    return dma_async_request_make_l3_write(request, ram_dev, l3_dst, l2_src, bytes);
}

int dma_contract_submit(
    dma_contract_future_t *future,
    const dma_contract_request_t *request
) {
    return dma_async_submit(future, request);
}

int dma_contract_wait(
    dma_contract_future_t *future,
    dma_contract_wait_policy_t wait_policy,
    dma_contract_direction_t direction_filter
) {
    return dma_async_wait(future, wait_policy, direction_filter);
}

int dma_contract_wait_barrier(dma_contract_future_t *futures, size_t count) {
    return dma_async_wait_barrier(futures, count);
}

int dma_contract_poll(const dma_contract_future_t *future) {
    return dma_async_poll(future);
}

int dma_contract_l2_copy_start(
    dma_contract_future_t *future,
    const pi_cl_dma_copy_t *copy
) {
    return dma_async_compat_l2_copy_start(future, copy);
}

int dma_contract_l2_copy_wait(dma_contract_future_t *future) {
    return dma_async_compat_l2_copy_wait(future);
}

int dma_contract_l2_copy_sync(const pi_cl_dma_copy_t *copy) {
    return dma_async_compat_l2_copy_sync(copy);
}
