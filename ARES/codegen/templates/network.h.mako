/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <stdint.h>
#include <stddef.h>
#include "pmsis.h"

#ifdef __cplusplus
extern "C" {
#endif

// Run network test from L3 RAM
// Returns 0 on success, non-zero on failure
int network_run_test_from_l3(
    void *input_l3,
% if additional_input_entries:
    void *additional_inputs_l3[],  // Additional inputs for multi-input models
    size_t num_additional_inputs,
% endif
    void *weights_l3[],
    size_t num_weights,
    void *biases_l3[],
    size_t num_biases,
    void *golden_l3,
    void *intermediate_golden_l3[],  // May be NULL when no intermediate references
    size_t num_intermediate,
    void *l3_activation_buffers[],  // L3 fallback buffers (NULL if not needed)
<%
    ne16_param_layers_h = [l for l in param_layers if l.get('ne16_eligible', False)]
%>
    size_t num_l3_activations${''.join(', void *' + entry['c_name'] + '_ssm_params_l3[]' for entry in ssm_entries)}${''.join(', void *' + entry['c_name'] + '_mamba_params_l3[]' for entry in mamba_block_entries)}\
${', void *ne16_packed_l3[], void *ne16_bias_corr_l3[], void *ne16_hw_scale_l3[], void *ne16_hw_scale_shift_l3[], size_t num_ne16' if ne16_param_layers_h else ''}${''.join(', void *' + entry['c_name'] + '_alt_attn_ne16_l3[]' for entry in alt_attn_ne16_entries)},
    struct pi_device *cluster_dev
);

#ifdef __cplusplus
}
#endif
