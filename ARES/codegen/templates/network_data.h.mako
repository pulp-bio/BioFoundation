/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *path;
    size_t      size_bytes;
    uint32_t    checksum;
    const char *label;
} network_file_t;

// Binary file indices
% for idx, file in enumerate(binary_files):
#define NETWORK_FILE_IDX_${file['label'].upper().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('.', '_')} ${idx}
% endfor

// Binary file descriptors
% for file in binary_files:
extern const network_file_t g_${file['c_symbol']};
% endfor

// Helper function
void network_dump_manifest(void);

#ifdef __cplusplus
}
#endif
