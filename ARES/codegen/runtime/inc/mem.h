/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __MEM_H__
#define __MEM_H__

#include <stddef.h>

void mem_init();
struct pi_device *get_ram_ptr();
void *ram_malloc(size_t size);
void ram_free(void *ptr, size_t size);
void ram_read(void *dest, void *src, size_t size);
void ram_write(void *dest, void *src, size_t size);
void *cl_ram_malloc(size_t size);
void cl_ram_free(void *ptr, size_t size);
void cl_ram_read(void *dest, void *src, size_t size);
void cl_ram_write(void *dest, void *src, size_t size);
size_t load_file_to_ram(const void *dest, const char *filename);

#endif  // __MEM_H__

