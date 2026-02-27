/*
 * Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mem.h"
#include "pmsis.h"
#include <stdint.h>
#include "bsp/bsp.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"

// Use target-aware BSP device configuration types.
#if defined(__GAP9__)
#include "bsp/flash/spiflash.h"
typedef struct pi_mx25u51245g_conf flash_conf_t;
#define FLASH_CONF_INIT(conf_ptr) pi_mx25u51245g_conf_init(conf_ptr)
#define FLASH_DEVICE_NAME "MX25U51245G"
typedef struct pi_default_ram_conf ram_conf_t;
#define RAM_CONF_INIT(conf_ptr) pi_default_ram_conf_init(conf_ptr)
#elif defined(USE_HYPERFLASH)
#include "bsp/flash/hyperflash.h"
typedef struct pi_hyperflash_conf flash_conf_t;
#define FLASH_CONF_INIT(conf_ptr) pi_hyperflash_conf_init(conf_ptr)
#define FLASH_DEVICE_NAME "HyperFlash"
typedef struct pi_hyperram_conf ram_conf_t;
#define RAM_CONF_INIT(conf_ptr) pi_hyperram_conf_init(conf_ptr)
#else
#include "bsp/flash/spiflash.h"
typedef struct pi_spiflash_conf flash_conf_t;
#define FLASH_CONF_INIT(conf_ptr) pi_spiflash_conf_init(conf_ptr)
#define FLASH_DEVICE_NAME "SPI flash"
typedef struct pi_hyperram_conf ram_conf_t;
#define RAM_CONF_INIT(conf_ptr) pi_hyperram_conf_init(conf_ptr)
#endif
#include "bsp/ram/hyperram.h"

// --- Global Device Structures and their Configurations ---

// Flash Device and Configuration
static struct pi_device flash_dev;
static flash_conf_t flash_configuration;

// Filesystem Device and Configuration
static struct pi_device fs_dev;
static struct pi_readfs_conf fs_configuration;

// RAM Device and Configuration
static struct pi_device ram_dev;
static ram_conf_t ram_configuration;
static uint8_t ram_is_initialized = 0; // Flag to indicate successful RAM init

#ifndef MEM_C_LOAD_BUFFER_SIZE
// Larger chunks drastically reduce the overhead of FS reads + L3 writes, especially in GVSOC.
#define MEM_C_LOAD_BUFFER_SIZE 4096
#endif
PI_L2 static uint8_t mem_c_load_buffer[MEM_C_LOAD_BUFFER_SIZE];

void mem_init() {
  // Initialize flash using the configured BSP flash backend.
  FLASH_CONF_INIT(&flash_configuration);
  pi_open_from_conf(&flash_dev, &flash_configuration);
  if (pi_flash_open(&flash_dev)) {
    printf("ERROR: Cannot open flash device! Exiting...\n");
    pmsis_exit(-1);
  }
#ifndef MINIMAL_OUTPUT
  printf("INFO: %s device opened successfully.\n", FLASH_DEVICE_NAME);
#endif

  // Initialize Filesystem
  pi_readfs_conf_init(&fs_configuration);
  fs_configuration.fs.flash = &flash_dev;
  pi_open_from_conf(&fs_dev, &fs_configuration);
  if (pi_fs_mount(&fs_dev)) {
    printf("ERROR: Cannot mount filesystem! Exiting...\n");
    pmsis_exit(-2);
  }
#ifndef MINIMAL_OUTPUT
  printf("INFO: Filesystem mounted successfully.\n");
#endif

  // Initialize external RAM using target-specific BSP configuration.
  RAM_CONF_INIT(&ram_configuration);
  pi_open_from_conf(&ram_dev, &ram_configuration);
  if (pi_ram_open(&ram_dev)) {
    printf("ERROR: Cannot open RAM device! Exiting...\n");
    pmsis_exit(-3);
  }
  ram_is_initialized = 1; // Set flag
#ifndef MINIMAL_OUTPUT
  printf("INFO: RAM device opened successfully.\n");
#endif
}

struct pi_device *get_ram_ptr() {
 return &ram_dev;
}

void *ram_malloc(size_t size) {
  uint32_t addr = 0;
  if (!ram_is_initialized) {
      printf("ERROR: ram_malloc - RAM device not successfully initialized.\n");
      return NULL;
  }
  if (pi_ram_alloc(&ram_dev, &addr, size) != 0) return NULL;
  return (void *) (uintptr_t) addr;
}

void ram_free(void *ptr, size_t size) {
  if (!ram_is_initialized || ptr == NULL) return;
  pi_ram_free(&ram_dev, (uint32_t) (uintptr_t) ptr, size);
}

void ram_read(void *dest, void *src, const size_t size) {
  if (!ram_is_initialized) {
      printf("ERROR: ram_read - RAM device not successfully initialized.\n");
      return;
  }
  pi_ram_read(&ram_dev, (uint32_t) (uintptr_t) src, dest, size);
}

void ram_write(void *dest, void *src, const size_t size) {
  if (!ram_is_initialized) {
      printf("ERROR: ram_write - RAM device not successfully initialized.\n");
      return;
  }
  pi_ram_write(&ram_dev, (uint32_t) (uintptr_t) dest, src, size);
}

void *cl_ram_malloc(size_t size) {
  uint32_t addr = 0;
  pi_cl_ram_alloc_req_t req;
  if (!ram_is_initialized) {
      printf("ERROR: cl_ram_malloc - RAM device not successfully initialized.\n");
      return NULL;
  }
  pi_cl_ram_alloc(&ram_dev, size, &req);
  if (pi_cl_ram_alloc_wait(&req, &addr) != 0) return NULL;
  return (void *) (uintptr_t) addr;
}

void cl_ram_free(void *ptr, size_t size) {
  pi_cl_ram_free_req_t req;
  if (!ram_is_initialized || ptr == NULL) return;
  pi_cl_ram_free(&ram_dev, (uint32_t) (uintptr_t) ptr, size, &req);
  pi_cl_ram_free_wait(&req);
}

void cl_ram_read(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  if (!ram_is_initialized) {
    printf("ERROR: cl_ram_read - RAM device not successfully initialized.\n");
    return;
  }
  pi_cl_ram_read(&ram_dev, (uint32_t) (uintptr_t) src, dest, size, &req);
  pi_cl_ram_read_wait(&req);
}

void cl_ram_write(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  if (!ram_is_initialized) {
    printf("ERROR: cl_ram_write - RAM device not successfully initialized.\n");
    return;
  }
  pi_cl_ram_write(&ram_dev, (uint32_t) (uintptr_t) dest, src, size, &req);
  pi_cl_ram_write_wait(&req);
}

size_t load_file_to_ram(const void *dest, const char *filename) {
  pi_fs_file_t *fd = pi_fs_open(&fs_dev, filename, 0);
  if (fd == NULL) {
    printf("ERROR: load_file_to_ram - Cannot open file %s! Exiting...\n", filename);
    pmsis_exit(-4);
  }

  const size_t file_actual_size = fd->size;
  if (file_actual_size == 0) {
      pi_fs_close(fd);
      printf("INFO: load_file_to_ram - File %s is empty.\n", filename);
      return 0;
  }

  if (!ram_is_initialized) {
      printf("ERROR: load_file_to_ram - RAM device not open/initialized, cannot write file content to L3 for %s.\n", filename);
      pi_fs_close(fd);
      pmsis_exit(-6);
  }

  size_t total_bytes_written = 0;
  do {
    const size_t bytes_to_read_this_iteration = (file_actual_size - total_bytes_written > MEM_C_LOAD_BUFFER_SIZE) ?
                                          MEM_C_LOAD_BUFFER_SIZE : (file_actual_size - total_bytes_written);
    if (bytes_to_read_this_iteration == 0) break;

    const size_t actual_read_bytes = pi_fs_read(fd, mem_c_load_buffer, bytes_to_read_this_iteration);
    if (actual_read_bytes == 0 && (file_actual_size - total_bytes_written > 0)) {
        printf("ERROR: load_file_to_ram - pi_fs_read returned 0 prematurely for %s.\n", filename);
        pi_fs_close(fd);
        pmsis_exit(-5);
    }
    if (actual_read_bytes > 0) {
        ram_write((void*)((uintptr_t)dest + total_bytes_written), mem_c_load_buffer, actual_read_bytes);
        total_bytes_written += actual_read_bytes;
    }
    if (actual_read_bytes < bytes_to_read_this_iteration) break;
  } while (total_bytes_written < file_actual_size);

  pi_fs_close(fd);

  if (total_bytes_written != file_actual_size) {
      printf("WARNING: load_file_to_ram - Wrote %zu bytes but file size was %zu for %s.\n", total_bytes_written, file_actual_size, filename);
  }
  return total_bytes_written;
}
