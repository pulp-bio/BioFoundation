/*
 * ARES POSIX/Linux target: drop-in replacement for the PULP/GAP9 PMSIS API subset used by
 * ARES-generated networks, so the SAME generated C runs on Linux hosts and on the Kria
 * KV260's Arm Cortex-A53 cores (and an x86-64 laptop for testing).
 *
 * Mapping:
 *   cluster cores        -> NUM_CORES POSIX threads (pi_core_id = thread-local id)
 *   pi_cl_team_fork      -> run fn on all team threads, return when all finish
 *   pi_cl_team_barrier   -> pthread barrier over the team
 *   L1 / L2 / L3 memory  -> one arena mapped BELOW 4 GiB (ARES casts pointers to uint32_t)
 *   cluster DMA, L3 RAM  -> synchronous memcpy
 *   readfs on flash      -> files in $ARES_FS_DIR (default ".")
 *   pi_perf_read(CYCLES) -> nanoseconds from CLOCK_MONOTONIC (reported as "cycles")
 *   RISC-V SIMD dotp     -> portable C (auto-vectorised by the compiler)
 * Link non-PIE so static data also lies below 4 GiB; main() is renamed and run on a thread
 * whose stack is in the arena (see Makefile.posix).
 */
#ifndef ARES_POSIX_PMSIS_H
#define ARES_POSIX_PMSIS_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ARES_TARGET_POSIX 1

#ifndef NUM_CORES
#define NUM_CORES 4
#endif

/* RISC-V `fence` used verbatim in ARES sources -> native barrier via an assembler macro.
   (Do not build with -flto: each translation unit defines the macro once.) */
#if defined(__aarch64__)
__asm__(".macro fence args:vararg\n dmb ish\n .endm\n");
#elif defined(__x86_64__) || defined(__i386__)
__asm__(".macro fence args:vararg\n mfence\n .endm\n");
#elif defined(__arm__)
__asm__(".macro fence args:vararg\n dmb\n .endm\n");
#endif

/* ---- section attributes (all memory is ordinary RAM here) ---- */
#define PI_L2
#define PI_L1
#define PI_CL_L1
#define PI_FC_L1
#define PI_FC_TINY
#define PI_INLINE_CL_DMA_0 static inline

/* ---- PULP SIMD types and builtins ---- */
typedef signed char v4s __attribute__((vector_size(4)));
typedef unsigned char v4u __attribute__((vector_size(4)));
typedef short v2s __attribute__((vector_size(4)));

static inline int32_t ares_sdotsp4(v4s a, v4s b, int32_t c) {
    return c + (int32_t)a[0] * b[0] + (int32_t)a[1] * b[1] + (int32_t)a[2] * b[2] + (int32_t)a[3] * b[3];
}
static inline int32_t ares_sdotusp4(v4u a, v4s b, int32_t c) {
    return c + (int32_t)a[0] * b[0] + (int32_t)a[1] * b[1] + (int32_t)a[2] * b[2] + (int32_t)a[3] * b[3];
}
#define __builtin_pulp_sdotsp4(a, b, c) ares_sdotsp4((a), (b), (c))
#define __builtin_pulp_sdotusp4(a, b, c) ares_sdotusp4((v4u)(a), (b), (c))
/* p.clipr: clamp to [-(bound+1), bound] */
static inline int32_t ares_clip_r(int32_t x, int32_t bound) {
    return x > bound ? bound : (x < -bound - 1 ? -bound - 1 : x);
}
#define __builtin_pulp_clip_r(x, b) ares_clip_r((x), (b))

/* ---- devices ---- */
struct pi_device { void *config; void *data; int id; };
void pi_open_from_conf(struct pi_device *dev, void *conf);
void pmsis_exit(int code);
int pmsis_kickoff(void *entry);

/* ---- cluster ---- */
struct pi_cluster_conf { int id; int cc_stack_size; int icache_conf; };
#define PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE 0
#define PI_CLUSTER_ICACHE_PREFETCH_ENABLE 0
#define PI_CLUSTER_ICACHE_ENABLE 0
struct pi_cluster_task {
    void (*entry)(void *); void *arg; void *stack_ptr;
    uint32_t stack_size; uint32_t slave_stack_size; int nb_cores;
};
void pi_cluster_conf_init(struct pi_cluster_conf *conf);
int pi_cluster_open(struct pi_device *dev);
int pi_cluster_close(struct pi_device *dev);
struct pi_cluster_task *pi_cluster_task(struct pi_cluster_task *task, void (*entry)(void *), void *arg);
int pi_cluster_send_task_to_cl(struct pi_device *dev, struct pi_cluster_task *task);
static inline int pi_cluster_send_task(struct pi_device *dev, struct pi_cluster_task *t) { return pi_cluster_send_task_to_cl(dev, t); }

int pi_core_id(void);
static inline int pi_cluster_id(void) { return 0; }
static inline int pi_is_fc(void) { return 0; }
int pi_cl_cluster_nb_cores(void);
void pi_cl_team_fork(int nb_cores, void (*entry)(void *), void *arg);
void pi_cl_team_barrier(void);
static inline int pi_cl_team_nb_cores(void) { return pi_cl_cluster_nb_cores(); }

/* ---- memory ---- */
void *pi_l2_malloc(size_t size);
void pi_l2_free(void *ptr, size_t size);
void *pi_l1_malloc(struct pi_device *dev, size_t size);
void pi_l1_free(struct pi_device *dev, void *ptr, size_t size);
void *pi_cl_l1_malloc(struct pi_device *dev, size_t size);
void pi_cl_l1_free(struct pi_device *dev, void *ptr, size_t size);
static inline void *pi_fc_l1_malloc(size_t size) { return pi_l2_malloc(size); }
static inline void pi_fc_l1_free(void *p, size_t size) { pi_l2_free(p, size); }

/* ---- cluster DMA (synchronous) ---- */
typedef enum { PI_CL_DMA_DIR_LOC2EXT = 0, PI_CL_DMA_DIR_EXT2LOC = 1 } pi_cl_dma_dir_e;
typedef struct {
    pi_cl_dma_dir_e dir; int merge; uint32_t size; int id; uint32_t ext; uint32_t loc;
    uint32_t stride; uint32_t length;
} pi_cl_dma_copy_t;
typedef pi_cl_dma_copy_t pi_cl_dma_copy_2d_t;
void pi_cl_dma_memcpy(pi_cl_dma_copy_t *copy);
void pi_cl_dma_memcpy_2d(pi_cl_dma_copy_2d_t *copy);
static inline void pi_cl_dma_wait(void *copy) { (void)copy; }
static inline void pi_cl_dma_flush(void) {}
void pi_cl_dma_cmd(uint32_t ext, uint32_t loc, uint32_t size, pi_cl_dma_dir_e dir, pi_cl_dma_copy_t *cmd);
static inline void pi_cl_dma_cmd_wait(pi_cl_dma_copy_t *c) { (void)c; }

/* ---- L3 RAM ---- */
typedef struct { int done; } pi_cl_ram_req_t;
typedef struct { uint32_t addr; int err; } pi_cl_ram_alloc_req_t;
typedef struct { int done; } pi_cl_ram_free_req_t;
struct pi_default_ram_conf { int dummy; };
struct pi_hyperram_conf { int dummy; };
void pi_default_ram_conf_init(struct pi_default_ram_conf *c);
void pi_hyperram_conf_init(struct pi_hyperram_conf *c);
int pi_ram_open(struct pi_device *dev);
int pi_ram_close(struct pi_device *dev);
int pi_ram_alloc(struct pi_device *dev, uint32_t *addr, uint32_t size);
int pi_ram_free(struct pi_device *dev, uint32_t addr, uint32_t size);
void pi_ram_read(struct pi_device *dev, uint32_t addr, void *data, uint32_t size);
void pi_ram_write(struct pi_device *dev, uint32_t addr, void *data, uint32_t size);
void pi_cl_ram_read(struct pi_device *dev, uint32_t addr, void *data, uint32_t size, pi_cl_ram_req_t *req);
void pi_cl_ram_write(struct pi_device *dev, uint32_t addr, void *data, uint32_t size, pi_cl_ram_req_t *req);
void pi_cl_ram_read_2d(struct pi_device *dev, uint32_t addr, void *data, uint32_t size, uint32_t stride, uint32_t length, pi_cl_ram_req_t *req);
static inline int pi_cl_ram_read_wait(pi_cl_ram_req_t *r) { (void)r; return 0; }
static inline int pi_cl_ram_write_wait(pi_cl_ram_req_t *r) { (void)r; return 0; }
void pi_cl_ram_alloc(struct pi_device *dev, uint32_t size, pi_cl_ram_alloc_req_t *req);
int pi_cl_ram_alloc_wait(pi_cl_ram_alloc_req_t *req, uint32_t *addr);
void pi_cl_ram_free(struct pi_device *dev, uint32_t addr, uint32_t size, pi_cl_ram_free_req_t *req);
static inline int pi_cl_ram_free_wait(pi_cl_ram_free_req_t *r) { (void)r; return 0; }

/* ---- flash + read-only filesystem ---- */
struct pi_hyperflash_conf { int dummy; };
struct pi_spiflash_conf { int dummy; };
struct pi_mx25u51245g_conf { int dummy; };
struct pi_readfs_conf { struct { struct pi_device *flash; } fs; };
typedef struct { FILE *fp; uint32_t size; uint32_t offset; } pi_fs_file_t;
void pi_hyperflash_conf_init(struct pi_hyperflash_conf *c);
void pi_spiflash_conf_init(struct pi_spiflash_conf *c);
void pi_mx25u51245g_conf_init(struct pi_mx25u51245g_conf *c);
void pi_readfs_conf_init(struct pi_readfs_conf *c);
int pi_flash_open(struct pi_device *dev);
int pi_fs_mount(struct pi_device *dev);
void pi_fs_unmount(struct pi_device *dev);
pi_fs_file_t *pi_fs_open(struct pi_device *dev, const char *name, int flags);
int pi_fs_read(pi_fs_file_t *f, void *buf, uint32_t size);
void pi_fs_seek(pi_fs_file_t *f, uint32_t offset);
void pi_fs_close(pi_fs_file_t *f);

/* ---- performance counters ---- */
enum { PI_PERF_CYCLES = 0, PI_PERF_ACTIVE_CYCLES = 1, PI_PERF_INSTR = 2, PI_PERF_LD_STALL = 3,
       PI_PERF_JR_STALL = 4, PI_PERF_IMISS = 5, PI_PERF_LD = 6, PI_PERF_ST = 7, PI_PERF_TCDM_CONT = 8 };
void pi_perf_conf(uint32_t events);
void pi_perf_reset(void);
void pi_perf_start(void);
void pi_perf_stop(void);
uint32_t pi_perf_read(int event);

/* ---- misc ---- */
static inline void pi_time_wait_us(int us) { (void)us; }
static inline uint32_t pi_freq_get(int dom) { (void)dom; return 1000000000u; }
#define PI_FREQ_DOMAIN_FC 0
#define PI_FREQ_DOMAIN_CL 1

#endif /* ARES_POSIX_PMSIS_H */
