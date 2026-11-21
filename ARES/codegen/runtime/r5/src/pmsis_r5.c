/* SPDX-License-Identifier: Apache-2.0 */
/* ARES PMSIS shim for bare-metal Cortex-R5F (Zynq UltraScale+ RPU0), single core.
 * Same API as the POSIX shim (posix/inc/pmsis.h):
 *   cluster cores   -> the calling R5 core (NUM_CORES = 1; fork = direct call)
 *   L1/L2/L3        -> one bump/best-fit arena in DDR (ARENA_BASE .. +ARENA_SIZE)
 *   DMA             -> memcpy
 *   read-only FS    -> a blob preloaded by U-Boot at FS_BASE (see tools/pack_fs.py)
 *   perf "cycles"   -> R5 PMU cycle counter (PMCCNTR, /64 prescaler, returned in CPU cycles)
 *   stdout          -> UART1 (0xFF010000, the KV260 console), and a copy in DDR at LOG_BASE
 * At exit a result record is left at MBOX_BASE for U-Boot "md". */
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>

#define ARENA_BASE 0x20000000u
#define ARENA_SIZE 0x10000000u            /* 256 MB */
#define FS_BASE    0x30000000u
#define LOG_BASE   0x3E000000u            /* 1 MB console copy */
#define LOG_SIZE   0x00100000u
#define MBOX_BASE  0x3FF00000u
#define UART_BASE  0xFF010000u

/* ---------------------------------------------------------------- console + newlib */
static volatile uint32_t *const uart_sr = (volatile uint32_t *)(UART_BASE + 0x2C);
static volatile uint32_t *const uart_fifo = (volatile uint32_t *)(UART_BASE + 0x30);
static uint32_t log_pos;
static void putc_uart(char c) {
    while (*uart_sr & 0x10) {}               /* TXFULL */
    *uart_fifo = (uint32_t)(uint8_t)c;
}
static void out_char(char c) {
    if (c == '\n') putc_uart('\r');
    putc_uart(c);
    if (log_pos < LOG_SIZE - 1) { ((volatile char *)LOG_BASE)[log_pos++] = c; ((volatile char *)LOG_BASE)[log_pos] = 0; }
}
int _write(int fd, const char *p, int n) { (void)fd; for (int i = 0; i < n; i++) out_char(p[i]); return n; }
int _read(int fd, char *p, int n) { (void)fd; (void)p; (void)n; return 0; }
int _close(int fd) { (void)fd; return -1; }
int _lseek(int fd, int o, int w) { (void)fd; (void)o; (void)w; return 0; }
int _fstat(int fd, struct stat *st) { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd) { (void)fd; return 1; }
int _getpid(void) { return 1; }
int _kill(int p, int s) { (void)p; (void)s; errno = EINVAL; return -1; }
extern char end;
extern char __heap_limit;
void *_sbrk(int incr) {
    static char *brk = &end;
    char *prev = brk;
    if (brk + incr > &__heap_limit) { errno = ENOMEM; return (void *)-1; }
    brk += incr; return prev;
}

/* ---------------------------------------------------------------- PMU cycle counter */
static inline void pmu_init(void) {
    uint32_t v = (1u << 0) | (1u << 2) | (1u << 3);   /* E, C reset, D (count every 64 cycles) */
    __asm__ volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(v));
    v = 1u << 31;                                     /* enable cycle counter */
    __asm__ volatile("mcr p15, 0, %0, c9, c12, 1" :: "r"(v));
}
static inline uint32_t pmu_read64(void) { uint32_t v; __asm__ volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(v)); return v; }
static uint32_t perf_t0;
void pi_perf_conf(uint32_t e) { (void)e; }
void pi_perf_reset(void) { perf_t0 = pmu_read64(); }
void pi_perf_start(void) {}
void pi_perf_stop(void) {}
uint32_t pi_perf_read(int event) { (void)event; return (pmu_read64() - perf_t0) * 64u; }  /* CPU cycles */

/* ---------------------------------------------------------------- arena */
#define ALIGN 64
typedef struct { uint32_t addr, size; int used; } blk_t;
#define MAXBLK 8192
static blk_t blk[MAXBLK]; static uint32_t nblk;
static uint8_t *arena_top = (uint8_t *)ARENA_BASE;
static uint32_t arena_used, arena_peak;
static void *arena_alloc(size_t size) {
    if (!size) size = 1;
    size = (size + ALIGN - 1) & ~(size_t)(ALIGN - 1);
    blk_t *best = 0;
    for (uint32_t i = 0; i < nblk; ++i)
        if (!blk[i].used && blk[i].size >= size && (!best || blk[i].size < best->size)) best = &blk[i];
    void *p;
    if (best) { best->used = 1; p = (void *)best->addr; arena_used += best->size; }
    else {
        if ((uint32_t)arena_top + size > ARENA_BASE + ARENA_SIZE || nblk == MAXBLK) { printf("ARES-R5: arena exhausted\n"); return 0; }
        blk[nblk++] = (blk_t){(uint32_t)arena_top, (uint32_t)size, 1};
        p = arena_top; arena_top += size; arena_used += size;
    }
    if (arena_used > arena_peak) arena_peak = arena_used;
    return p;
}
static void arena_free(void *p) {
    if (!p) return;
    for (uint32_t i = nblk; i-- > 0;)
        if (blk[i].addr == (uint32_t)p && blk[i].used) { blk[i].used = 0; arena_used -= blk[i].size; break; }
}
#include "pmsis.h"
void *pi_l2_malloc(size_t size) { return arena_alloc(size); }
void pi_l2_free(void *ptr, size_t size) { (void)size; arena_free(ptr); }
void *pi_l1_malloc(struct pi_device *dev, size_t size) { (void)dev; return arena_alloc(size); }
void pi_l1_free(struct pi_device *dev, void *ptr, size_t size) { (void)dev; (void)size; arena_free(ptr); }
void *pi_cl_l1_malloc(struct pi_device *dev, size_t size) { (void)dev; return arena_alloc(size); }
void pi_cl_l1_free(struct pi_device *dev, void *ptr, size_t size) { (void)dev; (void)size; arena_free(ptr); }

/* ---------------------------------------------------------------- cluster (single core) */
static int cur_core = -1;
int pi_core_id(void) { return cur_core < 0 ? NUM_CORES : cur_core; }
int pi_cl_cluster_nb_cores(void) { return NUM_CORES; }
void pi_cl_team_fork(int nb, void (*entry)(void *), void *arg) {
    (void)nb;
    for (int c = 0; c < NUM_CORES; ++c) { cur_core = c; entry(arg); }   /* NUM_CORES == 1 */
    cur_core = -1;
}
void pi_cl_team_barrier(void) {}
void pi_cluster_conf_init(struct pi_cluster_conf *c) { memset(c, 0, sizeof(*c)); }
int pi_cluster_open(struct pi_device *dev) { (void)dev; return 0; }
int pi_cluster_close(struct pi_device *dev) { (void)dev; return 0; }
struct pi_cluster_task *pi_cluster_task(struct pi_cluster_task *t, void (*entry)(void *), void *arg) {
    memset(t, 0, sizeof(*t)); t->entry = entry; t->arg = arg; return t;
}
int pi_cluster_send_task_to_cl(struct pi_device *dev, struct pi_cluster_task *t) { (void)dev; t->entry(t->arg); return 0; }
void pi_open_from_conf(struct pi_device *dev, void *conf) { dev->config = conf; }

/* ---------------------------------------------------------------- DMA + L3 */
static inline void *P(uint32_t a) { return (void *)a; }
void pi_cl_dma_memcpy(pi_cl_dma_copy_t *c) {
    if (c->dir == PI_CL_DMA_DIR_EXT2LOC) memcpy(P(c->loc), P(c->ext), c->size); else memcpy(P(c->ext), P(c->loc), c->size);
}
void pi_cl_dma_memcpy_2d(pi_cl_dma_copy_2d_t *c) {
    uint32_t done = 0, ext = c->ext, loc = c->loc;
    while (done < c->size) {
        uint32_t n = c->length; if (n > c->size - done) n = c->size - done;
        if (c->dir == PI_CL_DMA_DIR_EXT2LOC) memcpy(P(loc), P(ext), n); else memcpy(P(ext), P(loc), n);
        done += n; loc += n; ext += c->stride;
    }
}
void pi_cl_dma_cmd(uint32_t ext, uint32_t loc, uint32_t size, pi_cl_dma_dir_e dir, pi_cl_dma_copy_t *cmd) {
    if (dir == PI_CL_DMA_DIR_EXT2LOC) memcpy(P(loc), P(ext), size); else memcpy(P(ext), P(loc), size);
    if (cmd) { cmd->ext = ext; cmd->loc = loc; cmd->size = size; cmd->dir = dir; }
}
void pi_default_ram_conf_init(struct pi_default_ram_conf *c) { (void)c; }
void pi_hyperram_conf_init(struct pi_hyperram_conf *c) { (void)c; }
int pi_ram_open(struct pi_device *dev) { (void)dev; return 0; }
int pi_ram_close(struct pi_device *dev) { (void)dev; return 0; }
int pi_ram_alloc(struct pi_device *dev, uint32_t *addr, uint32_t size) { (void)dev; void *p = arena_alloc(size); *addr = (uint32_t)p; return p ? 0 : -1; }
int pi_ram_free(struct pi_device *dev, uint32_t addr, uint32_t size) { (void)dev; (void)size; arena_free(P(addr)); return 0; }
void pi_ram_read(struct pi_device *dev, uint32_t a, void *d, uint32_t s) { (void)dev; memcpy(d, P(a), s); }
void pi_ram_write(struct pi_device *dev, uint32_t a, void *d, uint32_t s) { (void)dev; memcpy(P(a), d, s); }
void pi_cl_ram_read(struct pi_device *dev, uint32_t a, void *d, uint32_t s, pi_cl_ram_req_t *r) { (void)dev; (void)r; memcpy(d, P(a), s); }
void pi_cl_ram_write(struct pi_device *dev, uint32_t a, void *d, uint32_t s, pi_cl_ram_req_t *r) { (void)dev; (void)r; memcpy(P(a), d, s); }
void pi_cl_ram_read_2d(struct pi_device *dev, uint32_t a, void *d, uint32_t size, uint32_t stride, uint32_t len, pi_cl_ram_req_t *r) {
    (void)dev; (void)r; uint8_t *dst = d; uint32_t done = 0;
    while (done < size) { uint32_t n = len; if (n > size - done) n = size - done; memcpy(dst + done, P(a), n); done += n; a += stride; }
}
void pi_cl_ram_alloc(struct pi_device *dev, uint32_t size, pi_cl_ram_alloc_req_t *r) { r->err = pi_ram_alloc(dev, &r->addr, size); }
int pi_cl_ram_alloc_wait(pi_cl_ram_alloc_req_t *r, uint32_t *addr) { *addr = r->addr; return r->err; }
void pi_cl_ram_free(struct pi_device *dev, uint32_t a, uint32_t s, pi_cl_ram_free_req_t *r) { (void)r; pi_ram_free(dev, a, s); }

/* ---------------------------------------------------------------- read-only FS from a DDR blob
 * blob: "ARESFS01", u32 count, then count x {char name[120]; u32 offset; u32 size;}, data (offsets from blob start) */
typedef struct { char name[120]; uint32_t off, size; } fs_ent_t;
void pi_hyperflash_conf_init(struct pi_hyperflash_conf *c) { (void)c; }
void pi_spiflash_conf_init(struct pi_spiflash_conf *c) { (void)c; }
void pi_mx25u51245g_conf_init(struct pi_mx25u51245g_conf *c) { (void)c; }
void pi_readfs_conf_init(struct pi_readfs_conf *c) { memset(c, 0, sizeof(*c)); }
int pi_flash_open(struct pi_device *dev) { (void)dev; return 0; }
int pi_fs_mount(struct pi_device *dev) { (void)dev; return memcmp((void *)FS_BASE, "ARESFS01", 8) ? -1 : 0; }
void pi_fs_unmount(struct pi_device *dev) { (void)dev; }
pi_fs_file_t *pi_fs_open(struct pi_device *dev, const char *name, int flags) {
    (void)dev; (void)flags;
    if (memcmp((void *)FS_BASE, "ARESFS01", 8)) { printf("ARES-R5: no FS blob at 0x%08x\n", (unsigned)FS_BASE); return 0; }
    uint32_t n = *(uint32_t *)(FS_BASE + 8);
    fs_ent_t *e = (fs_ent_t *)(FS_BASE + 12);
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].name, name)) {
            pi_fs_file_t *f = calloc(1, sizeof(*f));
            f->fp = (FILE *)(FS_BASE + e[i].off); f->size = e[i].size; f->offset = 0;
            return f;
        }
    printf("ARES-R5: file not found: %s\n", name);
    return 0;
}
int pi_fs_read(pi_fs_file_t *f, void *buf, uint32_t size) {
    if (f->offset + size > f->size) size = f->size - f->offset;
    memcpy(buf, (uint8_t *)f->fp + f->offset, size); f->offset += size; return (int)size;
}
void pi_fs_seek(pi_fs_file_t *f, uint32_t off) { f->offset = off; }
void pi_fs_close(pi_fs_file_t *f) { free(f); }

/* ---------------------------------------------------------------- entry + mailbox */
typedef struct { uint32_t magic, state, ret, cycles64_total, arena_peak; } mbox_t;   /* magic 'R5OK' */
static volatile mbox_t *const mbox = (volatile mbox_t *)MBOX_BASE;
void pmsis_exit(int code) { fflush(stdout); mbox->ret = (uint32_t)code; mbox->state = 2; for (;;) __asm__ volatile("wfi"); }
int pmsis_kickoff(void *entry) { ((void (*)(void))entry)(); return 0; }
int ares_app_main(void);
void r5_halt(void) { mbox->state = 2; }
int main(void) {
    mbox->magic = 0x4B4F3552u; mbox->state = 1; mbox->ret = 0xFFFFFFFFu;
#ifdef R5_QEMU
    *(volatile uint32_t *)(UART_BASE + 0x0) = 0x14;   /* QEMU: UART comes up with TX disabled */
#endif
    pmu_init();
    printf("\nARES-R5: start (Cortex-R5F, %d core)\n", NUM_CORES);
    uint32_t t0 = pmu_read64();
    int r = ares_app_main();
    uint32_t t1 = pmu_read64();
    mbox->cycles64_total = t1 - t0; mbox->arena_peak = arena_peak; mbox->ret = (uint32_t)r;
    printf("ARES-R5: done ret=%d, wall %lu cycles, arena peak %lu bytes\n", r, (unsigned long)(t1 - t0) * 64ul, (unsigned long)arena_peak);
    fflush(stdout);
    mbox->state = 2;
    for (;;) __asm__ volatile("wfi");
}

/* newlib hooks needed with -nostartfiles */
void _init(void) {}
void _fini(void) {}
void _exit(int code) { pmsis_exit(code); for (;;) {} }

/* exception reporter (called from crt0 vectors): type, return address */
void r5_fault(uint32_t type, uint32_t lr) {
    uint32_t dfsr, dfar, ifsr, ifar;
    __asm__ volatile("mrc p15, 0, %0, c5, c0, 0" : "=r"(dfsr));
    __asm__ volatile("mrc p15, 0, %0, c6, c0, 0" : "=r"(dfar));
    __asm__ volatile("mrc p15, 0, %0, c5, c0, 1" : "=r"(ifsr));
    __asm__ volatile("mrc p15, 0, %0, c6, c0, 2" : "=r"(ifar));
    static const char *nm[] = {"?", "undef", "svc", "prefetch abort", "data abort", "?", "irq", "fiq"};
    char buf[160];
    int n = snprintf(buf, sizeof buf, "\nARES-R5 FAULT: %s at lr=0x%08lx DFSR=0x%08lx DFAR=0x%08lx IFSR=0x%08lx IFAR=0x%08lx\n",
                     nm[type & 7], (unsigned long)lr, (unsigned long)dfsr, (unsigned long)dfar, (unsigned long)ifsr, (unsigned long)ifar);
    for (int i = 0; i < n; i++) out_char(buf[i]);
    mbox->state = 0xDEAD0000u | type; mbox->ret = lr;
    for (;;) __asm__ volatile("wfi");
}
