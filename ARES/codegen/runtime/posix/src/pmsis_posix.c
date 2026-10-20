/*
 * ARES POSIX/Linux target runtime (see inc/pmsis.h). Compiled WITHOUT -Dmain=... so that
 * this file provides the process entry point and runs the generated program's main
 * (renamed ares_app_main) on a thread whose stack lies inside the low-4GiB arena.
 */
#define _GNU_SOURCE
#include "pmsis.h"
#include <pthread.h>
#include <sys/mman.h>
#include <time.h>
#include <errno.h>

/* ------------------------------------------------------------------ low arena */
/* Bookkeeping is kept OUT OF BAND (a separate table), and every block is followed by a
   red zone, so an overrun in generated code cannot corrupt the allocator. */
#define ALIGN 64
#define REDZONE 256
typedef struct { uintptr_t addr; size_t size; int used; } blk_t;
static uint8_t *arena_base, *arena_end, *arena_top;
static blk_t *blk; static size_t nblk, capblk;
static pthread_mutex_t arena_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t arena_peak, arena_used;

static void arena_init(void) {
    if (arena_base) return;
    size_t mb = getenv("ARES_ARENA_MB") ? (size_t)atol(getenv("ARES_ARENA_MB")) : 768;
    size_t sz = mb << 20;
    static const uintptr_t hints[] = {0x40000000u, 0x20000000u, 0x60000000u, 0x80000000u, 0x10000000u};
    for (size_t i = 0; i < sizeof(hints) / sizeof(hints[0]); ++i) {
        void *p = mmap((void *)hints[i], sz, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
        if (p != MAP_FAILED && (uintptr_t)p + sz <= 0xFFFFFFFFu) { arena_base = p; break; }
        if (p != MAP_FAILED) munmap(p, sz);
    }
    if (!arena_base) { fprintf(stderr, "ARES-POSIX: cannot map arena below 4 GiB\n"); exit(90); }
    arena_end = arena_base + sz; arena_top = arena_base;
}

static void *arena_alloc(size_t size) {
    if (!size) size = 1;
    size = (size + ALIGN - 1) & ~(size_t)(ALIGN - 1);
    pthread_mutex_lock(&arena_mu);
    arena_init();
    blk_t *best = NULL;                                   /* best-fit reuse of a freed block */
    for (size_t i = 0; i < nblk; ++i)
        if (!blk[i].used && blk[i].size >= size && (!best || blk[i].size < best->size)) best = &blk[i];
    void *p;
    if (best) { best->used = 1; p = (void *)best->addr; arena_used += best->size; }
    else {
        if (arena_top + size + REDZONE > arena_end) {
            pthread_mutex_unlock(&arena_mu);
            fprintf(stderr, "ARES-POSIX: arena exhausted (%zu bytes)\n", size); return NULL;
        }
        if (nblk == capblk) { capblk = capblk ? 2 * capblk : 1024; blk = realloc(blk, capblk * sizeof(blk_t)); }
        blk[nblk++] = (blk_t){(uintptr_t)arena_top, size, 1};
        p = arena_top; arena_top += size + REDZONE; arena_used += size;
    }
    if (arena_used > arena_peak) arena_peak = arena_used;
    pthread_mutex_unlock(&arena_mu);
    return p;
}

static void arena_free(void *p) {
    if (!p) return;
    pthread_mutex_lock(&arena_mu);
    for (size_t i = nblk; i-- > 0;)
        if (blk[i].addr == (uintptr_t)p && blk[i].used) { blk[i].used = 0; arena_used -= blk[i].size; break; }
    pthread_mutex_unlock(&arena_mu);              /* unknown / interior pointers are ignored */
}

void *pi_l2_malloc(size_t size) { return arena_alloc(size); }
void pi_l2_free(void *ptr, size_t size) { (void)size; arena_free(ptr); }
void *pi_l1_malloc(struct pi_device *dev, size_t size) { (void)dev; return arena_alloc(size); }
void pi_l1_free(struct pi_device *dev, void *ptr, size_t size) { (void)dev; (void)size; arena_free(ptr); }
void *pi_cl_l1_malloc(struct pi_device *dev, size_t size) { (void)dev; return arena_alloc(size); }
void pi_cl_l1_free(struct pi_device *dev, void *ptr, size_t size) { (void)dev; (void)size; arena_free(ptr); }

/* ------------------------------------------------------------------ team of cores */
static __thread int tls_core_id = -1;
static pthread_t team_thr[NUM_CORES];
static pthread_mutex_t team_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t team_go = PTHREAD_COND_INITIALIZER, team_done = PTHREAD_COND_INITIALIZER;
static void (*team_fn)(void *); static void *team_arg;
static int team_n, team_gen, team_left, team_started;
static pthread_barrier_t team_bar;

int pi_core_id(void) { return tls_core_id < 0 ? NUM_CORES : tls_core_id; }   /* master = core NUM_CORES */
int pi_cl_cluster_nb_cores(void) { return NUM_CORES; }

static void *team_main(void *p) {
    int id = (int)(intptr_t)p, seen = 0;
    tls_core_id = id;
    for (;;) {
        pthread_mutex_lock(&team_mu);
        while (team_gen == seen) pthread_cond_wait(&team_go, &team_mu);
        seen = team_gen;
        int run = id < team_n; void (*fn)(void *) = team_fn; void *arg = team_arg;
        pthread_mutex_unlock(&team_mu);
        if (run) fn(arg);
        pthread_mutex_lock(&team_mu);
        if (--team_left == 0) pthread_cond_signal(&team_done);
        pthread_mutex_unlock(&team_mu);
    }
    return NULL;
}

static pthread_t spawn_low_stack(void *(*fn)(void *), void *arg, size_t stack) {
    pthread_attr_t at; pthread_t t;
    pthread_attr_init(&at);
    pthread_attr_setstack(&at, arena_alloc(stack), stack);
    if (pthread_create(&t, &at, fn, arg)) { fprintf(stderr, "ARES-POSIX: pthread_create failed\n"); exit(91); }
    pthread_attr_destroy(&at);
    return t;
}

static void team_start(void) {
    if (team_started) return;
    team_started = 1;
    for (int i = 0; i < NUM_CORES; ++i) team_thr[i] = spawn_low_stack(team_main, (void *)(intptr_t)i, 8u << 20);
}

void pi_cl_team_fork(int nb, void (*entry)(void *), void *arg) {
    if (nb <= 0 || nb > NUM_CORES) nb = NUM_CORES;
    if (tls_core_id >= 0) { entry(arg); return; }         /* already inside a team */
    team_start();
    pthread_mutex_lock(&team_mu);
    pthread_barrier_init(&team_bar, NULL, (unsigned)nb);
    team_fn = entry; team_arg = arg; team_n = nb; team_left = NUM_CORES; team_gen++;
    pthread_cond_broadcast(&team_go);
    while (team_left) pthread_cond_wait(&team_done, &team_mu);
    pthread_barrier_destroy(&team_bar);
    pthread_mutex_unlock(&team_mu);
}

void pi_cl_team_barrier(void) { if (tls_core_id >= 0) pthread_barrier_wait(&team_bar); }

/* ------------------------------------------------------------------ cluster */
void pi_cluster_conf_init(struct pi_cluster_conf *c) { memset(c, 0, sizeof(*c)); }
int pi_cluster_open(struct pi_device *dev) { (void)dev; team_start(); return 0; }
int pi_cluster_close(struct pi_device *dev) { (void)dev; return 0; }
struct pi_cluster_task *pi_cluster_task(struct pi_cluster_task *t, void (*entry)(void *), void *arg) {
    memset(t, 0, sizeof(*t)); t->entry = entry; t->arg = arg; return t;
}
int pi_cluster_send_task_to_cl(struct pi_device *dev, struct pi_cluster_task *t) {
    (void)dev; t->entry(t->arg); return 0;     /* runs on the (low-stack) master thread */
}
void pi_open_from_conf(struct pi_device *dev, void *conf) { dev->config = conf; }

/* ------------------------------------------------------------------ DMA */
static inline void *P(uint32_t a) { return (void *)(uintptr_t)a; }
void pi_cl_dma_memcpy(pi_cl_dma_copy_t *c) {
    if (c->dir == PI_CL_DMA_DIR_EXT2LOC) memcpy(P(c->loc), P(c->ext), c->size);
    else memcpy(P(c->ext), P(c->loc), c->size);
}
void pi_cl_dma_memcpy_2d(pi_cl_dma_copy_2d_t *c) {   /* 2D on the EXT side, LOC contiguous */
    uint8_t *loc = P(c->loc), *ext = P(c->ext);
    for (uint32_t done = 0; done < c->size; done += c->length, loc += c->length, ext += c->stride) {
        uint32_t n = c->size - done < c->length ? c->size - done : c->length;
        if (c->dir == PI_CL_DMA_DIR_EXT2LOC) memcpy(loc, ext, n); else memcpy(ext, loc, n);
    }
}
void pi_cl_dma_cmd(uint32_t ext, uint32_t loc, uint32_t size, pi_cl_dma_dir_e dir, pi_cl_dma_copy_t *cmd) {
    pi_cl_dma_copy_t c = {.dir = dir, .size = size, .ext = ext, .loc = loc}; pi_cl_dma_memcpy(&c); (void)cmd;
}

/* ------------------------------------------------------------------ L3 RAM (same arena) */
void pi_default_ram_conf_init(struct pi_default_ram_conf *c) { (void)c; }
void pi_hyperram_conf_init(struct pi_hyperram_conf *c) { (void)c; }
int pi_ram_open(struct pi_device *dev) { (void)dev; arena_init(); return 0; }
int pi_ram_close(struct pi_device *dev) { (void)dev; return 0; }
int pi_ram_alloc(struct pi_device *dev, uint32_t *addr, uint32_t size) {
    (void)dev; void *p = arena_alloc(size); *addr = (uint32_t)(uintptr_t)p; return p ? 0 : -1;
}
int pi_ram_free(struct pi_device *dev, uint32_t addr, uint32_t size) { (void)dev; (void)size; arena_free(P(addr)); return 0; }
void pi_ram_read(struct pi_device *dev, uint32_t a, void *d, uint32_t s) { (void)dev; memcpy(d, P(a), s); }
void pi_ram_write(struct pi_device *dev, uint32_t a, void *d, uint32_t s) { (void)dev; memcpy(P(a), d, s); }
void pi_cl_ram_read(struct pi_device *dev, uint32_t a, void *d, uint32_t s, pi_cl_ram_req_t *r) { (void)dev; (void)r; memcpy(d, P(a), s); }
void pi_cl_ram_write(struct pi_device *dev, uint32_t a, void *d, uint32_t s, pi_cl_ram_req_t *r) { (void)dev; (void)r; memcpy(P(a), d, s); }
void pi_cl_ram_read_2d(struct pi_device *dev, uint32_t a, void *d, uint32_t size, uint32_t stride, uint32_t len, pi_cl_ram_req_t *r) {
    (void)dev; (void)r; uint8_t *dst = d, *src = P(a);
    for (uint32_t done = 0; done < size; done += len, dst += len, src += stride)
        memcpy(dst, src, size - done < len ? size - done : len);
}
void pi_cl_ram_alloc(struct pi_device *dev, uint32_t size, pi_cl_ram_alloc_req_t *r) { r->err = pi_ram_alloc(dev, &r->addr, size); }
int pi_cl_ram_alloc_wait(pi_cl_ram_alloc_req_t *r, uint32_t *addr) { *addr = r->addr; return r->err; }
void pi_cl_ram_free(struct pi_device *dev, uint32_t a, uint32_t s, pi_cl_ram_free_req_t *r) { (void)r; pi_ram_free(dev, a, s); }

/* ------------------------------------------------------------------ flash + readfs */
void pi_hyperflash_conf_init(struct pi_hyperflash_conf *c) { (void)c; }
void pi_spiflash_conf_init(struct pi_spiflash_conf *c) { (void)c; }
void pi_mx25u51245g_conf_init(struct pi_mx25u51245g_conf *c) { (void)c; }
void pi_readfs_conf_init(struct pi_readfs_conf *c) { memset(c, 0, sizeof(*c)); }
int pi_flash_open(struct pi_device *dev) { (void)dev; return 0; }
int pi_fs_mount(struct pi_device *dev) { (void)dev; return 0; }
void pi_fs_unmount(struct pi_device *dev) { (void)dev; }
pi_fs_file_t *pi_fs_open(struct pi_device *dev, const char *name, int flags) {
    (void)dev; (void)flags;
    const char *dir = getenv("ARES_FS_DIR") ? getenv("ARES_FS_DIR") : ".";
    char path[4096]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    pi_fs_file_t *f = calloc(1, sizeof(*f));
    f->fp = fp; fseek(fp, 0, SEEK_END); f->size = (uint32_t)ftell(fp); fseek(fp, 0, SEEK_SET);
    return f;
}
int pi_fs_read(pi_fs_file_t *f, void *buf, uint32_t size) { int n = (int)fread(buf, 1, size, f->fp); f->offset += n; return n; }
void pi_fs_seek(pi_fs_file_t *f, uint32_t off) { fseek(f->fp, off, SEEK_SET); f->offset = off; }
void pi_fs_close(pi_fs_file_t *f) { if (f) { fclose(f->fp); free(f); } }

/* ------------------------------------------------------------------ perf (ns) */
static __thread uint64_t perf_t0;
static uint64_t now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec; }
void pi_perf_conf(uint32_t e) { (void)e; }
void pi_perf_reset(void) { perf_t0 = now_ns(); }
void pi_perf_start(void) { if (!perf_t0) perf_t0 = now_ns(); }
void pi_perf_stop(void) {}
uint32_t pi_perf_read(int event) { return event == PI_PERF_CYCLES || event == PI_PERF_ACTIVE_CYCLES ? (uint32_t)(now_ns() - perf_t0) : 0; }

/* ------------------------------------------------------------------ entry */
static int app_ret;
int ares_app_main(void);
static void *app_thread(void *p) { (void)p; app_ret = ares_app_main(); return NULL; }
void pmsis_exit(int code) { fflush(stdout); exit(code); }
int pmsis_kickoff(void *entry) { ((void (*)(void))entry)(); return 0; }

int main(void) {
    arena_init();
    perf_t0 = now_ns();
    uint64_t t0 = now_ns();
    pthread_t t = spawn_low_stack(app_thread, NULL, 64u << 20);
    pthread_join(t, NULL);
    fprintf(stderr, "ARES-POSIX: wall %.3f ms, arena peak %.2f MiB, cores %d\n",
            (now_ns() - t0) / 1e6, arena_peak / 1048576.0, NUM_CORES);
    return app_ret;
}
