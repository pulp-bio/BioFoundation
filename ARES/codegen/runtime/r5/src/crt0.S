/* SPDX-License-Identifier: Apache-2.0 */
/* Cortex-R5F bare-metal start-up for ARES on the Zynq UltraScale+ RPU (R5_0).
 * Entered from a 2-word stub in ATCM (ldr pc,=_start) after U-Boot "cpu 4 release 0 split".
 * Sets the stack, enables the VFP, programs the MPU (DDR normal write-back, peripherals device),
 * enables I/D caches, clears .bss and calls main(). */
    .syntax unified
    .arm
    .section .text.start, "ax"
    .global _start
_start:
    cpsid   if
    /* SVC mode stack; abort/undef modes share a small stack below it */
    cps     #0x17                        /* abort */
    ldr     sp, =(__stack_top - 0x10000)
    cps     #0x1B                        /* undef */
    ldr     sp, =(__stack_top - 0x10000)
    cps     #0x13                        /* svc */
    ldr     sp, =__stack_top
    /* install exception vectors at 0x0 (ATCM, low vectors) */
    ldr     r0, =r5_vectors
    mov     r1, #0
    mov     r2, #16
3:  ldr     r3, [r0], #4
    str     r3, [r1], #4
    subs    r2, r2, #1
    bne     3b
    dsb
    isb
    /* disable MPU and caches while configuring */
    mrc     p15, 0, r0, c1, c0, 0
    bic     r0, r0, #0x1                 /* M */
    bic     r0, r0, #0x4                 /* C */
    bic     r0, r0, #0x1000              /* I */
    dsb
    mcr     p15, 0, r0, c1, c0, 0
    isb
    /* invalidate caches: I-cache all, D-cache all (Cortex-R5 specific c15,c5,0) */
    mov     r0, #0
    mcr     p15, 0, r0, c7, c5, 0
    mcr     p15, 0, r0, c15, c5, 0
    dsb
    isb
    /* enable VFP: CPACR cp10/cp11 full access, then FPEXC.EN */
    mrc     p15, 0, r0, c1, c0, 2
    orr     r0, r0, #(0xF << 20)
    mcr     p15, 0, r0, c1, c0, 2
    isb
    mov     r0, #0x40000000
    vmsr    fpexc, r0
    /* MPU regions: base, size field (log2(size)-1)<<1 | enable, access control */
    /* region 0: 0x00000000, 2 GB DDR, normal WB-WA, RW, executable */
    mov     r0, #0
    mcr     p15, 0, r0, c6, c2, 0
    ldr     r1, =0x00000000
    mcr     p15, 0, r1, c6, c1, 0
    ldr     r1, =0x30B                    /* AP=011 full, TEX=001 C=1 B=1 */
    mcr     p15, 0, r1, c6, c1, 4
    ldr     r1, =((30 << 1) | 1)          /* 2^31 */
    mcr     p15, 0, r1, c6, c1, 2
    /* region 1: 0x80000000, 1 GB (PL), device, XN */
    mov     r0, #1
    mcr     p15, 0, r0, c6, c2, 0
    ldr     r1, =0x80000000
    mcr     p15, 0, r1, c6, c1, 0
    ldr     r1, =0x1305                   /* XN, AP=011, TEX=000 C=0 B=1 (shareable device: S=1) */
    mcr     p15, 0, r1, c6, c1, 4
    ldr     r1, =((29 << 1) | 1)          /* 2^30 */
    mcr     p15, 0, r1, c6, c1, 2
    /* region 2: 0xC0000000, 512 MB (QSPI/PCIe low), device, XN */
    mov     r0, #2
    mcr     p15, 0, r0, c6, c2, 0
    ldr     r1, =0xC0000000
    mcr     p15, 0, r1, c6, c1, 0
    ldr     r1, =0x1305
    mcr     p15, 0, r1, c6, c1, 4
    ldr     r1, =((28 << 1) | 1)          /* 2^29 */
    mcr     p15, 0, r1, c6, c1, 2
    /* region 3: 0xE0000000, 512 MB peripherals (UART, CRL, OCM, TCM globals), strongly ordered */
    mov     r0, #3
    mcr     p15, 0, r0, c6, c2, 0
    ldr     r1, =0xE0000000
    mcr     p15, 0, r1, c6, c1, 0
    ldr     r1, =0x1300                   /* XN, AP=011, strongly ordered */
    mcr     p15, 0, r1, c6, c1, 4
    ldr     r1, =((28 << 1) | 1)
    mcr     p15, 0, r1, c6, c1, 2
    /* region 4: 0xFFFC0000, 256 KB OCM, normal non-cacheable, executable (TCM stub lives at 0) */
    mov     r0, #4
    mcr     p15, 0, r0, c6, c2, 0
    ldr     r1, =0xFFFC0000
    mcr     p15, 0, r1, c6, c1, 0
    ldr     r1, =0x308                    /* AP=011, TEX=001 C=0 B=0: normal non-cacheable */
    mcr     p15, 0, r1, c6, c1, 4
    ldr     r1, =((17 << 1) | 1)          /* 2^18 */
    mcr     p15, 0, r1, c6, c1, 2
    /* enable MPU with background region, then caches */
    mrc     p15, 0, r0, c1, c0, 0
    orr     r0, r0, #0x20000              /* BR: background region for privileged accesses */
    orr     r0, r0, #0x1                  /* M */
    dsb
    mcr     p15, 0, r0, c1, c0, 0
    isb
    mrc     p15, 0, r0, c1, c0, 0
    orr     r0, r0, #0x4                  /* C */
    orr     r0, r0, #0x1000               /* I */
    dsb
    mcr     p15, 0, r0, c1, c0, 0
    isb
    /* clear .bss */
    ldr     r0, =__bss_start__
    ldr     r1, =__bss_end__
    mov     r2, #0
1:  cmp     r0, r1
    strlo   r2, [r0], #4
    blo     1b
    bl      __libc_init_array
    bl      main
    bl      r5_halt
2:  wfi
    b       2b

/* exception vector table copied to 0x0: ldr pc, [pc, #24] -> absolute handler addresses */
    .section .rodata.vectors, "a"
    .align 5
    .global r5_vectors
r5_vectors:
    ldr pc, [pc, #24]
    ldr pc, [pc, #24]
    ldr pc, [pc, #24]
    ldr pc, [pc, #24]
    ldr pc, [pc, #24]
    nop
    ldr pc, [pc, #24]
    ldr pc, [pc, #24]
    .word _start, r5_undef, r5_svc, r5_pabort, r5_dabort, 0, r5_irq, r5_fiq

    .text
    .arm
r5_undef:  mov r0, #1; mov r1, lr; b r5_fault
r5_svc:    mov r0, #2; mov r1, lr; b r5_fault
r5_pabort: mov r0, #3; mov r1, lr; b r5_fault
r5_dabort: mov r0, #4; mov r1, lr; b r5_fault
r5_irq:    mov r0, #6; mov r1, lr; b r5_fault
r5_fiq:    mov r0, #7; mov r1, lr; b r5_fault
