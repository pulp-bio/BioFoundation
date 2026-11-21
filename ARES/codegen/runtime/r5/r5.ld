/* SPDX-License-Identifier: Apache-2.0 */
/* ARES on Cortex-R5F (RPU0), everything in DDR. U-Boot loads the image at 0x10000000. */
ENTRY(_start)
MEMORY {
  DDR (rwx) : ORIGIN = 0x10000000, LENGTH = 0x08000000   /* 128 MB: code, data, bss, heap, stack */
}
SECTIONS {
  .text : { KEEP(*(.text.start)) *(.text .text.*) *(.glue_7 .glue_7t .vfp11_veneer .v4_bx) } > DDR
  .init : { KEEP(*(.init)) } > DDR
  .fini : { KEEP(*(.fini)) } > DDR
  .rodata : { *(.rodata .rodata.*) } > DDR
  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > DDR
  .ARM.exidx : { __exidx_start = .; *(.ARM.exidx* .gnu.linkonce.armexidx.*); __exidx_end = .; } > DDR
  .preinit_array : { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)); PROVIDE_HIDDEN(__preinit_array_end = .); } > DDR
  .init_array : { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(SORT(.init_array.*))); KEEP(*(.init_array*)); PROVIDE_HIDDEN(__init_array_end = .); } > DDR
  .fini_array : { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(SORT(.fini_array.*))); KEEP(*(.fini_array*)); PROVIDE_HIDDEN(__fini_array_end = .); } > DDR
  .data : { *(.data .data.*) } > DDR
  .bss (NOLOAD) : { . = ALIGN(4); __bss_start__ = .; *(.bss .bss.*) *(COMMON); . = ALIGN(4); __bss_end__ = .; } > DDR
  . = ALIGN(64);
  end = .; PROVIDE(_end = .);                 /* newlib heap (sbrk) starts here */
  __heap_limit = 0x17000000;
  __stack_top = 0x18000000;                   /* 16 MB stack below 0x18000000 */
}
