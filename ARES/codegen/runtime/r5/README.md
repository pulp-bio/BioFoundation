# ARES on the Zynq UltraScale+ RPU (Cortex-R5F), bare metal

This target runs ARES-generated C code on one Cortex-R5F core of a Zynq UltraScale+ MPSoC
(tested on the AMD Kria KV260 / K26 SOM). The R5F is the microcontroller-class core of the
SoC, so this gives a like-for-like comparison with GAP9-class deployments on the same chip
that also runs the Linux/POSIX target on the Cortex-A53.

The generated project is compiled unchanged; only the runtime shim differs:

| GAP9 / PMSIS concept | R5F implementation (`src/pmsis_r5.c`) |
|---|---|
| cluster cores | the calling R5F core (`NUM_CORES=1`, `pi_cl_team_fork` = direct call) |
| L1 / L2 / L3 memories | one arena in DDR (`0x20000000`, 256 MB), 64-byte aligned blocks |
| DMA | `memcpy` |
| read-only file system | a blob preloaded into DDR at `0x30000000` (`pack_fs.py`) |
| performance counter | R5F PMU cycle counter (`PMCCNTR`, /64 prescaler), reported in CPU cycles |
| stdout | UART1 (`0xFF010000`, the KV260 console) plus a copy in DDR at `0x3E000000` |

`src/crt0.S` sets up the stacks, installs exception vectors in ATCM (faults are reported on
the console with DFSR/DFAR), enables the VFP, programs the MPU (DDR normal write-back,
peripherals device / strongly ordered) and enables the I/D caches.

## Build

Requires `arm-none-eabi-gcc` with newlib (Ubuntu: `gcc-arm-none-eabi libnewlib-arm-none-eabi`).
From a generated project directory (`src/`, `inc/`, `bin/`):

```
make -f <ARES>/codegen/runtime/r5/Makefile.r5 EXTRA_CFLAGS=-DMINIMAL_OUTPUT
# -> ares_r5.bin (load at 0x10000000) and ares_fs.bin (load at 0x30000000)
```

`-DMINIMAL_OUTPUT` keeps per-layer debug prints off; at 115200 baud they dominate the run time.

## Run from U-Boot

```
ext4load mmc 1:2 0x10000000 /path/ares_r5.bin     # or: loady 0x10000000 (YMODEM over the console)
ext4load mmc 1:2 0x30000000 /path/ares_fs.bin
dcache flush
cpu 4 reset
cpu 4 release 10000000 split                       # U-Boot writes a TCM trampoline to 0x10000000
```

The program prints `ARES_OUT` logits, `Total cluster cycles` (CPU cycles of the network run) and
leaves a result record at `0x3FF00000` (`md.l 0x3ff00000 5`). Divide the cycles by the RPU clock
(`clk dump`, `cpu_r5`; 533.33 MHz on the KV260) to get the latency.

Note: newlib's `printf` has no `%a`; compare the `ARES_OUTX` lines (raw IEEE-754 bits of each logit), which every POSIX-family target prints.
