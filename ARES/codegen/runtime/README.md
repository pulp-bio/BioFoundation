Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.
# codegen/runtime

Shared C “runtime” sources compiled into generated projects.

## Why this exists

Historically, large shared implementations (kernels, DMA pipeline, L3 prefetch helpers, etc.) lived under `codegen/templates/` and were copied into every `tests/outputs/*/generated/` project. That made the repo heavy and hard to diff/review because the same code existed thousands of times.

The runtime keeps those shared sources **in one place**:
- Generated projects compile `codegen/runtime/src/*.c` directly (via their Makefile).
- Generated projects include headers from `codegen/runtime/inc/`.

## What stays generated

Per-network glue remains generated (e.g. `src/network.c`, `inc/network_data.h`, `inc/network.h`, and per-test binaries under `bin/`).

