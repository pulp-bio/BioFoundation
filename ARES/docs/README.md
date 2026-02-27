Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.
# Documentation Hub

Use this folder as the “map” for reading the repo. Start with the short entry-point docs, then go deeper as needed.

## Start Here

- Getting Started → `GETTING_STARTED.md`
- Architecture → `ARCHITECTURE.md`
- Adding Operations → `ADDING_OPERATIONS.md`
- Supported Operations → `SUPPORTED_OPERATIONS.md`

## Performance & Optimization

- `PERFORMANCE_BASELINE.md`: Cycles, MACs, and MACs/cycle metrics per test
- `../tools/perf/README.md`: Refactor gate tooling (collect, compare, markdown report)

## Code map (where things live now)

- **User-facing model building blocks:** `ares/nn/`
- **Shared runtime (single source of truth):** `codegen/runtime/`
  - Sources: `codegen/runtime/src/`
  - Headers: `codegen/runtime/inc/`
  - Central config defaults: `codegen/runtime/inc/ares_config.h`
- **Per-network codegen templates:** `codegen/templates/`
  - Main generator: `codegen/generate_c_code.py`
  - Main template (data-driven executor): `codegen/templates/network.c.mako`
- **Generated projects (current layout):**
  - Wrapper: `generated/src/network.c`
  - Network internals: `generated/src/net/*.c`
  - Op modules (only ops present): `generated/src/ops/*.c`
  - Internal header: `generated/inc/network_internal.h`
  - Shared runtime is compiled from `codegen/runtime/src/*.c` (not copied into each generated test).

## Artifact policy (tests/outputs)

- The curated committed reference output is `tests/outputs/test_1_simplecnn/`.
- For any other test, regenerate locally via `tests/generate_all_tests.py`.
