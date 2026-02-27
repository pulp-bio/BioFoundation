Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.
# Codegen Checkpoints and Replay

This module provides deterministic JSON snapshots of codegen state at key
stage boundaries to isolate regressions without rerunning the full pipeline.

## Files

- `pre_fusion.json`
- `post_fusion.json`
- `post_tiling.json`
- `post_memory_plan.json`

## Enabling Checkpoint Export

Checkpoint export is disabled by default. Enable it by setting:

```bash
export ARES_CHECKPOINT_DIR="tests/outputs/<test>/checkpoints"
```

Optional metadata tag:

```bash
export ARES_CHECKPOINT_TAG="debug_snapshot"
```

Then run codegen normally. If enabled, `generate_c_code.py` writes stage
snapshots under `ARES_CHECKPOINT_DIR`.

## Replay Workflow

Replay utility restores generator state from a checkpoint payload:

```python
from codegen.checkpoints import replay_from_checkpoint_file

generator = CCodeGenerator(...)
replay_from_checkpoint_file(generator, "tests/outputs/<test>/checkpoints/post_memory_plan.json")
```

This is useful for reproducing template/runtime emission issues from a known
intermediate state.

## Checkpoint Payload Schema

Each file has:

- `schema_version` (currently `1.0`)
- `stage`
- `created_utc`
- `state`
- `metadata`
