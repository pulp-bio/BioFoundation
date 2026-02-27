Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.
# ARES Optimization Framework

This module provides automated optimization for ARES neural network code generation through a knowledge base and auto-tuning system.

## Automatic Workflow (Default)

When running `tests/generate_all_tests.py`, the default workflow is:

1. **Code Generation**: Generate C code with heuristic tiling
2. **GVSOC Validation**: Run on simulator to verify correctness
3. **Auto-Tuning**: Identify bottleneck layers (>500K cycles) and tune
4. **KB Recording**: Store best configs in knowledge_base.json
5. **Regeneration**: Regenerate code with KB configs
6. **Verification**: Re-run GVSOC to verify improvement

To disable auto-tuning (for CI speed):
```bash
python tests/generate_all_tests.py --skip-auto-tune
```

## Manual Tuning

```bash
# Dry-run to see what would be tuned
python tools/auto_tune.py --test test_41_luna_base --dry-run

# Tune all bottleneck layers
python tools/auto_tune.py --test test_41_luna_base --tune-all

# Tune specific layers
python tools/auto_tune.py --test test_41_luna_base --layers freq_fc1 cross_attn_unify

# Tune with more iterations (slower but more thorough)
python tools/auto_tune.py --test test_41_luna_base --tune-all --max-iter 50

# Debug mode for full tracebacks
python tools/auto_tune.py --test test_41_luna_base --tune-all --debug
```

## Knowledge Base CLI

```bash
# List all entries
python tools/knowledge_base_cli.py list

# List by op type
python tools/knowledge_base_cli.py list --op-type linear_int8

# Search for configs matching a shape
python tools/knowledge_base_cli.py search --op-type linear_int8 --shape '{"M": 400, "N": 768}'

# Show statistics
python tools/knowledge_base_cli.py stats

# Show negative results (failed configs to avoid)
python tools/knowledge_base_cli.py negatives
```

## How It Works

### Knowledge Base Auto-Application

When code is generated, the KB is queried **before** tiling decisions are made:

```
Layer processing:
1. _prepare_kb_config(layer_name, op_type, shape)  <- Query KB FIRST
   |
   +-> If KB entry found: populate layer_config_overrides
   |
2. _determine_*_memory_tier(...)                    <- Use KB config if available
   |
   +-> Check _get_layer_override() for hints
   |
3. Generate layer spec with optimized tiling
```

### Auto-Tuner Workflow

```
tune_and_regenerate():
1. Profile baseline with GVSOC
2. Identify bottleneck layers (cycles > threshold)
3. For each bottleneck:
   a. Generate candidate configurations
   b. Run GVSOC for each config
   c. Measure cycles, MACs/cycle
   d. Record best to KB
   e. Record failures as negative results
4. Regenerate code with new KB entries
5. Verify improvement with GVSOC
```