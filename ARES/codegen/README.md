Copyright (C) 2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.
# Code Generation Module

Generates production-ready C code for GAP9 RISC-V processor from verified INT8 neural networks with L1 tiling, async DMA pipelining, L3 memory staging, and cross-layer fusion.

---

## Overview

The code generation pipeline transforms a trained and quantized Brevitas model into optimized deployable C code for GAP9:

```
network_info.json + weights/ + test_cases/
            ↓
    generate_c_code.py (execution plan builder, fusion detection, tiling)
            ↓
generated/
├── inc/                        # Headers
│   ├── network.h               # Main network interface
│   ├── network_data.h          # Weight/bias metadata
│   ├── network_internal.h      # Internal shared declarations
│   └── mem.h                   # Memory management
├── src/                        # Implementation
│   ├── main.c                  # Entry point
│   ├── network.c               # Public API wrapper
│   ├── net/                    # Network internals
│   │   ├── network_layers.c
│   │   ├── network_cluster.c
│   │   ├── network_fc.c
│   │   └── network_globals.c
│   └── ops/                    # Op workers/helpers (only ops present)
│   └── mem.c                   # Memory allocators
├── bin/                        # Binary weight/golden files
│   ├── *_weight.bin
│   ├── *_bias.bin
│   ├── *_golden.bin
│   └── input.bin
└── Makefile                    # GAP9 build system
```

---

## Quick Start

```bash
# Generate C code for a test network
cd tests/outputs/test_1_simplecnn/
PY=${PY:-python}
$PY ../../../codegen/generate_c_code.py

# Build and run on GAP9 simulator
cd generated/
source ../../../tools/gap9_env_gvsoc.sh
make clean all platform=gvsoc
make run platform=gvsoc

# Check results
cat gvsoc_run.log
# Expected: "Test PASSED" with 0.0% error rate
```

---

## Architecture Highlights

### Key Features

**Memory Optimization:**
- **L1 Tiling:** Conv2D, Linear, MaxPool, AvgPool, GlobalAvgPool (128KB L1 budget)
- **L3 Staging:** Automatic L2 vs L3 residency (36KB threshold), async prefetch
- **Cross-Layer Fusion:** Conv→ReLU→Quant, Linear→ReLU→Quant (66% L2 traffic reduction)
- **Double Buffering:** Ping-pong L1 buffers for overlapped DMA

**Performance:**
- **Async DMA Pipeline:** PROLOGUE/STEADY/EPILOGUE stages (86-96% overlap for Conv2D)
- **Multi-Core Parallelism:** 8 cluster cores with barriers
- **DMA Batching:** Row-wise transfers batched to avoid hardware counter overflow

**Correctness:**
- **Fusion-Aware Golden Generation:** Python mirrors C code fusion decisions
- **Layer-by-Layer Validation:** Intermediate golden output comparison

---

## Input Format

### network_info.json

Network structure and metadata extracted by `tools/pytorch_extractor.py`:

```json
{
  "input_quant": {
    "type": "QuantIdentity",
    "scale": 0.0078125
  },
  "conv1": {
    "type": "QuantConv2d",
    "in_channels": 1,
    "out_channels": 16,
    "kernel_size": [3, 3],
    "stride": [1, 1],
    "padding": [1, 1],
    "scale_input": 0.0078125,
    "scale_weight": 0.003456,
    "scale_output": 0.015234
  },
  "relu1": {
    "type": "QuantReLU",
    "scale_input": 0.015234,
    "scale_output": 0.015234
  },
  "add": {
    "type": "QuantAdd",
    "input1": "relu2_out",
    "input2": "quant1_out",
    "scale1": 0.015,
    "scale2": 0.012,
    "scale_output": 0.018
  },
  "__layer_order__": ["input_quant", "conv1", "relu1", "add", ...]
}
```

**Key fields:**
- `type`: Layer operation (QuantConv2d, QuantLinear, QuantReLU, QuantAdd, etc.)
- `scale_*`: Quantization scales for inputs/weights/outputs
- `__layer_order__`: Execution sequence (topological order)

### weights/ directory

INT8 weights and FP32 biases as NumPy arrays:

```
golden_outputs/weights/
├── conv1_weight_int8.npy
├── conv1_bias_fp32.npy
├── conv2_weight_int8.npy
├── conv2_bias_fp32.npy
└── ...
```

### test_cases/ directory

Golden reference outputs for validation:

```
golden_outputs/test_cases/
├── summary.json
├── test_case_1/
│   ├── input_fp32.npy
│   ├── output_fp32.npy
│   ├── metadata.json
│   └── intermediate_int8/
│       ├── conv1_int8.npy
│       ├── relu1_int8.npy
│       └── ...
```

---

## Execution Plan Builder

`CCodeGenerator.build_execution_plan()` transforms layer_order into executable specs with fusion, tiling, and memory tier assignment.

### Process

1. **Shape Inference:** Calculate output dimensions for each layer
2. **Fusion Detection:** Identify Conv→ReLU→Quant, Linear→ReLU→Quant patterns
3. **Tile Calculation:** Determine L1 tile sizes (128KB budget)
4. **Memory Tier Assignment:** L2_RESIDENT vs L3_STAGED (36KB threshold)
5. **Buffer Allocation:** Determine activation buffers needed
6. **Scale Propagation:** Track quantization scales through network
7. **Parameter Indexing:** Map weights/biases to binary file offsets
8. **Golden Output Mapping:** Associate intermediate outputs for validation

### Example Spec (Conv2D with L1 Tiling)

```python
{
    'op': 'conv2d',
    'layer_name': 'conv1',
    'input_buffer': 'input_quant_out',
    'output_buffer': 'conv1_out',
    'in_h': 28, 'in_w': 28, 'in_ch': 1,
    'out_h': 28, 'out_w': 28, 'out_ch': 16,
    'kernel_h': 3, 'kernel_w': 3,
    'stride_h': 1, 'stride_w': 1,
    'pad_h': 1, 'pad_w': 1,
    'scale_input': 0.0078125,
    'scale_weight': 0.003456,
    'scale_output': 0.015234,
    'weight_entry': {...},
    'bias_entry': {...},
    'tile_config': {
        'tile_h': 14, 'tile_w': 14,        # L1 tile dimensions
        'num_tiles_h': 2, 'num_tiles_w': 2,  # 4 tiles total
        'halo_top': 1, 'halo_bottom': 1,     # Padding halo
        'halo_left': 1, 'halo_right': 1
    },
    'memory_tier': 'L2_RESIDENT',          # Weights always in L2
    'fused': ['relu1', 'quant1']           # Fusion metadata
}
```

---

## Fusion Detection

**Detected patterns:**

1. **Conv→ReLU→Quant** (3-way fusion)
   - Single kernel call: `conv2d_relu_quant_fused()`
   - Eliminates 2 intermediate buffers
   - 66% L2 traffic reduction

2. **Linear→ReLU→Quant** (3-way fusion)
   - Single kernel call: `linear_relu_quant_fused()`
   - Same benefits for fully connected layers

3. **GlobalAvgPool→Quant** (2-way fusion)
   - Single kernel call: `globalavgpool_quant_fused()`
   - Critical for accuracy (avoids double rounding)

4. **AvgPool→Quant** (2-way fusion)
   - Single kernel call: `avgpool_quant_fused()`
   - Same accuracy benefits

**Implementation:** `codegen/generate_c_code.py:850-1050`

```python
def _detect_fusion_opportunities(self, layer_specs):
    """Detects Conv/Linear→ReLU→Quant fusion patterns."""
    for i in range(len(layer_specs) - 2):
        if (layer_specs[i]['op'] in ['conv2d', 'linear_int8'] and
            layer_specs[i+1]['op'] == 'relu' and
            layer_specs[i+2]['op'] == 'requantize'):
            # Mark as fused, skip intermediate layers
            layer_specs[i]['fused'] = [layer_specs[i+1]['layer_name'],
                                       layer_specs[i+2]['layer_name']]
            layer_specs[i+1]['skip'] = True
            layer_specs[i+2]['skip'] = True
```

---

## L1 Tiling

### Tile Size Calculation

**Location:** `codegen/gap9_model.py`

**Conv2D tiling:**
```python
def calculate_conv2d_tile_size(in_h, in_w, in_ch, out_ch, k_h, k_w, stride_h, stride_w, pad_h, pad_w):
    """
    Calculates optimal L1 tile dimensions.

    Budget: 128KB L1
    Strategy: Iteratively reduce tile_h, tile_w until fits

    Returns:
        Conv2DTileConfig with tile_h, tile_w, halo sizes, byte budgets
    """
```

**Linear tiling:**
```python
def calculate_linear_tile_size(in_features, out_features):
    """
    Tiles over output features: tile_out = min(out_features, budget / in_features)

    L1 layout: [input_vector | output_tile | weight_tile]
    """
```

**Pooling tiling:**
- MaxPool, AvgPool, GlobalAvgPool: 2D spatial tiling (no halo for MaxPool)

### Tile Iteration (Async DMA Pipeline)

**Location:** `codegen/runtime/src/network_dma_pipeline.c`

**Three-stage pipeline:**

```c
// PROLOGUE: Load first tile
pi_cl_dma_cmd((uint32_t)l2_addr, (uint32_t)l1_ping, tile_bytes, PI_CL_DMA_DIR_EXT2LOC, &copy);
pi_cl_dma_wait(&copy);

// STEADY STATE: Overlap load(i+1) with compute(i)
for (int tile = 0; tile < num_tiles - 1; tile++) {
    // Start async load of next tile to alternate buffer
    pi_cl_dma_cmd((uint32_t)l2_next, (uint32_t)l1_next, tile_bytes,
                  PI_CL_DMA_DIR_EXT2LOC, &copy);

    // Compute current tile (Cores 0-7)
    pi_cl_team_fork(CL_NUM_CORES, worker_kernel, &args);

    // Wait for both DMA and compute
    pi_cl_dma_wait(&copy);
    pi_cl_team_barrier();

    // Swap buffers
    swap(l1_ping, l1_pong);
}

// EPILOGUE: Compute last tile
pi_cl_team_fork(CL_NUM_CORES, worker_kernel, &args);
```

---

## L3 Memory Staging

### Memory Tier Assignment

**Policy:** `codegen/generate_c_code.py:620-680`

```python
STAGING_THRESHOLD_BYTES = 36864  # 36KB

if weight_size_bytes > STAGING_THRESHOLD_BYTES:
    layer_spec['memory_tier'] = 'L3_STAGED'
else:
    layer_spec['memory_tier'] = 'L2_RESIDENT'
```

**L2_RESIDENT:**
- Weights always in L2
- Fast access for all layers
- Examples: Early conv layers, classifier

**L3_STAGED:**
- Weights in L3, loaded JIT to L2
- Async prefetch before layer execution
- Examples: ResNet-18 layer4.0.conv2 (147KB)

### Async Prefetch

**Location:** `codegen/runtime/src/network_l3_prefetch.c`

```c
void prefetch_layer_weights_async(int layer_id, pi_cl_dma_copy_t *copy) {
    // Initiate L3→L2 DMA transfer (FC DMA controller)
    pi_cl_dma_cmd((uint32_t)l3_addr, (uint32_t)l2_addr,
                  weight_bytes, PI_CL_DMA_DIR_EXT2LOC, copy);
    // Returns immediately, compute continues
}

// In network.c:
prefetch_layer_weights_async(layer_id+1, &prefetch_copy);  // Prefetch next layer
network_conv2d_tiled_l1_pipeline(...);                      // Compute current layer
pi_cl_dma_wait(&prefetch_copy);                             // Wait before next layer
```

**Benefits:**
- 53% L2 peak memory reduction
- Hides L3→L2 transfer latency
- Enables deep networks (ResNet-18: 77 layers)

---

## Mako Template System

Uses [Mako](https://www.makotemplates.org/) for dynamic C code generation.

### Key Templates

#### 1. network.c.mako - Main Orchestration

**FC setup** (lines ~100-300):
- L2 buffer allocation via `pi_l2_malloc()`
- L3→L2 weight loading for L3_STAGED layers
- L1 buffer allocation via `pi_l1_malloc()`
- Cluster initialization and task dispatch

**Cluster entry point** (lines ~400-900):
- Core 8 (orchestrator) execution
- Per-layer tiling and DMA orchestration
- Worker core dispatch via `pi_cl_team_fork()`
- Layer-wise golden output comparison

**Generated code example:**
```c
% for spec in layer_specs:
    % if spec.get('skip', False):
        <% continue %>  ## Skip fused layers
    % endif

    % if spec['op'] == 'conv2d':
        % if spec.get('fused'):
            // Conv2D with fused ReLU+Quant
            conv2d_relu_quant_fused(
                a->${spec['input_buffer']},
                a->${spec['output_buffer']},
                ${spec['in_h']}, ${spec['in_w']}, ${spec['in_ch']},
                /* ... more params ... */
            );
        % else:
            // Standard Conv2D
            % if spec.get('tile_config'):
                // L1 tiled execution
                conv2d_tiled_l1_pipeline(...);
            % else:
                // L2-only execution (fallback)
                network_conv2d_int8(...);
            % endif
        % endif
    % elif spec['op'] == 'linear_int8':
        // Similar branching for Linear
    % endif
% endfor
```

#### 2. network_dma_pipeline.c.mako - Async DMA (722 lines)

**Key functions:**
- `conv2d_tiled_l1_pipeline()` - Conv2D async DMA with double buffering
- `linear_tiled_l1_pipeline()` - Linear async DMA
- `maxpool_tiled_l1_pipeline()` - MaxPool async DMA
- `avgpool_tiled_l1_pipeline()` - AvgPool async DMA
- `globalavgpool_tiled_l1_pipeline()` - GlobalAvgPool with partial sums
- `requantize_int8_inplace_l1()` - Fused requantization

**DMA batching**
```c
#define MAX_DMA_ROWS 8
pi_cl_dma_copy_t row_copies[MAX_DMA_ROWS];

for (int row_start = 0; row_start < valid_h; row_start += MAX_DMA_ROWS) {
    int batch_size = min(MAX_DMA_ROWS, valid_h - row_start);

    // Queue batch
    for (int i = 0; i < batch_size; i++) {
        pi_cl_dma_memcpy(&row_copies[i]);
    }

    // Wait for batch
    for (int i = 0; i < batch_size; i++) {
        pi_cl_dma_wait(&row_copies[i]);
    }
}
```

#### 3. network_kernels.c - Worker Kernels

**Operations:**
- `network_conv2d_int8()` - Standard Conv2D
- `conv2d_relu_quant_fused()` - Conv2D with fused ReLU+Quant
- `network_linear_int8()` - Intermediate Linear (INT8→INT8)
- `network_linear_int8_to_fp32()` - Final Linear (INT8→FP32)
- `linear_relu_quant_fused()` - Linear with fused ReLU+Quant
- `network_maxpool_int8()` - MaxPool
- `network_avgpool_int8()` - AvgPool
- `avgpool_quant_fused()` - AvgPool with fused Quant
- `network_global_avgpool_int8()` - GlobalAvgPool
- `globalavgpool_quant_fused()` - GlobalAvgPool with fused Quant
- `network_add_int8()` - Element-wise addition (ResNet)
- `network_concat_int8()` - Channel concatenation (DenseNet)
- `relu_int8_inplace()` - ReLU activation

**Multi-core parallelism:**
```c
void relu_int8_worker(void *args) {
    int core_id = pi_core_id();  // 0-7
    int chunk_size = total_size / CL_NUM_CORES;
    int start = core_id * chunk_size;
    int end = (core_id == CL_NUM_CORES - 1) ? total_size : start + chunk_size;

    for (int i = start; i < end; i++) {
        data[i] = (data[i] < 0) ? 0 : data[i];
    }
}

// In main kernel:
pi_cl_team_fork(CL_NUM_CORES, relu_int8_worker, args);
pi_cl_team_barrier();
```

#### 4. network_l3_prefetch.c - L3 Staging

**Async L3→L2 DMA:**
```c
void prefetch_layer_weights_async(
    int layer_id,
    uint8_t *l2_dest,
    uint32_t size_bytes,
    pi_cl_dma_copy_t *copy
) {
    // FC DMA controller (separate from cluster DMA)
    uint32_t l3_addr = get_l3_weight_addr(layer_id);
    pi_cl_dma_cmd(l3_addr, (uint32_t)l2_dest, size_bytes,
                  PI_CL_DMA_DIR_EXT2LOC, copy);
}
```

#### 5. network_data.h.mako - Binary File Descriptors

```c
static const BinaryFileDescriptor binary_files[] = {
    {"bin/conv1_weight.bin", 432, 0x1A2B3C4D},
    {"bin/conv1_bias.bin", 64, 0x5E6F7A8B},
    {"bin/conv1_golden.bin", 6272, 0x9C8D7E6F},
    // ...
};
```
---

## Adding New Operations

To add a new operation to the code generator:

### 1. Add Atomic Operation

Implement in `atomic_ops/new_op.py` with unit test.

### 2. Define Layer Spec in build_execution_plan()

`codegen/generate_c_code.py`:

```python
elif layer_type == 'NewOperation':
    # Extract parameters
    param1 = layer_data.get('param1', default)

    # Calculate output shape
    out_shape = compute_output_shape(current_shape, param1)

    # Allocate output buffer
    output_entry = register_buffer(...)

    # Create layer spec
    spec.update({
        'op': 'new_op',
        'param1': param1,
        'scale_input': current_scale,
        'scale_output': layer_data.get('scale_output')
    })
```

### 3. Add Template Case in network.c.mako

```mako
% elif spec['op'] == 'new_op':
    network_new_op_int8(
        a->${spec['input_buffer']},
        a->${spec['output_buffer']},
        ${spec['param1']},
        ${spec['scale_input']}f,
        ${spec['scale_output']}f
    );
```

### 4. Implement Kernel in `codegen/runtime/src/network_kernels.c`

```c
void network_new_op_int8(
    const int8_t *input,
    int8_t *output,
    int param1,
    float scale_input,
    float scale_output
) {
    // INT8 implementation
    // Use INT32 accumulation for multiply-accumulate
}
```

### 5. Test with New Network

```bash
PY=${PY:-python}
$PY tests/generate_all_tests.py --test test_N_name
```

---

## Configuration

**Key parameters:**

```python
CCodeGenerator(
    network_info_path="golden_outputs/network_info.json",
    weights_dir="golden_outputs/weights",
    test_case_dir="golden_outputs/test_cases/test_case_3",
    output_dir="generated"
)
```

**Tiling budget:** `L1_BUDGET_BYTES = 128 * 1024` (128KB)
**Staging threshold:** `STAGING_THRESHOLD_BYTES = 36 * 1024` (36KB)
**DMA batch size:** `MAX_DMA_ROWS = 8` (hardware counter limit)
**Stack size:** `8192` bytes per cluster core (used for current suite, including ResNet-18 and transformer/EMG workloads)

---

## Phase Checkpoints (Debug/Replay)

Phase checkpoint export is disabled by default and can be enabled for
reproducible codegen-state snapshots:

```bash
export ARES_CHECKPOINT_DIR="tests/outputs/<test>/checkpoints"
export ARES_CHECKPOINT_TAG="debug_snapshot"   # optional
```

When enabled, codegen emits:

- `pre_fusion.json`
- `post_fusion.json`
- `post_tiling.json`
- `post_memory_plan.json`

Replay helpers are under `codegen/checkpoints/` (`README.md`, `replay.py`).

---

## Troubleshooting

### Code Generation Fails

**Problem:** `KeyError: 'weight_int8'`
**Solution:** Verify network_info.json has all required fields, re-run pytorch_extractor.py

**Problem:** `FileNotFoundError: weights/conv1_weight_int8.npy`
**Solution:** Run `tools/pytorch_extractor.py` to regenerate weights

**Problem:** `ValueError: Shape mismatch`
**Solution:** Check layer_order, verify shapes propagate correctly

### Generated Code Won't Compile

**Problem:** Undeclared identifier
**Solution:** Ensure all buffers registered in `activation_buffers`

**Problem:** Type mismatch
**Solution:** Check pointer types (int8_t*, float*, etc.)
---

---

## References

- [Main README](../README.md) - Repository overview
- [Atomic Operations](../atomic_ops/README.md) - INT8 operation references
- [Tools](../tools/README.md) - Extraction and validation
- [Testing](../tests/README.md) - Test network validation
- [Mako Templates](https://www.makotemplates.org/)
