# Debugging Guide

This document provides debugging strategies for ARES/ARES when running on GAP9.

---

## Error Rate-Based Diagnosis

Use the error magnitude from `make run platform=gvsoc` to narrow down problems:

### ~60% Error Rate
**Likely Cause:** Bias addition order bug or wrong golden file

**Check:**
- Verify kernels use `(acc + bias_int32) * scale`, not `(acc * scale) + bias`
- Check FC loaded correct golden reference files (checksums in output)

### 100% Error (All Zeros)
**Likely Cause:** Missing L3 prefetch or data type mismatch

**Check:**
- Look for `[ASYNC] Started prefetch of XXX weights (0 bytes)` → weights not loaded
- Verify FC cache flush before cluster execution
- Check if INT32 data interpreted as FP32 or vice versa

### Random/Garbage Values
**Likely Cause:** L1 buffer overrun or cache incoherency

**Check:**
- Verify tile sizes don't exceed L1 budget (look for "L1 buffer too small" messages)
- Check DMA descriptor batching (must stay under 16 concurrent transfers)
- Verify L3 data is cache-coherent (FC must flush before cluster reads)

### 1-10% Error (Small but Persistent)
**Likely Cause:** Quantization scale mismatch or missing requantization

**Check:**
- Verify scales in network_info.json match golden expectations
- Check for missing QuantIdentity layers between scale changes
- Review layer-by-layer golden comparisons to isolate failing layer

---

## Large Model Debugging (ResNet-18, Transformers)

### "0 bytes" Async Prefetch
```
CL:   [ASYNC] Started prefetch of layer3_0_conv2 weights (0 bytes)
```

**Root Cause:** Weight tensor classified as `L2_RESIDENT` instead of `L3_STAGED`

**Diagnosis:**
```bash
# Check residency in network_info.json
grep -A5 "layer3_0_conv2" tests/outputs/test_14_resnet18/golden_outputs/network_info.json

# Check actual weight size (if > 36KB, should be L3_STAGED)
ls -lh tests/outputs/test_14_resnet18/golden_outputs/weights/layer3.0.conv2_weight.npy
```

### L2 Addresses Outside Arena
```
CL: Allocating L2 Arena: 38416 bytes
  Arena: 1C03E280 - 1C047890
  layer3_0_conv2_weight: 1C0CDF90  ← Outside arena!
```

**Root Cause:** Stray `pi_l2_malloc()` calls bypassing memory planner

**Fix:** Remove malloc calls, use arena offsets only.

---

## Isolation Techniques

### Force L2-Only Execution
Temporarily hack `gap9_model.py` to return `None` for all tile configs:
- Error goes away → bug in DMA pipeline or tile management
- Error persists → bug in kernel logic or data preparation

### Force Synchronous DMA
Add `pi_cl_dma_wait()` immediately after every `pi_cl_dma_memcpy()`:
- Error goes away → race condition in async DMA pipeline
- Error persists → bug is not timing-related

### Probe L3 Data Integrity
```c
// In main.c, before pi_cluster_send_task_to_cl()
ram_read(&ram, fc_buffer, l3_addr, size);
// Print first few values to verify data loaded correctly
```

---

## Common Issues

### Quantization Errors
- Check layer-by-layer golden comparisons in GAP9 output
- Verify scales propagated correctly in network_info.json
- Ensure QuantIdentity layers after operations that change scale
- Max diff of 1 is acceptable (quantization tolerance), >1 indicates bug

### Extraction Failures
- Verify Brevitas model in eval mode
- Check all layers have quantization enabled
- Ensure sample input shape matches expected network input
- Review layer_order for missing/duplicate layers

### Stale Metadata (Multi-Input Operations)
If Add/Concat show incorrect input mappings, regenerate:
```bash
python tests/generate_all_tests.py --test test_N_name --skip-gvsoc
```
Symptoms: Non-zero errors on Add/Concat, or adding buffer to itself

### Code Generation Failures
- Verify network_info.json is complete
- Check weights directory contains all .npy files
- Ensure test_case directory exists
- Review execution plan for shape mismatches

### GAP9 Build Failures
- Source GAP SDK environment before make
- Check binary files present in bin/
- Verify Makefile has correct paths
- Review compiler warnings for type mismatches

### Runtime Errors
- Check for stack overflow (increase stack size)
- Verify L2 memory allocation sufficient
- Review buffer sizes in network_data.h
- Check for uninitialized variables or null pointers

---

## Performance Issues

### L2 Fallback (3x+ Slowdown)

**Check PI_CL_SLAVE_STACK_SIZE** in `codegen/templates/Makefile.mako`:
```makefile
PI_CL_SLAVE_STACK_SIZE ?= 0x400  # 1KB per core (optimal)
```
- `0x400` → ~110KB L1 available
- `0x800` → ~102KB L1 available (8KB less, may cause fallbacks)

**Check MHSA Softmax Mode:**
- Integer softmax: 105KB L1 requirement
- FP32 softmax: 134KB L1 requirement (+29KB)

**Symptoms of L2 Fallback:**
```
# BAD - L2 fallback:
[MHSA L1] FALLBACK - need 134144 bytes, have 102400
[MHSA INNER] QK=0, Softmax=0, AV=0, DMA=90000000

# GOOD - L1 tiled:
[MHSA L1] Using L1 tiled path (need 105344, have 110000)
[MHSA INNER] QK=8648583, Softmax=0, AV=0, DMA=88982
```

---

## Kernel Hang Issues

### pi_cl_team_barrier() Hang

**Cause:** Kernel using `pi_core_id()` called directly from Core 8 without `pi_cl_team_fork()`

**Example of broken code:**
```c
// Called from Core 8 entry point
network_conv1d_depthwise_int8(...);  // Uses pi_core_id() internally
// pi_core_id() returns 8, loop skips all work
// pi_cl_team_barrier() hangs - no team to synchronize!
```

**Fix:** Wrap with `pi_cl_team_fork()`:
```c
pi_cl_team_fork(NUM_CORES, conv1d_worker, &args);
```

### DMA Counter Exhaustion

**Symptom:** Hang after exactly 16 DMA calls

**Cause:** Queuing >16 DMAs without waiting (only 16 hardware counters)

**Fix:** Batch DMAs in groups of 16, wait after each batch.
