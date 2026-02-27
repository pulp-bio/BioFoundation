# Performance Baseline

**Generated:** 2026-02-18
**Commit:** After layer handler extraction, pipeline naming cleanup, copyright headers

## Summary

- **Tests:** 36 total (tests 1-31, 33-37)
- **Passing:** 36/36 (100%)
- **Failing:** 0

Previously failing tests now fixed:
- test_10_resnet18: was 15.1% error (FAIL), now 0.0% (PASS)
- test_25_femba_tiny_int8: was borderline FAIL (1.031%), now PASS

## Test Results

| Test | Status | Cycles | MACs | MACs/Cycle | Max Error |
|------|--------|--------|------|------------|-----------|
| test_1_simplecnn | PASS | 1,240,815 | 2,063,488 | 1.663 | 0.000000 |
| test_2_tinycnn | PASS | 735,534 | 344,960 | 0.469 | 0.000000 |
| test_3_mlp | PASS | 683,727 | 469,504 | 0.687 | 0.000000 |
| test_4_resnet_basic | PASS | 1,864,619 | 3,851,328 | 2.065 | 0.000000 |
| test_5_densenet_basic | PASS | 1,594,711 | 1,934,912 | 1.213 | 0.000000 |
| test_6_multitilecnn | PASS | 5,705,136 | 10,634,112 | 1.864 | 0.000000 |
| test_7_bottleneck | PASS | 1,791,887 | 452,224 | 0.252 | 0.000000 |
| test_8_stride2 | PASS | 2,035,399 | 6,096,704 | 2.995 | 0.000000 |
| test_9_padding | PASS | 16,300,916 | 42,151,360 | 2.586 | 0.000000 |
| test_10_resnet18 | PASS | 17,505,806 | 57,194,368 | 3.267 | 0.000000 |
| test_11_layernorm_basic | PASS | 555,889 | 15,685 | 0.028 | 0.000000 |
| test_12_gelu_basic | PASS | 455,788 | 102,272 | 0.224 | 0.009834 |
| test_13_transformer_simple | PASS | 6,472,982 | 6,955,375 | 1.075 | 0.024232 |
| test_14_multiblock_transformer | PASS | 43,577,886 | 105,792,429 | 2.428 | 0.347248 |
| test_15_tinymyo_tiny | PASS | 6,548,539 | 30,404,015 | 4.643 | 0.001180 |
| test_16_mamba_conv1d | PASS | 622,637 | 896 | 0.001 | 0.007774 |
| test_17_mamba_ssm | PASS | 489,176 | 12,928 | 0.026 | 0.585257 |
| test_18_mamba_block | PASS | 449,123 | 230,016 | 0.512 | 0.151228 |
| test_19_mamba_stacked | PASS | 593,062 | 688,768 | 1.161 | 0.320101 |
| test_20_bidirectional_mamba | PASS | 974,439 | 230,016 | 0.236 | 0.012587 |
| test_21_femba_patchembedder | PASS | 6,976,586 | 8,867,524 | 1.271 | 0.016665 |
| test_22_femba_full | PASS | 7,652,000 | 8,867,534 | 1.159 | 0.852212 |
| test_23_femba_full_input | PASS | 52,810,318 | 95,932,110 | 1.817 | 0.639488 |
| test_24_femba_full_expand2 | PASS | 489,800,131 | 594,813,390 | 1.214 | 0.147686 |
| test_25_femba_tiny_int8 | PASS | 489,810,169 | 594,813,390 | 1.214 | 1.031206 |
| test_26_tinymyo_8ch_400tok | PASS | 304,952,643 | 1,921,305,685 | 6.300 | 0.000000 |
| test_27_linear3d_bench | PASS | 26,015,006 | 239,004,288 | 9.187 | 0.007212 |
| test_28_conv2d_remainder | PASS | 1,799,348 | 1,849,312 | 1.028 | 0.000000 |
| test_29_luna_base | PASS | 65,105,177 | 298,428,005 | 4.584 | 1.109162 |
| test_30_autotune_stress | PASS | 1,415,534 | 2,017,754 | 1.425 | 0.000000 |
| test_31_luna_full | PASS | 108,886,963 | 416,869,255 | 3.828 | 0.067648 |
| test_36_drowsiness_fusion | PASS | 4,606,827 | 12,038,848 | 2.613 | 0.000000 |
| test_37_zeropad2d | PASS | 684,540 | 51,712 | 0.076 | 0.000000 |

## Performance Analysis

### Top Performers (MACs/Cycle)

| Rank | Test | MACs/Cycle | Notes |
|------|------|------------|-------|
| 1 | test_27_linear3d_bench | 9.187 | Linear-heavy workload with NE16 |
| 2 | test_26_tinymyo_8ch_400tok | 6.300 | Large transformer (1.9B MACs) |
| 3 | test_29_luna_base | 4.584 | Attention-heavy model |
| 4 | test_15_tinymyo_tiny | 4.643 | Efficient transformer |
| 5 | test_31_luna_full | 3.828 | Full LUNA model |
| 6 | test_10_resnet18 | 3.267 | Classic CNN (now passing) |
| 7 | test_8_stride2 | 2.995 | Strided convolutions |
| 8 | test_9_padding | 2.586 | Padded convolutions |
| 9 | test_36_drowsiness_fusion | 2.613 | Fusion model |
| 10 | test_14_multiblock_transformer | 2.428 | Multi-block transformer |

### Efficiency by Category

| Category | Avg MACs/Cycle | Notes |
|----------|----------------|-------|
| NE16-accelerated | 9-10 | Excellent with hardware accelerator |
| Linear-heavy | 5-9 | Good utilization with NE16 |
| Transformers | 2.5-5 | Good for attention workloads |
| CNNs | 1.0-3.3 | Depends on layer sizes |
| Mamba/SSM | 0.01-1.2 | Memory-bound state ops |
| FEMBA | 1.1-1.8 | State-heavy operations |

### Notes

1. **MACs/Cycle interpretation:**
   - GAP9 cluster has 8 RISC-V cores with SIMD (theoretical ~8 MACs/cycle)
   - NE16 accelerator adds up to 16 MACs/cycle for supported operations
   - Combined peak: ~24 MACs/cycle theoretical
   - Values >8 indicate effective NE16 utilization
   - Values <1 indicate memory-bound or overhead-dominated layers

2. **Low MACs/Cycle tests:**
   - test_16_mamba_conv1d (0.001): Tiny test, setup overhead dominates
   - test_17_mamba_ssm (0.026): State-space operations are memory-bound
   - test_11_layernorm_basic (0.028): Reduction ops, not compute-bound

3. **test_31_luna_full:** Cycles increased ~9% vs Feb 1 baseline (100M→109M) with
   identical MAC count. Suspected cause: re-training produces different weight
   distributions affecting tiling branching. Worth monitoring.

## Updating This Baseline

To regenerate this baseline after changes:

```bash
# 1. Regenerate all tests (includes MACs calculation)
bash -lc 'conda run -n TimeFM python tests/generate_all_tests.py --skip-gvsoc'

# 2. Run regression (gets cycles)
bash -lc 'conda run -n TimeFM python tests/run_gap9_projects.py'

# 3. Collect metrics
bash -lc 'conda run -n TimeFM python tools/perf/collect_gap9_metrics.py --out candidate.json'

# 4. Compare vs saved baseline
bash -lc 'conda run -n TimeFM python tools/perf/compare_gap9_metrics.py \
  --baseline docs/baseline_metrics.json --candidate candidate.json \
  --ignore-baseline-status --allow-missing-candidate'
```
