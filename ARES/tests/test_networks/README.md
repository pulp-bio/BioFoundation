# Test Networks (`tests/test_networks/`)

These files define Brevitas/PyTorch models used by `tests/generate_all_tests.py` to generate end-to-end GAP9 projects under `tests/outputs/<test_name>/`.

## Status table

Legend:
- **PASS**: in the default regression set (`tests/run_gap9_projects.py:DEFAULT_TESTS`) and expected to pass.
- **SPECIAL**: known limitations, special flags, or not in the default regression set.

| Test | Status | Architecture | Purpose |
|------|--------|--------------|---------|
| test_1_simplecnn | PASS | CNN | Baseline reference (bit-exact) |
| test_2_tinycnn | PASS | CNN | Minimal CNN with 5x5 kernel |
| test_3_mlp | PASS | MLP | Dense-only network |
| test_4_resnet_basic | PASS | ResNet-ish | Add + GlobalAvgPool validation |
| test_5_densenet_basic | PASS | DenseNet-ish | Concat + AvgPool validation |
| test_6_multitilecnn | PASS | CNN | Forces Conv2D tiling |
| test_7_bottleneck | PASS | CNN | 1x1 bottleneck blocks |
| test_8_stride2 | PASS | CNN | Stride-2 conv edge cases |
| test_9_padding | PASS | CNN | Padding edge cases |
| test_10_resnet18 | PASS | ResNet-18 | Large-model validation (L3 staging) |
| test_11_layernorm_basic | PASS | MLP | LayerNorm validation |
| test_12_gelu_basic | PASS | MLP | GELU validation |
| test_13_transformer_simple | PASS | Transformer | 1-block transformer validation |
| test_14_multiblock_transformer | PASS | Transformer | Multi-block transformer validation |
| test_15_tinymyo_tiny | PASS | TinyMyo | Fast TinyMyo correctness |
| test_16_mamba_conv1d | PASS | Mamba | Depthwise Conv1D + SiLU validation |
| test_17_mamba_ssm | PASS | Mamba | SSM discretization + scan validation |
| test_18_mamba_block | PASS | Mamba | Full Mamba block validation |
| test_19_mamba_stacked | PASS | Mamba | Stacked Mamba blocks |
| test_20_bidirectional_mamba | PASS | Mamba | Bidirectional wrapper validation |
| test_21_femba_patchembedder | PASS | FEMBA | Patch-embedder + front-end validation |
| test_22_femba_full | PASS | FEMBA | Full FEMBA architecture test |
| test_23_femba_full_input | PASS | FEMBA | FEMBA with full input dimensions |
| test_24_femba_full_expand2 | PASS | FEMBA | FEMBA expand=2 (L3 streaming) |
| test_25_femba_tiny_int8 | PASS | FEMBA | FEMBA production INT8 |
| test_26_tinymyo_8ch_400tok | PASS | TinyMyo | 8-channel TinyMyo stress test |
| test_27_linear3d_bench | PASS | Microbench | Linear3D tiling benchmark |
| test_28_conv2d_remainder | PASS | Microbench | Conv2D im2col remainder benchmark |
| test_29_luna_base | PASS | LUNA | GroupNorm, RFFT, RoPE validation |
| test_30_autotune_stress | PASS | Microbench | Auto-tuner stress test |
| test_31_luna_full | PASS | LUNA | Full LUNA architecture |
| test_36_drowsiness_fusion | PASS | Fusion | Dual-input EEG+PPG detection |
| test_37_zeropad2d | PASS | Microbench | ZeroPad2d asymmetric padding |
| test_38_ne16_linear | SPECIAL | NE16 | NE16 accelerator validation |
| test_39_ne16_large | SPECIAL | NE16 | NE16 efficiency scaling |
| test_40_depthwise_conv | SPECIAL | CNN | MobileNet-style depthwise separable |
| test_41_large_depthwise | SPECIAL | CNN | Large depthwise for NE16 benchmarking |
| test_42_llama_minimal | SPECIAL | Llama | Minimal Llama block |
| test_43_cerebro_original | SPECIAL | Transformer | Full Cerebro EEG transformer |
| test_44_llama_swiglu | SPECIAL | Llama | Llama with SwiGLU FFN |

## Default Regression

33 tests (1-31, 36-37; no test_32) are in the default regression suite and pass. Tests 38-44 are development tests not included in the default set. See `tests/run_gap9_projects.py`.

## How to add a new test

1. Create `tests/test_networks/test_N_<name>.py` with a small Brevitas/PyTorch model.
2. Import the model class in `tests/generate_all_tests.py`.
3. Add an entry to `TestGenerator.NETWORKS` with `class`, `description`, and `epochs`.
4. Run:
   - `python tests/generate_all_tests.py --test test_N_<name> --skip-gvsoc`
   - `bash -lc "python tests/run_gap9_projects.py --tests test_N_<name>"` (requires GAP SDK)

## Naming convention

- File name: `test_<N>_<short_name>.py`
- Registry key (what scripts use): `test_<N>_<short_name>`
- Keep `<short_name>` descriptive (e.g., `mhsa_medium`, `mamba_block`, `conv2d_remainder`)
