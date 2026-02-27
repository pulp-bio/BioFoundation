# Supported Operations

This document is the reference for what the current ARES extraction flow can map from PyTorch/Brevitas into generated C runtime execution.

## Scope

- Source of truth for extraction support: `tools/pytorch_extractor.py`
- Source of truth for custom blocks: `tests/test_networks/brevitas_custom_layers.py`
- Source of truth for C op enums: `codegen/runtime/inc/layer_descriptors.h`

This document describes what is currently supported without changing extraction/codegen behavior.

## High-Level Compatibility Rules

1. Use Brevitas quantized primitives (`QuantConv2d`, `QuantLinear`, `QuantReLU`, `QuantIdentity`) for core INT8 paths.
2. For residuals and multi-input flow, use explicit modules (`QuantAdd`, `QuantConcatenate`, `QuantMean`), not implicit graph math.
3. Keep models in `eval()` mode before extraction.
4. Composite blocks (attention/Mamba/PatchEmbed/etc.) are extracted as whole blocks; their internals are skipped by the extractor.
5. Prefer class names and construction patterns already used in `tests/test_networks/`.

## Operation Table

| Category | Extracted layer type | Python class / detector | Runtime OpType mapping | Notes |
|---|---|---|---|---|
| Quantization | `QuantIdentity` | `brevitas.nn.QuantIdentity` | `OP_REQUANTIZE` | Often fused away in neighboring ops. |
| Convolution | `QuantConv2d` | `brevitas.nn.QuantConv2d` | `OP_CONV2D` | Use dilation=1 for guaranteed compatibility. |
| Linear | `QuantLinear` | `brevitas.nn.QuantLinear` | `OP_LINEAR_INT8` / `OP_LINEAR_FP32` | Output path depends on layer placement/config. |
| Activation | `QuantReLU` | `brevitas.nn.QuantReLU` | `OP_RELU` | Standard INT8 activation path. |
| Activation | `GELU` | `torch.nn.GELU` | `OP_GELU` | Typically surrounded by QuantIdentity in test models. |
| Activation | `SiLU` | `QuantSiLU` custom layer | `OP_SILU` | Class-name detector support included. |
| Pooling | `MaxPool2d` | `torch.nn.MaxPool2d` | `OP_MAXPOOL` | INT8 maxpool path. |
| Pooling | `AvgPool2d` | `torch.nn.AvgPool2d` | `OP_AVGPOOL` | INT8 avgpool with scale propagation. |
| Pooling | `GlobalAvgPool` | `torch.nn.AdaptiveAvgPool2d` | `OP_GLOBAL_AVGPOOL` | Used for CNN heads. |
| Pooling | `AdaptiveAvgPool1d` | `torch.nn.AdaptiveAvgPool1d` | `SIMPLE_ADAPTIVE_AVGPOOL1D / OP_AVGPOOL path` | Used in sequence models. |
| Normalization | `LayerNorm` | `torch.nn.LayerNorm` | `OP_LAYERNORM` | FP32 stats + quantized I/O path. |
| Normalization | `RMSNorm` | `torch.nn.RMSNorm` or class name `RMSNorm` | `OP_RMSNORM` | Runtime enum exists behind `ARES_LLAMA_SUPPORT`. |
| Normalization | `GroupNorm` | `torch.nn.GroupNorm` | `OP_GROUPNORM` | Used by LUNA paths. |
| Element-wise | `Add` | `QuantAdd` custom layer | `OP_ADD` | Preferred residual implementation. |
| Element-wise | `Concatenate` | `QuantConcatenate` custom layer | `OP_CONCAT` | Dense/cross-branch merge. |
| Element-wise | `Mean` | `QuantMean` custom layer | `OP_MEAN` | Composite custom op; children skipped. |
| Shape/Layout | `Flatten` | `torch.nn.Flatten` | `OP_FLATTEN` | No arithmetic change. |
| Shape/Layout | `Squeeze` | class name `Squeeze` | `OP_SQUEEZE` | Class-name detector support. |
| Shape/Layout | `Reshape` | class name `Reshape` | `OP_RESHAPE` | Class-name detector support. |
| Shape/Layout | `Permute` | class name `Permute` | `OP_TRANSPOSE_2D` | Class-name detector support. |
| Shape/Layout | `ZeroPad2d` | `torch.nn.ZeroPad2d` | `OP_ZEROPAD2D` | Explicit padding op. |
| Embedding | `Embedding` | `torch.nn.Embedding` | `OP_EMBEDDING` | Weights quantized and scales extracted. |
| Signal | `RFFT` | class name in `RFFT`, `RFFTFeatures`, `RFFTFeature` | `OP_RFFT` | Class-name detector support. |
| Attention | `MultiheadSelfAttention` | `QuantSelfAttention`, `QuantMultiHeadAttention`, `QuantRoPESelfAttention`, `QuantMHSA` | `OP_MHSA` | Composite op; children skipped. |
| Attention | `CrossAttention` | `QuantCrossAttention` | `OP_CROSS_ATTENTION` | Composite op; children skipped. |
| Attention | `AlternatingAttention` | `QuantAlternatingAttention` | `OP_ALTERNATING_ATTENTION` | Composite op; children skipped. |
| Mamba/SSM | `Conv1dDepthwise` | `QuantConv1dDepthwise` | `OP_CONV1D_DEPTHWISE` | Custom depthwise sequence op. |
| Mamba/SSM | `SSM` | `QuantSSM` | `OP_SSM` | Composite op; children skipped. |
| Mamba/SSM | `MambaBlock` | `QuantMambaBlock` | `OP_MAMBA_BLOCK` | Composite block. |
| Mamba/SSM | `MambaWrapper` | `QuantMambaWrapper` | `OP_MAMBA_WRAPPER` | Composite bidirectional wrapper. |
| Mamba/SSM | `PatchEmbed` | `QuantPatchEmbed` | `OP_PATCH_EMBED` | Composite embedding block. |

## Quantization Requirements

The current tested path assumes:

1. INT8 quantized weights for quantized linear/conv modules.
2. Activation scales extractable from quant modules (for example `module.quant`, output quantizers, or known wrappers).
3. Bias and accumulator handling consistent with extractor/runtime expectations (INT32 accumulator domain for INT8 MAC paths).
4. A consistent sequence of quantized modules so scale propagation remains explicit and stable.

Recommended practical pattern:

- Add `QuantIdentity` at key boundaries (input, after pooling/layout changes, before sensitive residual joins).
- Keep residual/add flows explicit via `QuantAdd`.
- Use custom ARES-compatible blocks for attention and Mamba/SSM families.

## Model Structure Requirements

1. Extraction walks `model.named_modules()` in order and emits supported layers.
2. Submodules of composite blocks are skipped intentionally once the parent composite is recognized.
3. Unsupported standalone modules are not converted automatically; use the migration table below.
4. Keep unsupported training-time-only layers (for example `Dropout`) out of inference graphs where possible.

## Migration Guide

| Existing pattern | ARES-compatible migration |
|---|---|
| `BatchNorm2d`, `BatchNorm1d` | Replace with `LayerNorm`/`GroupNorm` or fold into previous affine layer. |
| `Dropout` | Remove for inference graphs; no-op in eval mode. |
| `nn.MultiheadAttention` | Use custom `QuantMultiHeadAttention` implementation (ARES-compatible extraction points). |
| Implicit residual `x = x + y` | Prefer explicit `QuantAdd` module. |
| Implicit concat `torch.cat([...])` | Prefer explicit `QuantConcatenate` module. |
| `ConvTranspose*` | Restructure network with supported ops (upsample + conv style decomposition). |
| Dilated convolution | Use dilation=1 path where possible. |

## Custom Layer Imports

Current tested custom layers live in:

- `tests/test_networks/brevitas_custom_layers.py`

The additive `ares.nn` package provides user-facing wrappers/re-exports so model code can avoid depending on test module paths directly.

## Verification Workflow

Run the compatibility checker before full test/codegen flow:

```bash
python tools/check_model_compatibility.py --model-file my_model.py --model-class MyModel
```

For built-in test networks:

```bash
python tools/check_model_compatibility.py --test-network test_13_transformer_simple --model-class SimpleTransformer
```

The checker returns exit code `0` for compatible models and `1` for incompatible ones.
