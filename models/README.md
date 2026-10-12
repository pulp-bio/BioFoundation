Copyright (C) 2025-2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Models

This directory contains the PyTorch `nn.Module` implementations for the BioFoundation model families. Hydra model settings live in [`../config/model`](../config/model/), while the canonical family metadata and batch requirements live in [`../biofoundation/model_registry.py`](../biofoundation/model_registry.py).

Families come in two shapes. The five original families bundle their output layer into the model and select it from `num_classes` at construction time. S-CEReBrO separates the two: [`s_cerebro.py`](s_cerebro.py) is an encoder that emits token embeddings, and [`model_heads`](model_heads/) holds the prediction heads that consume them, configured through [`../config/model_head`](../config/model_head/). Both shapes are supported; the [protocols](../biofoundation/core/protocols.py) describe the second.

## Available Models

| Model | Signals | Summary | Resources |
| --- | --- | --- | --- |
| FEMBA | EEG | Efficient bidirectional Mamba model for long EEG sequences. | [Documentation](../docs/model/FEMBA.md) / [Hugging Face](https://huggingface.co/PulpBio/FEMBA) |
| LUNA | EEG | Query-based channel unification for topology-agnostic EEG modeling. | [Documentation](../docs/model/LUNA.md) / [Hugging Face](https://huggingface.co/PulpBio/LUNA) |
| TinyMyo | sEMG | Compact rotary Transformer designed for flexible EMG processing and edge deployment. | [Documentation](../docs/model/TinyMyo.md) / [Hugging Face](https://huggingface.co/PulpBio/TinyMyo) |
| LuMamba | EEG | LUNA-style channel unification with efficient Mamba temporal modeling. | [Documentation](../docs/model/LuMamba.md) / [Hugging Face](https://huggingface.co/PulpBio/LuMamba) |
| PanLUNA | EEG, ECG, PPG | Sensor-aware query unification for unimodal and multimodal biosignal learning. | [Documentation](../docs/model/PanLUNA.md) / [Hugging Face](https://huggingface.co/PulpBio/PanLUNA) |
| S-CEReBrO | EEG | Windowed alternating attention over per-channel patches, with a separate prediction head. | [Documentation](../docs/model/SCEReBrO.md) / [Hugging Face](https://huggingface.co/PulpBio/S-CEReBrO) |

Use the matching pre-training or fine-tuning experiment in [`../config/experiment`](../config/experiment/) rather than instantiating a model in isolation when starting a reproducible run.
