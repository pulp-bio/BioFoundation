Copyright (C) 2025-2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Models

This directory contains the PyTorch `nn.Module` implementations for the BioFoundation model families. Hydra model settings live in [`../config/model`](../config/model/), while the canonical family metadata and batch requirements live in [`../biofoundation/model_registry.py`](../biofoundation/model_registry.py).

## Available Models

| Model | Signals | Summary | Resources |
| --- | --- | --- | --- |
| FEMBA | EEG | Efficient bidirectional Mamba model for long EEG sequences. | [Documentation](../docs/model/FEMBA.md) / [Hugging Face](https://huggingface.co/PulpBio/FEMBA) |
| LUNA | EEG | Query-based channel unification for topology-agnostic EEG modeling. | [Documentation](../docs/model/LUNA.md) / [Hugging Face](https://huggingface.co/PulpBio/LUNA) |
| TinyMyo | sEMG | Compact rotary Transformer designed for flexible EMG processing and edge deployment. | [Documentation](../docs/model/TinyMyo.md) / [Hugging Face](https://huggingface.co/PulpBio/TinyMyo) |
| LuMamba | EEG | LUNA-style channel unification with efficient Mamba temporal modeling. | [Documentation](../docs/model/LuMamba.md) / [Hugging Face](https://huggingface.co/PulpBio/LuMamba) |
| PanLUNA | EEG, ECG, PPG | Sensor-aware query unification for unimodal and multimodal biosignal learning. | [Documentation](../docs/model/PanLUNA.md) / [Hugging Face](https://huggingface.co/PulpBio/PanLUNA) |

Use the matching pre-training or fine-tuning experiment in [`../config/experiment`](../config/experiment/) rather than instantiating a model in isolation when starting a reproducible run.
