Copyright (C) 2025-2026 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Documentation

Welcome to the BioFoundation documentation. This directory covers the model families, biosignal datasets, and PyTorch Lightning training tasks used by the repository.

Start with the root [`README.md`](../README.md) to choose a model and launch an experiment. Use the pages here when adapting an architecture, dataset, or task.

---

## Table of Contents

This documentation is organized into the following sections:

### 1. [Models](./model/)

Architecture and usage notes are available for every published model family:

- [FEMBA](./model/FEMBA.md): bidirectional Mamba for EEG.
- [LUNA](./model/LUNA.md): query-unified, topology-agnostic EEG.
- [TinyMyo](./model/TinyMyo.md): compact foundation model for sEMG.
- [LuMamba](./model/LuMamba.md): query-unified Mamba for EEG.
- [PanLUNA](./model/PanLUNA.md): sensor-aware modeling across EEG, ECG, and PPG.

Each model also has a pretrained release linked from the root model zoo. The canonical machine-readable index is [`biofoundation/model_registry.py`](../biofoundation/model_registry.py).

### 2. [Datasets](./datasets/)

The dataset pages describe preparation and evaluation protocols for the TUH EEG corpora used in pre-training and downstream tasks, including TUEG, TUAB, TUAR, and TUSL. Additional preprocessing entry points for EEG, EMG, ECG, and PPG datasets live in [`make_datasets`](../make_datasets/).

### 3. [Tasks](./tasks/)

The task pages describe the Lightning modules for self-supervised pre-training, classification, and regression. Task steps normalize incoming data through [`SignalBatch`](../biofoundation/core/batch.py), which provides a common input contract while preserving model-specific channel and sensor metadata.

### 4. [Training Guide](./TRAINING.md)

The training guide covers environment variables, Hydra experiment selection, the shared batch contract, distributed training, checkpoints, and fast validation.

### 5. Project References

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) defines extension and pull request expectations.
- [`CITATIONS.md`](./CITATIONS.md) contains BibTeX for all five model families.
- [`config/README.md`](../config/README.md) explains Hydra composition and overrides.
- [`make_datasets/README.md`](../make_datasets/README.md) documents preprocessing and HDF5 conversion.

ARES deployment documentation remains self-contained under [`ARES/docs`](../ARES/docs/) because it uses an independent environment and workflow.
