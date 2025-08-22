Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Data Modules

This directory contains the PyTorch Lightning `LightningDataModule` classes for the project. A `LightningDataModule` is a shareable, reusable class that encapsulates all the steps needed to process data for a given task. It handles the downloading, splitting, and loading of data, making our experiments clean and reproducible.

---

## What is a Data Module?

A DataModule is responsible for:
-   **Data Splitting**: Creating the `train`, `validation`, and `test` datasets.
-   **Creating DataLoaders**: Providing the `train_dataloader`, `val_dataloader`, and `test_dataloader` that feed batches of data to the model.

This abstraction keeps all data-related code organized in one place, separate from the model and training logic.

---

## Implementations

We provide two primary `DataModule` implementations to handle the different phases of our research:

### 1. **Pre-training Data Module (`pretrain_data_module.py`)**

This module is designed for the self-supervised pre-training phase, where the model learns from a large, unlabeled dataset.

**Key Features:**
-   **Multi-Dataset Handling**: It can take a list of different datasets (e.g., from various `HDF5Loader` instances) and concatenate them into a single, large dataset.
-   **Automatic Splitting**: It automatically splits the combined dataset into training and validation sets based on a specified `train_val_split_ratio` (default is 80/20).
-   **Configurability**: The datasets to be loaded are specified in the Hydra configuration, making it easy to add or remove data sources for pre-training experiments.

**Configuration Example (`config/data_module/pretrain_data_module.yaml`):**
```yaml
data_module:
  _target_: 'data_module.pretrain_data_module.PretrainDataModule'
  train_val_split_ratio: 0.8
  datasets:
    TUEG_data_part1:
      _target_: 'datasets.hdf5_dataset.HDF5Loader'
      hdf5_file: "/path/to/tueg_part1.h5"
      finetune: False
    TUEG_data_part2:
      _target_: 'datasets.hdf5_dataset.HDF5Loader'
      hdf5_file: "/path/to/tueg_part2.h5"
      finetune: False
```
### 2. **Fine-tuning Data Module (`finetune_data_module.py`)**
This module is used for supervised fine-tuning on downstream tasks, such as classification. It assumes you have pre-defined `train`, `validation`, and `test` splits.
**Key Features:**
-  **Distinct Dataloaders**: It manages seperate `Dataloader` instances for the training, validation, and test datasets.
-  **Clear Separation**: Unlike the pre-training module, it does not perform any splitting, relying on the user to provide distinct datasets for each phase of the supervised learning pipeline.
**Configuration Example (`config/data_module/finetune_data_module.yaml`):**
```yaml
data_module:
  _target_: 'data_module.finetune_data_module.FinetuneDataModule'
  train:
    _target_: 'datasets.hdf5_dataset.HDF5Loader'
    hdf5_file: '${env:DATA_PATH}/TUAB_data/train.h5'
  val:
    _target_: 'datasets.hdf5_dataset.HDF5Loader'
    hdf5_file: '${env:DATA_PATH}/TUAB_data/val.h5'
  test:
    _target_: 'datasets.hdf5_dataset.HDF5Loader'
    hdf5_file: '${env:DATA_PATH}/TUAB_data/test.h5'
```