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

### 3. **Multiloader Data Module (`multiloader_data_module.py`)**

This module is designed for the self-supervised pre-training phase, where the model learns from a large, unlabeled dataset. Different from the pre-training data module, this data module supports datasets with different electrode configurations. 

**Key Features:**
-   **Multi-Dataset Handling**: It can take a list of different datasets and concatenate them into a single, large dataset. The datasets are processed sequentially so that a single batch contains data from a single dataset, which make sure that datasets with different number of electrodes can be processed together. It is important that the number of channels for each dataset is specified in the configuration.
-   **Automatic Splitting**: It automatically splits the combined dataset into training and validation sets based on a specified `train_val_split_ratio` (default is 80/20).
-   **Subset Loading**: Only a small subset of data can be loaded by specifying `subset_ratio`, which might be useful for debugging.

**Configuration Example (`config/data_module/pretrain_data_module.yaml`):**
```yaml
data_module:
  _target_: 'data_module.multiloader_data_module.VaryingChannelsDataModule'
  name: "eeg"
  cfg:
    num_workers: ${num_workers}
    batch_size: ${batch_size}
  train_val_split_ratio: 0.8
  datasets:
    Siena_dataset:
      _target_: 'datasets.siena_dataset.Siena_Dataset'
      hdf5_file: "#CHANGEME"
      num_channels: 29
    TUEG_20_channels_0:
      _target_: 'datasets.tueg_dataset.TUEG_Dataset'
      hdf5_file: "#CHANGEME"
      num_channels: 20
    TUEG_20_channels_1:
      _target_: 'datasets.tueg_dataset.TUEG_Dataset'
      hdf5_file: "#CHANGEME"
      num_channels: 20
```

### 4. **Subject-independent Data Module (`subject_independent_data_module.py`)**
This module is used for supervised fine-tuning with a custom train/val/test split based on the subjects. For each subject, different sessions are used for training, validation and testing to make sure that every subject has some data in all the splits.

**Key Features:**
-  **Distinct Dataloaders**: It manages seperate `Dataloader` instances for the training, validation, and test datasets.
-  **Split Parameter**: This data module can combine different datasets. For each dataset, the data module relies on user-provided train, validation and test size to extract the sessions to be used for every subject.
**Configuration Example (`config/data_module/finetune_data_module.yaml`):**
```yaml
data_module:
  _target_: 'data_module.subject_independent_data_module.SubjectIndependentDataModule'
  cfg:
    num_workers: ${num_workers}
    batch_size: ${batch_size}
  split_dir: '#CHANGEME'
  datasets:
    SEEDV_dataset:
      _target_: 'datasets.seed_v_dataset.CustomSEEDDataset'
      root_path: '#CHANGEME'
      num_workers: ${num_workers}
      num_channels: 62
      io_path: '#CHANGEME'
      train_size: 5
      val_size: 5
      test_size: 5
```