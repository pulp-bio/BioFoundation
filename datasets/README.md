Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Datasets

This directory contains the PyTorch `Dataset` classes for the project. In PyTorch, a `Dataset` class is responsible for storing and providing access to the samples in your data (e.g., EEG signals and their corresponding labels). This directory defines how individual data points are loaded and processed.

---

## The HDF5Loader

To handle the large-scale EEG datasets used in our experiments, we use a custom `HDF5Loader` class defined in `hdf5_dataset.py`. Storing data in the HDF5 format is highly efficient for I/O operations, which is crucial when working with datasets that can be several gigabytes in size.

**Key Features:**
-   **Efficient Loading**: Reads data directly from `.h5` files, which is significantly faster than loading thousands of individual files.
-   **Memory Caching**: Includes an optional caching mechanism to store recently accessed samples in memory, further speeding up data loading during training.
-   **Flexible Modes**: Supports both `finetune` mode (loading both signals and labels) and pre-training mode (loading signals only).
-   **Grouped Data**: Assumes data within the HDF5 file is organized into groups, which helps manage large datasets.

For details on how to convert raw EEG data into the required HDF5 format, please see the documentation in the [`../make_datasets`](../make_datasets) directory.

---

## Supported Datasets

Our framework is primarily designed to work with several of the Temple University Hospital (TUH) EEG corpora. For more detailed information on each, please refer to the documentation in the [`../docs/datasets`](../docs/datasets) directory.

### 1. **TUH Abnormal EEG (TUAB) Dataset**
-   **Purpose**: Used for classifying EEG sessions as either "normal" or "abnormal".
-   **Size**: Contains over 3,000 EEG sessions.
-   **Task**: Binary classification.

### 2. **TUH Artifact (TUAR) Dataset**
-   **Purpose**: Designed for detecting various types of artifacts in EEG signals.
-   **Annotations**: Covers five common artifact types, including eye movement, chewing, and electrode noise.
-   **Tasks**: Supports multiple classification protocols such as binary, multi-label, and multi-class classification.

### 3. **TUH Slowing (TUSL) Dataset**
-   **Purpose**: Curated for classifying different types of slowing events in EEG signals, which can be indicative of neurological disorders.
-   **Annotations**: Includes four classes: slowing, seizure, complex background, and normal.
-   **Task**: Multi-class classification.

### 4. **TUH EEG (TUEG) Dataset**
-   **Purpose**: This is our primary dataset for self-supervised pre-training.
-   **Size**: One of the largest publicly available EEG corpora, containing over 21,000 hours of recordings from more than 14,000 patients.
-   **Usage**: We use this large, diverse dataset to train our foundation models to learn robust representations of EEG signals before fine-tuning them on specific downstream tasks.