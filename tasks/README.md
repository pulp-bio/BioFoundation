Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Tasks (LightningModules)

This directory contains the core logic for our training pipelines, implemented as PyTorch Lightning `LightningModule` classes. In the context of this repository, a "Task" defines everything related to the training, validation, and testing of a model for a specific objective (e.g., pre-training or fine-tuning).

---

## What is a Task?

A `LightningModule` (which we call a Task) is a powerful abstraction that organizes PyTorch code and removes boilerplate. It defines:
-   The model to be trained.
-   The forward pass logic (`training_step`, `validation_step`, `test_step`).
-   The loss function (criterion).
-   The optimizers and learning rate schedulers to be used.
-   How metrics are calculated and logged.

This keeps our model definitions (`models/`) separate from the training logic, making the code cleaner and more modular.

---

## Implementations

We provide two main Task implementations for the different stages of our research.

### 1. **Pre-training Task (`pretrain_task.py`)**

`MaskTask` (the class name in the file) defines the self-supervised pre-training pipeline based on **masked signal modeling**. The goal is to train a model to reconstruct missing (masked) parts of an EEG signal, forcing it to learn meaningful representations of the underlying data patterns.

**Key Features:**
-   **Dynamic Masking**: On each training step, it generates a random mask to hide a certain percentage (`masking_ratio`) of the input signal patches.
-   **Reconstruction Loss**: It computes the reconstruction loss (e.g., Smooth L1 loss) only on the masked regions, ensuring the model learns to generate contextually relevant information.
-   **Input Normalization**: Supports optional robust quartile normalization of the input signals.
-   **Visualization**: During validation, it logs plots of the original vs. reconstructed signals to TensorBoard, providing a qualitative way to assess model performance.

### 2. **Fine-tuning Task (`finetune_task.py`)**

`FinetuneTask` defines the supervised training pipeline for downstream classification tasks. It takes a pre-trained model and adapts it to a specific labeled dataset (e.g., TUAB for abnormal/normal classification).

**Key Features:**
-   **Flexible Classification**: Supports multiple classification modes, including binary (`bc`), multi-class (`mcc`), multi-label (`mc`), and multi-class multi-output (`mmc`).
-   **Comprehensive Metrics**: Uses `torchmetrics` to automatically track and log a wide range of classification metrics (Accuracy, Precision, Recall, F1-Score, AUROC, AUPR, etc.) for the train, validation, and test sets.
-   **Layer-wise Learning Rate Decay**: Implements an optional strategy to apply different learning rates to different layers of the model, which can improve fine-tuning stability.
-   **Pre-trained Checkpoint Loading**: Includes logic to load weights from a pre-trained model checkpoint and freeze/unfreeze layers as needed for fine-tuning.