Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Documentation

Welcome to the documentation for **BioFoundation**. This directory provides in-depth information about the core components of the repository, including the architecture of our models, the datasets we use, and the training tasks we implement.

Good documentation is essential for reproducible and collaborative research. We have aimed to provide clear and comprehensive explanations to help you understand, use, and extend our work.

---

## Table of Contents

This documentation is organized into the following sections:

### 1. **[Datasets](./datasets/)**

This section provides detailed descriptions of the various EEG datasets used for pre-training and fine-tuning in this project. You will find information on:
-   The **TUH EEG Corpus (TUEG)**, used for our large-scale self-supervised pre-training.
-   The **TUH Abnormal EEG (TUAB)** dataset for normal vs. abnormal classification.
-   The **TUH Artifact (TUAR)** dataset for artifact detection tasks.
-   The **TUH Slowing (TUSL)** dataset for identifying slowing events in EEG signals.

Each document covers the dataset's purpose, size, and specific characteristics.

### 2. **[Model](./model/)**

This section contains a detailed breakdown of our primary model architecture, **FEMBA**. The documentation covers:
-   The **architecture overview**, including the tokenizer, encoder, decoder, and classifier heads.
-   The **bidirectional Mamba block** that forms the core of our encoder.
-   Details on **model variants** (e.g., FEMBA-tiny, FEMBA-base, FEMBA-large).
-   The self-supervised learning objective and classification protocols.

### 3. **[Tasks](./tasks/)**

This section describes the PyTorch Lightning `LightningModule` implementations that define our training pipelines. It explains:
-   The **`PretrainTask`**, which handles the self-supervised masked signal reconstruction.
-   The **`FinetuneTask`**, which manages the supervised training for downstream classification tasks.
-   Details on the data flow, loss computation, metrics, and optimization strategies for each task.

We encourage you to explore these documents to gain a deeper understanding of the components that make up the BioFoundation framework.
