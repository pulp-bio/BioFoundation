Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Models

This directory contains the implementations of the deep learning models used in the **BioFoundation** project. Each model is defined as a PyTorch `nn.Module` and is designed to be configurable and extensible for various research tasks.

## Available Models

- **FEMBA**: A lightweight EEG model designed for both pretraining and fine-tuning tasks. For a more detailed description of the model check the [documentation](../docs/model/FEMBA.md).
- **LUNA**: An efficient EEG model specifically designed for handling different types of electrode configurations. For a more detailed description of the model check the [documentation](../docs/model/LUNA.md).
- **TinyMyo**: A 3.6M-parameter Transformer-based foundation model for surface EMG (sEMG). It is pretrained on >480 GB of EMG data and optimized for ultra-low-power, real-time deployment, including microcontrollers (GAP9) where it achieves an inference time of 0.785 s, energy of 44.91 mJ and power envelope of 57.18 mW. For a more detailed description of the model check the [documentation](../docs/model/TinyMyo.md).
