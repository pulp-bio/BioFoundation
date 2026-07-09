Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Models

This directory contains the implementations of the deep learning models used in the **BioFoundation** project. Each model is defined as a PyTorch `nn.Module` and is designed to be configurable and extensible for various research tasks.

## Available Models

- **FEMBA**: A lightweight EEG model designed for both pretraining and fine-tuning tasks. For a more detailed description of the model check the [documentation](../docs/model/FEMBA.md).
- **LUNA**: An efficient EEG model specifically designed for handling different types of electrode configurations. For a more detailed description of the model check the [documentation](../docs/model/LUNA.md).
- **TinyMyo**: A 3.6M-parameter Transformer-based foundation model for surface EMG (sEMG). It is pretrained on >480 GB of EMG data and optimized for ultra-low-power, real-time deployment, including microcontrollers (GAP9) where it achieves an inference time of 0.785 s, energy of 44.91 mJ and power envelope of 57.18 mW. For a more detailed description of the model check the [documentation](../docs/model/TinyMyo.md).
- **PanLUNA**: A 5.4M-parameter pan-modal biosignal foundation model that extends LUNA’s channel-unification mechanism from topology invariance to cross-modal fusion, jointly processing EEG, ECG, and PPG within a single shared encoder via sensor-type embeddings, with no modality-specific backbones or paired multimodal data required during pretraining; it is pretrained on ~40,000 hours of heterogeneous biosignal data using masked signal reconstruction and achieves strong performance across unimodal and multimodal evaluation settings. For more detailed description of the model check [documentation]../docs/model/PanLUNA.md).
