Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# TUAR Dataset

The Temple University Hospital Artifact (TUAR) dataset is a subset of the TUH EEG Corpus designed for research on artifact detection in EEG recordings.

- Includes *310* annotated EEG files from *213* patients.
- Sourced from the official TUH repository.

Annotations cover five artifact types:
- **EYEM**: Eye movement–related spike-like waveforms.
- **CHEW**: Jaw muscle–induced artifacts.
- **SHIV**: Sustained sharp waveforms caused by shivering.
- **ELEC**: Electrode-related disturbances (e.g., pops, static).
- **MUSC**: High-frequency sharp waves from muscle activity.

Multiple labeling protocols are supported:
- **Binary Classification (BC)**: Artifact vs. No artifact.
- **Multilabel Classification (MC)**: Channel-wise, multi-artifact labels.
- **Multiclass–Multioutput Classification (MMC)**: Per-channel, multi-class artifact detection.
- **Multiclass Classification (MCC)**: Single-label artifact classification (subset of classes).

