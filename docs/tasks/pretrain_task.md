Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Pretrain task

The PretrainTask defines a masked signal modeling pretraining pipeline for time-series data, specifically EEG signals. It employs a self-supervised learning strategy inspired by masked modeling, where portions of the input signal are masked and the model is trained to reconstruct the missing parts. This encourages the encoder to learn meaningful representations of the input signal.

1. Initialization

    Encoder (self.model): The backbone network is instantiated dynamically using Hydra from the configuration (self.hparams.model). It extracts high-level features from masked input signals.
    Loss Function (self.criterion): The reconstruction loss is instantiated via Hydra and supports the following options:
    L1Loss
    MSELoss (L2)
    SmoothL1Loss
    Input Normalization: If enabled in the configuration (self.hparams.input_normalization.normalize), the module applies robust normalization using RobustQuartileNormalize, based on specified lower and upper quantiles.
    Masking Configuration: The masking strategy is configured via self.hparams.masking, which defines:
    patch_size: the rectangular patch dimensions (H, W) used for masking.
    masking_ratio: the proportion of total patches to be masked during pretraining.
2. Data Flow

    Each training or validation batch contains:

    X = batch['input']: Raw input signal (e.g., EEG), shaped (B, C, T).
    The data flow is as follows:

    A binary mask is generated per input using generate_mask.
    The masked version of the input is computed and passed to the encoder: masked_X → self.model.
    The encoder reconstructs the original signal based only on the unmasked regions.
3. Masking Strategy

    The masking logic follows a patch-based scheme:

    The input signal is divided into non-overlapping rectangular patches based on the specified patch_size.
    A subset of these patches is randomly selected to be masked, guided by the configured masking_ratio.
    Masking is applied across the temporal and/or spatial dimensions, depending on the patch shape.
    Only the masked portions of the signal are used for loss computation — the model is not penalized for reconstructing parts it could already see.

4. Loss Computation

    The model returns:

    reconstructed_signal: the full prediction over the entire input shape.
    Intermediate embeddings (ignored in loss).
    Loss is computed as:

    loss = self.criterion(reconstructed_signal[mask], original_signal[mask])
    Only the masked regions (mask == 1) contribute to the loss, promoting genuine contextual learning.

5. Logging

    Training:
    The loss is logged as train_loss at both step and epoch levels.
    Validation:
    val_loss is logged at the end of each epoch.
    A fixed set of reconstructed samples are visualized in TensorBoard.
    Plots overlay the original and reconstructed signals.
    Masked areas are visually highlighted for qualitative inspection.
6. Optimizer and Scheduler

    Optimizer: Configured via self.hparams.optimizer.optim, supporting:
    SGD, Adam, AdamW, or LAMB.
    Learning Rate Scheduling:
    Instantiated using Hydra (hydra.utils.instantiate(self.hparams.scheduler, ...)).
    The scheduler supports step-based updates with frequency = 1.
    Handles scheduler.step_update when needed (e.g., for timm schedulers).
    Optional Differential Learning Rate:
    If differential_lr is enabled, the encoder and reconstruction head can receive different learning rates.
