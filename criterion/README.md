Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Criterion

This directory contains the loss functions (criteria) used for training the models in this repository. The configuration for these criteria is managed via Hydra, allowing you to easily specify which loss to use for a given experiment.

---

## Pre-training Criterion

For self-supervised pre-training, we use a custom criterion defined in `pretrain_criterion.py`. This loss function is designed for masked signal reconstruction tasks, where the model is trained to predict the values of masked-out portions of an input signal.

### Key Features

* **Masked Loss Calculation**: The criterion calculates the loss separately for the masked and unmasked regions of the signal. In our pre-training task (`pretrain_task.py`), we only use the `masked_loss` to train the model, forcing it to learn meaningful representations from the visible context.
* **Configurable Loss Types**: You can choose from three different reconstruction loss functions via the `loss_type` parameter in the configuration file (`config/criterion/pretrain_criterion.yaml`).

    * `'l1'`: **Mean Absolute Error (L1 Loss)**. This measures the average absolute difference between the predicted and original values.
    * `'l2'`: **Mean Squared Error (L2 Loss)**. This measures the average of the squares of the errors.
    * `'smooth_l1'`: **Smooth L1 Loss (Huber Loss)**. This is a combination of L1 and L2 loss. It behaves like L2 loss for small errors and L1 loss for large errors, making it less sensitive to outliers than L2 loss. This is the default for our pre-training experiments.

### Configuration Example
File: `config/criterion/pretrain_criterion.yaml`
```yaml
_target_: criterion.pretrain_criterion.PretrainCriterion
loss_type: 'smooth_l1' # Can be 'l1', 'l2', or 'smooth_l1'
```

## Fine-tuning Criterion
For supervised fine-tuning and classification tasks, we use the standard PyTorch loss functions.

### Key Features
* **Cross-Entropy Loss**: For binary and multi-class classification tasks, `torch.nn.CrossEntropyLoss` is used by default. This is a standard choice for classification problems and is configured in `config/criterion/finetune_criterion.yaml`.
* **BCE With Logits Loss:** For multi-label classification tasks (e.g., when `classification_type` is `"mc"`), `torch.nn.BCEWithLogitsLoss` is used to handle multiple labels per sample. This is also configured in `config/criterion/finetune_criterion.yaml`.

### Configuration Example
File: `config/criterion/finetune_criterion.yaml`
```yaml
# @package _global_

criterion:
  _target_: torch.nn.CrossEntropyLoss
```
This setup provides a clear and flexible way to manage the loss functions for different training paradigms within the BioFoundation framework.
