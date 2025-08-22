Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Learning Rate Schedulers

This directory contains custom learning rate (LR) schedulers used during model training. A learning rate scheduler dynamically adjusts the learning rate during training, which is a critical technique for achieving optimal convergence and model performance.

---

## Implementations

We provide custom wrappers and extensions for common LR scheduling strategies to add features like linear warmup, which is particularly effective for stabilizing training in the early stages.

### 1. **Cosine Annealing with Warmup (`cosine.py`)**

This directory includes `CosineLRSchedulerWrapper`, a wrapper around the `CosineLRScheduler` from the popular `timm` library.

**Key Features:**
-   **Cosine Annealing**: The learning rate follows a cosine curve, starting at the base LR and gradually decreasing to a minimum value (`min_lr`). This smooth decay often leads to better model performance.
-   **Linear Warmup**: For the first few epochs (`warmup_epochs`), the learning rate linearly increases from a small initial value (`warmup_lr_init`) to the base learning rate. This helps prevent large, unstable gradient updates at the beginning of training when the model weights are still random.
-   **Step-based Scheduling**: The scheduler operates on a per-step (i.e., per-batch) basis, allowing for fine-grained control over the learning rate trajectory.

**Configuration Example (`config/scheduler/cosine.yaml`):**
```yaml
scheduler:
  _target_: 'schedulers.cosine.CosineLRSchedulerWrapper'
  trainer: ${trainer}
  warmup_epochs: 5
  min_lr: 1e-6
  warmup_lr_init: 1e-6
```
### 2. **Multi-Step Decay with Warmup** (`multi_step_lr.py`)
This directory also provides `MultiStepLRWarmup`, which extends PyTorch’s standard `MultiStepLR` scheduler with an optional warmup phase.
**Key Features:**
-   **Step-wise Decay**: The learning rate is held constant until it reaches a specified milestone (an epoch or step number), at which point it is multiplied by a decay factor (`gamma`). This is a simple yet effective scheduling strategy.
-  **Linear Warmup**: Similar to the cosine scheduler, it supports a linear warmup phase at the beginning of training, controlled by the `warmup_iter` and `warmup_init_lr`.
**Configuration Example (`config/scheduler/multi_step.yaml`):**
```yaml
scheduler:
  _target_: 'schedulers.multi_step_lr.multi_step_lr'
  milestones: "200000+400000" # Decay LR at these steps
  gamma: 0.5                  # Decay factor
  warmup_iter: -1             # Number of warmup steps (-1 to disable)
  warmup_init_lr: 0
```
These schedulers are configured via Hydra and are automatically instantiated and managed by the PyTorch Lightning trainer within our `Task` modules.