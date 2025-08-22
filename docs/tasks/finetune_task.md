Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Finetune task

The FinetuneTask defines a flexible fine-tuning pipeline for classification tasks, supporting binary, multi-class, multi-label, and multi-output classification. It integrates modular model instantiation, configurable loss functions,  metric tracking, and optimization strategies such as layerwise learning rate decay.

1. Initialization
    Encoder (self.model): Instantiated dynamically using Hydra from the configuration (self.hparams.model). This is the backbone used to extract features from the input (e.g., waveform or spectrogram).
    Number of Classes and Classification Type: The number of output classes (self.num_classes) and the classification type (self.classification_type) are loaded from the model configuration. Supported classification types include:
    "mc": multi-label classification
    others: standard multi-class or binary classification
    Input Normalization: If enabled in the config (self.hparams.input_normalization.normalize), the module applies robust quartile normalization using the RobustQuartileNormalize function, based on specified lower and upper quantiles.
    Loss Function: The loss is selected based on the classification type:
    BCEWithLogitsLoss for "mc" (multi-label)
    CrossEntropyLoss for all other types (binary or multi-class)
    Classification Task Detection: A sanity check ensures self.num_classes is a valid integer ≥ 2. Based on its value, the task is flagged as:
    "binary" if self.num_classes == 2
    "multiclass" otherwise
    
2.  Data Flow: 
    The batch contains:
    X = batch['input']
    y = batch['label']

    Forward Pass:
    self.model(X): The encoder processes the input.

3. Loss:
    The script uses self.criterion(y_preds_logits, batch) to compute classification loss (i.e. Cross Entropy loss).

4. Metrics
    The classification pipeline employs a comprehensive set of evaluation metrics to monitor performance across training, validation, and test stages. These metrics are instantiated using TorchMetrics and are automatically adapted based on the specified classification type (bc, mc, ml, mcc, mmc).

    The following metrics are computed:

        Accuracy (macro-average)
        Recall (macro-average; configured with fixed task='multiclass')
        Precision (macro-average)
        F1 Score (macro-average)
        Cohen’s Kappa
        AUROC (Area Under the ROC Curve, macro-average)
        Average Precision (AUPR, macro-average)

    All metrics are cloned and tracked separately for each phase (train_, val_, and test_) to ensure stage-specific logging and monitoring.

    Predictions and probability distributions are derived based on the classification task:

        For binary, multi-class, and multi-label tasks (bc, mcc, ml), logits are passed through a softmax activation, and predicted classes are obtained via argmax.
        For multi-label classification (mc), logits are processed with a sigmoid activation, and predictions are computed by thresholding with round.
        For multi-class multi-output classification (mmc), logits are reshaped to match the output format, passed through sigmoid, and final predictions are obtained with argmax along the last dimension.
        This dynamic handling ensures that the predictions are in the correct form for both loss computation and metric evaluation.

5. Logging 
   Training step:
    Training loss is logged as train_loss (on both step and epoch). Metrics like accuracy, AUROC, AUPR, etc. are updated internally and then logged at the end of each epoch in on_train_epoch_end.

   Validation and Test step:
    on_validation_epoch_end logs metrics like val_acc, val_balanced_acc, val_auroc, val_aupr.
    on_test_epoch_end logs test metrics (test_acc, test_balanced_acc, test_auroc, test_aupr, test_precision, test_f1_score), enabling quick reference for final performance.

6. Layerwise Learning Rate Decay

    In configure_optimizers, the script configures a layerwise learning rate decay for the encoder if desired:
    num_blocks = self.hparams.model.num_blocks determines how many layers or blocks exist in the encoder.
    A base learning rate is multiplied by a decay factor (layerwise_lr_decay) for each subsequent layer.
    The classification head parameters always receive the base learning rate.
    This approach can help stabilize training by allowing deeper layers to receive a smaller LR, while keeping earlier layers more broadly tuned.

7. Optimizer and Scheduler

    Optimizer: Depending on self.hparams.optimizer.optim, it can be SGD, Adam, AdamW, or LAMB.
    Scheduler: The LR scheduler is instantiated via Hydra (hydra.utils.instantiate(self.hparams.scheduler, ...)) using the estimated number of training steps.
    LR Step: Because some custom schedulers from timm require manual stepping (scheduler.step_update), lr_scheduler_step is overridden to handle incremental updates (num_updates=self.global_step).

- Strip away any classification head weights from the checkpoint if head=False.
- Potentially freeze backbone layers if freeze_backbone=True.