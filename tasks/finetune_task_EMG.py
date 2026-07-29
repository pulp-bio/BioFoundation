# *----------------------------------------------------------------------------*
# * Copyright (C) 2025 ETH Zurich, Switzerland                                 *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Fasulo                                                     *
# *----------------------------------------------------------------------------*
from typing import Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from safetensors.torch import load_file
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    CohenKappa,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from biofoundation.core.batch import as_signal_batch
from util.train_utils import MinMaxNormalization


class FinetuneTask(pl.LightningModule):
    """PyTorch LightningModule for TinyMyo EMG fine-tuning.

    The classification path supports single-label tasks only: ``"bc"`` for
    two classes and ``"mcc"`` for three or more mutually exclusive classes.
    It trains with cross-entropy loss, configurable label smoothing, and
    epoch-level classification metrics. The regression path uses L1 loss and
    regression metrics.

    TinyMyo returns raw logits directly. This task supplies an all-false mask
    to keep the model interface shared with masked pretraining without masking
    any fine-tuning input samples.

    Attributes:
        model (nn.Module): The instantiated neural network.
        num_classes (int): Number of target classes or regression outputs.
        classification_type (str): ``"bc"`` or ``"mcc"`` for classification.
        task (str): The specific task type ('classification' or 'regression').
        normalize (bool): Whether input normalization is enabled.
        criterion (nn.Module): Cross-entropy or L1 loss.
    """

    def __init__(self, hparams: DictConfig):
        """Initializes the FinetuneTask with Hydra configurations.

        Sets up the model, loss functions, and metric collections based on the
        provided task type.

        Args:
            hparams (DictConfig): Configuration with model, optimizer,
                scheduler, finetuning, and classification settings.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.num_classes = self.hparams.model.num_classes
        self.task = self.hparams.model.task

        # Enable normalization if specified in parameters
        self.normalize = False
        if "input_normalization" in self.hparams and self.hparams.input_normalization.normalize:
            self.normalize = True
            self.normalize_fct = MinMaxNormalization()

        if self.task == "regression":
            self.criterion = nn.L1Loss()

            # Metric
            mean_metrics = MetricCollection({
                "rmse": MeanSquaredError(squared=False),
                "mae": MeanAbsoluteError(),
            })

            self.train_mean_metrics = mean_metrics.clone(prefix="train/")
            self.val_mean_metrics = mean_metrics.clone(prefix="val/")
            self.test_mean_metrics = mean_metrics.clone(prefix="test/")

        else:
            # Loss function
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.hparams.label_smoothing
            )
            self.classification_type = self.hparams.classification_type

            # Classification mode detection
            if not isinstance(self.num_classes, int):
                raise TypeError("Number of classes must be an integer.")
            elif self.num_classes < 2:
                raise ValueError("Number of classes must be at least 2.")
            elif self.num_classes == 2:
                self.classification_task = "binary"
            else:
                self.classification_task = "multiclass"

            expected_classification_type = (
                "bc" if self.classification_task == "binary" else "mcc"
            )
            if self.classification_type != expected_classification_type:
                raise ValueError(
                    "TinyMyo supports single-label binary ('bc') or multiclass "
                    f"('mcc') classification; got {self.classification_type!r} "
                    f"for num_classes={self.num_classes}."
                )

            # Metrics
            label_metrics = MetricCollection(
                {
                    "micro_acc": Accuracy(
                        task=self.classification_task,
                        num_classes=self.num_classes,
                        average="micro",
                    ),
                    "macro_acc": Accuracy(
                        task=self.classification_task,
                        num_classes=self.num_classes,
                        average="macro",
                    ),
                    "recall": Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
                    "precision": Precision(
                        task=self.classification_task,
                        num_classes=self.num_classes,
                        average="macro",
                    ),
                    "f1": F1Score(
                        task=self.classification_task,
                        num_classes=self.num_classes,
                        average="macro",
                    ),
                    "cohen_kappa": CohenKappa(task=self.classification_task, num_classes=self.num_classes),
                }
            )
            logit_metrics = MetricCollection(
                {
                    "auroc": AUROC(
                        task=self.classification_task,
                        num_classes=self.num_classes,
                        average="macro",
                    ),
                    "average_precision": AveragePrecision(
                        task=self.classification_task,
                        num_classes=self.num_classes,
                        average="macro",
                    ),
                }
            )
            self.train_label_metrics = label_metrics.clone(prefix="train/")
            self.val_label_metrics = label_metrics.clone(prefix="val/")
            self.test_label_metrics = label_metrics.clone(prefix="test/")
            self.train_logit_metrics = logit_metrics.clone(prefix="train/")
            self.val_logit_metrics = logit_metrics.clone(prefix="val/")
            self.test_logit_metrics = logit_metrics.clone(prefix="test/")

        # Freeze unused parameters during fine-tuning so DDP doesn't complain
        # when find_unused_parameters=False
        for name, param in self.model.named_parameters():
            if "mask_token" in name:
                param.requires_grad = False

    def load_pretrained_checkpoint(self, model_ckpt: str) -> None:
        """Loads a pretrained PyTorch Lightning checkpoint (.ckpt).

        This method loads the state dict, optionally freezes layers based on configuration,
        and ensures the model head remains trainable for fine-tuning.

        Args:
            model_ckpt (str): Path to the .ckpt file.
        """
        assert self.model.model_head is not None
        print("Loading pretrained checkpoint from .ckpt file")
        checkpoint = torch.load(model_ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys when loading checkpoint: {missing_keys}")
        print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        for name, param in self.model.named_parameters():
            if self.hparams.finetuning.freeze_layers:
                param.requires_grad = False
            if "model_head" in name:
                param.requires_grad = True  # Unfreeze model head
            if "mask_token" in name:
                param.requires_grad = False

        print("Pretrained model ready.")

    def load_safetensors_checkpoint(self, model_ckpt: str) -> None:
        """Loads a pretrained model checkpoint in safetensors format.

        Args:
            model_ckpt (str): Path to the .safetensors file.
        """
        assert self.model.model_head is not None
        print("Loading pretrained safetensors checkpoint")
        state_dict = load_file(model_ckpt)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys when loading checkpoint: {missing_keys}")
        print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")

        for name, param in self.model.named_parameters():
            if self.hparams.finetuning.freeze_layers:
                param.requires_grad = False
            if "model_head" in name:
                param.requires_grad = True
            if "mask_token" in name:
                param.requires_grad = False

        print("Pretrained model ready.")

    def generate_fake_mask(self, batch_size: int, C: int, T: int) -> torch.Tensor:
        """Creates an all-false mask for TinyMyo's shared model interface.

        Args:
            batch_size (int): Batch size (B).
            C (int): Number of channels.
            T (int): Sequence length (tokens).

        Returns:
            torch.Tensor: Boolean mask of shape (B, C, T); no samples are masked.
        """
        return torch.zeros(batch_size, C, T, dtype=torch.bool).to(self.device)

    def _step(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        """Performs a TinyMyo forward pass and derives auxiliary class outputs.

        Args:
            X (torch.Tensor): Input EMG tensor of shape (B, C, T).
            mask (Optional[torch.Tensor]): All-false fine-tuning mask. Defaults to None.

        Returns:
            dict: Raw ``logits`` plus softmax ``probs`` and argmax ``label``.
                The regression path consumes only the raw logits.

        """
        # TinyMyo returns raw classification logits directly.
        y_pred_logits = self.model(X, mask=mask)
        y_pred_probs = torch.softmax(y_pred_logits, dim=1)
        y_pred_label = torch.argmax(y_pred_probs, dim=1)

        return {
            "label": y_pred_label,
            "probs": y_pred_probs,
            "logits": y_pred_logits,
        }

    def training_step(self, batch, batch_idx):
        batch = as_signal_batch(batch)
        X, y = batch["input"], batch["label"]
        if self.normalize:
            X = self.normalize_fct(X)
        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask=mask)
        loss = self.criterion(y_pred["logits"], y)

        if self.task == "regression":
            logits_flat = y_pred["logits"].reshape(-1, self.num_classes)  # (B*T, num_classes)
            y_flat = y.reshape(-1, self.num_classes)  # (B*T, num_classes)
            self.train_mean_metrics(logits_flat, y_flat)
            self.log_dict(self.train_mean_metrics, on_step=True, on_epoch=False)
        else:
            self.train_label_metrics(y_pred["label"], y)
            self.train_logit_metrics(self._handle_binary(y_pred["logits"]), y)
            self.log_dict(self.train_label_metrics, on_step=False, on_epoch=True)
            self.log_dict(self.train_logit_metrics, on_step=False, on_epoch=True)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = as_signal_batch(batch)
        X, y = batch["input"], batch["label"]
        if self.normalize:
            X = self.normalize_fct(X)
        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask=mask)
        loss = self.criterion(y_pred["logits"], y)

        if self.task == "regression":
            logits_flat = y_pred["logits"].reshape(-1, self.num_classes)  # (B*T, num_classes)
            y_flat = y.reshape(-1, self.num_classes)  # (B*T, num_classes)
            self.val_mean_metrics(logits_flat, y_flat)
            self.log_dict(self.val_mean_metrics, on_step=False, on_epoch=True)
        else:
            self.val_label_metrics(y_pred["label"], y)
            self.val_logit_metrics(self._handle_binary(y_pred["logits"]), y)
            self.log_dict(self.val_label_metrics, on_step=False, on_epoch=True)
            self.log_dict(self.val_logit_metrics, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch = as_signal_batch(batch)
        X, y = batch["input"], batch["label"]
        if self.normalize:
            X = self.normalize_fct(X)
        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask=mask)
        loss = self.criterion(y_pred["logits"], y)

        if self.task == "regression":
            logits_flat = y_pred["logits"].reshape(-1, self.num_classes)  # (B*T, num_classes)
            y_flat = y.reshape(-1, self.num_classes)  # (B*T, num_classes)
            self.test_mean_metrics(logits_flat, y_flat)
            self.log_dict(self.test_mean_metrics, on_step=False, on_epoch=True)
        else:
            self.test_label_metrics(y_pred["label"], y)
            self.test_logit_metrics(self._handle_binary(y_pred["logits"]), y)
            self.log_dict(self.test_label_metrics, on_step=False, on_epoch=True)
            self.log_dict(self.test_logit_metrics, on_step=False, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def lr_scheduler_step(self, scheduler: torch.optim.lr_scheduler._LRScheduler, metric: Optional[torch.Tensor]) -> None:
        """Advance the scheduler using its configured time unit.

        Args:
            scheduler (torch.optim.lr_scheduler._LRScheduler): The optimizer scheduler.
            metric (Optional[torch.Tensor]): Optional metric for ReduceLROnPlateau.
        """
        if self.hparams.scheduler.t_in_epochs:
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(self.global_step)

    def configure_optimizers(self) -> dict:
        """Configures optimizers and learning rate schedulers.

        Applies layer-wise learning-rate decay to TinyMyo transformer blocks.
        The classification head and non-block parameters receive the base LR;
        earlier encoder blocks receive progressively smaller LRs.

        Returns:
            dict: Configuration for the PyTorch Lightning trainer.

        Raises:
            NotImplementedError: If the optimizer name is not supported.
        """
        num_blocks = self.hparams.model.n_layer
        params_to_pass = []
        base_lr = self.hparams.optimizer.lr
        decay_factor = self.hparams.layerwise_lr_decay

        for name, param in self.model.named_parameters():
            lr = base_lr
            if name.startswith("blocks."):
                block_nr = int(name.split(".")[1])
                lr *= decay_factor ** (num_blocks - block_nr)
            params_to_pass.append({"params": param, "lr": lr})

        if self.hparams.optimizer.optim == "AdamW":
            optimizer = torch.optim.AdamW(
                params_to_pass,
                lr=base_lr,
                weight_decay=self.hparams.optimizer.weight_decay,
                betas=self.hparams.optimizer.betas,
            )
        else:
            raise NotImplementedError("No valid optimizer name")
            
        total_training_opt_steps = (
          self.hparams.scheduler.total_training_opt_steps
          if self.hparams.scheduler.t_in_epochs
          else self.trainer.estimated_stepping_batches
      )

        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer=optimizer,
            total_training_opt_steps=total_training_opt_steps,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch" if self.hparams.scheduler.t_in_epochs else "step",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _handle_binary(self, preds: torch.Tensor) -> torch.Tensor:
        """Slices logits for binary classification task.

        Args:
            preds (torch.Tensor): Logit outputs from the model.

        Returns:
            torch.Tensor: Logits/probabilities for the positive class if binary, else full preds.
        """
        if self.classification_task == "binary":
            return preds[:, 1].squeeze()
        else:
            return preds
