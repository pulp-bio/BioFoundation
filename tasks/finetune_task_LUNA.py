#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Berkay Döner                                                      *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
import torch_optimizer as torch_optim
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy, Precision, Recall, AUROC,
    AveragePrecision, CohenKappa, F1Score
)

class ChannelWiseNormalize:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, tensor):
        with torch.no_grad():
            # tensor: (B, C, T)
            mean = tensor.mean(dim=2, keepdim=True)
            std = tensor.std(dim=2, keepdim=True)
            return (tensor - mean) / (std + self.eps)

class FinetuneTask(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning a classification model, with support for:

    - Classification types:
        - `bc`: Binary Classification
        - `ml`: Multi-Label Classification
        - 'mc': Multi-Label Classification for TUAR
        - `mcc`: Multi-Class Classification
        - `mmc`: Multi-Class Multi-Output Classification
  
    - Metric logging during training, validation, and testing, including accuracy, precision, recall, F1 score, AUROC, and more
    - Optional input normalization with configurable normalization functions
    - Custom optimizer support including SGD, Adam, AdamW, and LAMB
    - Learning rate schedulers with configurable scheduling strategies
    - Layer-wise learning rate decay for fine-grained learning rate control across model blocks
    """
    def __init__(self, hparams):
        """
        Initialize the FinetuneTask module.

        Args:
            hparams (DictConfig): Hyperparameters and configuration loaded via Hydra.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.num_classes = self.hparams.model.num_classes
        self.classification_type = self.hparams.classification_type

        # Input normalization
        if self.hparams.input_normalization is not None and self.hparams.input_normalization.normalize:
            self.normalize = True
            self.normalize_fct = ChannelWiseNormalize()

        # Loss function
        if self.classification_type == "mc":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Classification mode detection
        if not isinstance(self.num_classes, int):
            raise TypeError("Number of classes must be an integer.")
        elif self.num_classes < 2:
            raise ValueError("Number of classes must be at least 2.")
        elif self.num_classes == 2:
            self.classification_task = "binary"
        else:
            self.classification_task = "multiclass"

        # Metrics
        label_metrics = MetricCollection([
            Accuracy(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            Recall(task='multiclass', num_classes=self.num_classes, average="macro"),
            Precision(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            F1Score(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            CohenKappa(task=self.classification_task, num_classes=self.num_classes)
        ])
        logit_metrics = MetricCollection([
            AUROC(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            AveragePrecision(task=self.classification_task, num_classes=self.num_classes, average="macro"),
        ])
        self.train_label_metrics = label_metrics.clone(prefix='train_')
        self.val_label_metrics = label_metrics.clone(prefix='val_')
        self.test_label_metrics = label_metrics.clone(prefix='test_')
        self.train_logit_metrics = logit_metrics.clone(prefix='train_')
        self.val_logit_metrics = logit_metrics.clone(prefix='val_')
        self.test_logit_metrics = logit_metrics.clone(prefix='test_')

    def load_pretrained_checkpoint(self, model_ckpt):
        """
        Load a pretrained model checkpoint and unfreeze specific layers for fine-tuning.
        """
        assert self.model.classifier is not None
        print("Loading pretrained checkpoint")
        ckpt = torch.load(model_ckpt)
        state_dict = ckpt['state_dict']
        # Remove decoder head and channel embedding weights since they are not needed for fine-tuning
        state_dict = {k: v for k, v in state_dict.items() if 'decoder_head' not in k and "channel_emb" not in k}
        ckpt['state_dict'] = state_dict
        self.model.load_state_dict(ckpt['state_dict'], strict=False)

        for name, param in self.model.named_parameters():
            if self.hparams.finetuning.freeze_layers:
                param.requires_grad = False
            if 'classifier' in name:
                param.requires_grad = True

        print("Pretrained model ready.")

    def generate_fake_mask(self, batch_size, C, T):
        """
        Create a dummy mask tensor to simulate attention masking.

        Args:
            batch_size (int): Number of samples.
            C (int): Number of channels.
            T (int): Temporal dimension.
        
        Returns:
            torch.Tensor: Boolean mask tensor of shape (B, C, T).
        """
        return torch.zeros(batch_size, C, T, dtype=torch.bool).to(self.device)

    def _step(self, X, mask, channel_locations):
        """
        Perform forward pass and post-process predictions.

        Args:
            X (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask tensor.

        Returns:
            dict: Dictionary containing predicted labels, probabilities, and logits.
        """
        y_pred_logits, _ = self.model(X, mask, channel_locations)

        if self.classification_type in ("bc", "mcc", "ml"):
            y_pred_probs = torch.softmax(y_pred_logits, dim=1)
            y_pred_label = torch.argmax(y_pred_probs, dim=1)
        elif self.classification_type == "mc":
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_label = torch.round(y_pred_probs)
        elif self.classification_type == "mmc":
            y_pred_logits = y_pred_logits.view(-1, 6)
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_label = torch.argmax(y_pred_probs, dim=-1)

        return {
            'label': y_pred_label,
            'probs': y_pred_probs,
            'logits': y_pred_logits,
        }

    def training_step(self, batch, batch_idx):
        X, y = batch["input"], batch["label"]
        channel_locations = batch["channel_locations"]
        if self.normalize:
            X = self.normalize_fct(X)
        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask, channel_locations)
        if self.classification_type == "mmc":
            y = y.view(-1)
            loss = self.criterion(y_pred['logits'], y)
        elif self.classification_type == "mc":
            loss = self.criterion(y_pred['logits'], y.float())
        else:
            loss = self.criterion(y_pred['logits'], y)
        self.train_label_metrics(y_pred['label'], y)
        self.train_logit_metrics(self._handle_binary(y_pred['logits']), y)
        self.log_dict(self.train_label_metrics, on_step=True, on_epoch=False)
        self.log_dict(self.train_logit_metrics, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["input"], batch["label"]
        channel_locations = batch["channel_locations"]
        if self.normalize:
            X = self.normalize_fct(X)
        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask, channel_locations)
        if self.classification_type == "mmc":
            y = y.view(-1)
            loss = self.criterion(y_pred['logits'], y)
        elif self.classification_type == "mc":
            loss = self.criterion(y_pred['logits'], y.float())
        else:
            loss = self.criterion(y_pred['logits'], y)

        self.val_label_metrics(y_pred['label'], y)
        self.val_logit_metrics(self._handle_binary(y_pred['logits']), y)
        self.log_dict(self.val_label_metrics, on_step=False, on_epoch=True)
        self.log_dict(self.val_logit_metrics, on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch["input"], batch["label"]
        channel_locations = batch["channel_locations"]
        if self.normalize:
            X = self.normalize_fct(X)
        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask, channel_locations)
        if self.classification_type == "mmc":
            y = y.view(-1)
            loss = self.criterion(y_pred['logits'], y)
        elif self.classification_type == "mc":
            loss = self.criterion(y_pred['logits'], y.float())
        else:
            loss = self.criterion(y_pred['logits'], y)

        self.test_label_metrics(y_pred['label'], y)
        self.test_logit_metrics(self._handle_binary(y_pred['logits']), y)
        self.log_dict(self.test_label_metrics, on_step=False, on_epoch=True)
        self.log_dict(self.test_logit_metrics, on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log_dict(self.train_label_metrics, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log_dict(self.train_logit_metrics, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_label_metrics, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log_dict(self.val_logit_metrics, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_label_metrics, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log_dict(self.test_logit_metrics, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)

    def lr_scheduler_step(self, scheduler, metric):
        """
        Custom scheduler step function for step-based LR schedulers
        """
        scheduler.step_update(num_updates=self.global_step)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: Configuration dictionary with optimizer and LR scheduler.
        """
        num_blocks = self.hparams.model.depth
        params_to_pass = []
        base_lr = self.hparams.optimizer.lr
        decay_factor = self.hparams.layerwise_lr_decay

        for name, param in self.model.named_parameters():
            lr = base_lr
            if 'blocks.' in name or 'norm_layers' in name:
                block_nr = int(name.split('.')[1])
                lr *= decay_factor ** (num_blocks - block_nr)
            params_to_pass.append({"params": param, "lr": lr})

        if self.hparams.optimizer.optim == "SGD":
            optimizer = torch.optim.SGD(params_to_pass, lr=base_lr, momentum=self.hparams.optimizer.momentum)
        elif self.hparams.optimizer.optim == 'Adam':
            optimizer = torch.optim.Adam(params_to_pass, lr=base_lr, weight_decay=self.hparams.optimizer.weight_decay)
        elif self.hparams.optimizer.optim == 'AdamW':
            optimizer = torch.optim.AdamW(params_to_pass, lr=base_lr, weight_decay=self.hparams.optimizer.weight_decay, betas=self.hparams.optimizer.betas)
        elif self.hparams.optimizer.optim == 'LAMB':
            optimizer = torch_optim.Lamb(params_to_pass, lr=base_lr)
        else:
            raise NotImplementedError("No valid optimizer name")

        if self.hparams.scheduler_type == "multi_step_lr":
            scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        else:
            scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer,
                                                total_training_opt_steps=self.trainer.estimated_stepping_batches)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _handle_binary(self, preds):
        """
        Special handling for binary classification probabilities.

        Args:
            preds (torch.Tensor): Logit outputs.

        Returns:
            torch.Tensor: Probabilities for the positive class.
        """
        if self.classification_task == 'binary' and self.classification_type != 'mc':
            return preds[:, 1].squeeze()
        else:
            return preds
