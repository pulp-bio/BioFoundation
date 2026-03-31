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
#* Author:  Danaé Broustail                                                   *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
import torch_optimizer as torch_optim
import torch.nn.functional as F
from torchmetrics.regression import (
    R2Score,
    MeanSquaredError,
    PearsonCorrCoef
)

from tasks.finetune_task_LUNA import ChannelWiseNormalize
from safetensors.torch import load_file
from collections import OrderedDict


class FinetuneRegressionTask(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning a regression model (MoBI-style).
    - Multi-output regression (e.g. 12 joint angles)
    - MSE loss
    - Metrics:
        - R2 (model selection metric)
        - Pearson correlation
        - RMSE
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = hydra.utils.instantiate(self.hparams.model)

        # Number of regression targets (MoBI = 12)
        self.num_outputs = self.hparams.model.num_classes  # reuse config field

        # Input normalization
        self.normalize = False
        if self.hparams.input_normalization is not None and self.hparams.input_normalization.normalize:
            self.normalize = True
            self.normalize_fct = ChannelWiseNormalize()

        # Loss
        self.criterion = nn.MSELoss()

        # Metrics (epoch-level aggregation!)
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        self.train_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)
        self.val_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)
        self.test_pearson = PearsonCorrCoef(num_outputs=self.num_outputs)

    def load_pretrained_checkpoint(self, model_ckpt):
        """
        Load a pretrained model checkpoint and unfreeze specific layers for fine-tuning.
        """
        assert self.model.classifier is not None
        print("Loading pretrained checkpoint")
        ckpt = torch.load(model_ckpt, weights_only=False)
        state_dict = ckpt['state_dict']
        # Remove decoder head and channel embedding weights since they are not needed for fine-tuning
        state_dict = {k: v for k, v in state_dict.items() if 'decoder_head' not in k and "channel_emb" not in k}

        # added to remove "model." prefix if exists 
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("model.", "")  # sometimes Lightning saves weights as model.layer...
            new_state_dict[new_key] = v

        ckpt['state_dict'] = new_state_dict # state_dict without "model." prefix
        missing_keys, unexpected_keys = self.model.load_state_dict(ckpt['state_dict'], strict=False)

        print("Missing keys when loading pretrained checkpoint:", missing_keys)
        print("Unexpected keys when loading pretrained checkpoint:", unexpected_keys)        

        for name, param in self.model.named_parameters():
            if self.hparams.finetuning.freeze_layers:
                param.requires_grad = False
            if 'classifier' in name:
                print("Unfreezing layer:", name)
                param.requires_grad = True

        print("Pretrained model ready.")

    def load_safetensors_checkpoint(self, model_ckpt):
        """
        Load a pretrained model checkpoint in safetensors format and unfreeze specific layers for fine-tuning.
        """
        assert self.model.classifier is not None
        print("Loading pretrained safetensors checkpoint")
        state_dict = load_file(model_ckpt)

        # add model. prefix if needed
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith("model."):
                new_key = "model." + k
            else:
                new_key = k
            new_state_dict[new_key] = v

        state_dict = {k: v for k, v in new_state_dict.items() if 'decoder_head' not in k and "channel_emb" not in k}
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        print("Missing keys when loading pretrained checkpoint:", missing_keys)
        print("Unexpected keys when loading pretrained checkpoint:", unexpected_keys)    

        for name, param in self.model.named_parameters():
            if self.hparams.finetuning.freeze_layers:
                param.requires_grad = False
            if 'classifier' in name:
                param.requires_grad = True

        print("Pretrained model ready.")

    def generate_fake_mask(self, batch_size, C, T):
        return torch.zeros(batch_size, C, T, dtype=torch.bool).to(self.device)

    def _step(self, X, mask, channel_locations):
        y_pred, _ = self.model(X, mask, channel_locations)
        return y_pred

    def training_step(self, batch, batch_idx):
        X, y = batch["input"], batch["label"]
        channel_locations = batch["channel_locations"]

        if self.normalize:
            X = self.normalize_fct(X)

        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask, channel_locations)
        loss = self.criterion(y_pred, y)

        self.train_r2(y_pred, y)
        self.train_rmse(y_pred, y)
        self.train_pearson(y_pred, y)

        # Log metrics per step (Lightning handles aggregation / syncing)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_pearson", torch.mean(self.train_pearson.compute()), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["input"], batch["label"]
        channel_locations = batch["channel_locations"]

        if self.normalize:
            X = self.normalize_fct(X)

        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask, channel_locations)

        loss = self.criterion(y_pred, y)

        self.val_r2(y_pred, y)
        self.val_rmse(y_pred, y)
        self.val_pearson(y_pred, y)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_pearson", torch.mean(self.val_pearson.compute()), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch["input"], batch["label"]
        channel_locations = batch["channel_locations"]

        if self.normalize:
            X = self.normalize_fct(X)

        mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
        y_pred = self._step(X, mask, channel_locations)

        loss = self.criterion(y_pred, y)

        self.test_r2(y_pred, y)
        self.test_rmse(y_pred, y)
        self.test_pearson(y_pred, y)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_pearson", torch.mean(self.test_pearson.compute()), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        return loss

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
        # LUNA version
        num_blocks = self.hparams.model.depth if hasattr(self.hparams.model, 'depth') else None
        if num_blocks is None: # LuMamba version
            num_blocks = self.hparams.model.num_blocks if hasattr(self.hparams.model, 'num_blocks') else None
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