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
import pytorch_lightning as pl
import hydra
import torch_optimizer as torch_optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from criterion.query_specialization_criterion import QuerySpecializationCriterion

class ChannelWiseNormalize:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, tensor):
        with torch.no_grad():
            # tensor: (B, C, T)
            mean = tensor.mean(dim=2, keepdim=True)
            std = tensor.std(dim=2, keepdim=True)
            return (tensor - mean) / (std + self.eps)
        
class MaskTask(pl.LightningModule):
    """
    PyTorch Lightning module for training a model with masked reconstruction.

    Args:
        hparams (DictConfig): Parameters and configurations loaded via Hydra.
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)
        self.query_specialization_criterion = QuerySpecializationCriterion(**self.hparams.query_specialization_criterion)
        self.patch_size = self.hparams.masking.patch_size
        self.masking_ratio = self.hparams.masking.masking_ratio
        self.unmasked_loss_coeff = self.hparams.masking.unmasked_loss_coeff
        # Enable normalization if specified in parameters
        if self.hparams.input_normalization is not None and self.hparams.input_normalization.normalize:
            self.normalize = True
            self.normalize_fct = ChannelWiseNormalize()

    def generate_mask(self, batch_size, C, T):
        """
        Generate a boolean mask for block-wise rectangular masking.

        Args:
            batch_size (int): Batch size.
            C (int): Number of channels (height).
            T (int): Temporal length (width).

        Returns:
            torch.BoolTensor: Boolean mask of shape (batch_size, C, T),
                              with True in the masked regions.
        """
        patch_H, patch_W = self.patch_size
        masking_ratio = self.masking_ratio

        # Calculate total number of patch rectangles
        num_rectangles = (C // patch_H) * (T // patch_W)
        num_to_mask = int(num_rectangles * masking_ratio)

        row_indices = torch.arange(0, C, patch_H)
        col_indices = torch.arange(0, T, patch_W)
        rectangles = [(i, j) for i in row_indices for j in col_indices]

        # Randomly select which rectangles to mask
        selected_indices = torch.randperm(num_rectangles)[:num_to_mask]

        mask = torch.zeros(batch_size, C, T, dtype=torch.bool).to(self.device)

        # Set mask to True in the selected regions
        for idx in selected_indices:
            r, c = rectangles[idx]
            mask[:, r:r + patch_H, c:c + patch_W] = True  

        return mask

    def training_step(self, batch, batch_idx):
        """
        Training step: apply mask, normalize and compute loss.

        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        X = batch["input"]
        channel_locations = batch["channel_locations"]
        channel_names = batch.get("channel_names", None)
        mask = self.generate_mask(X.shape[0], X.shape[1], X.shape[2])

        if self.normalize:
            X = self.normalize_fct(X)

        # Pass masked input through the model to get reconstruction and embeddings
        x_reconstructed, x_original, attention_scores = self.model(X, mask, channel_locations, channel_names)

        # Compute loss only on masked parts
        masked_loss, unmasked_loss = self.criterion(x_reconstructed, x_original, mask)
        loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss
        if self.hparams.query_specialization_criterion is not None:
            query_specialization_loss = self.query_specialization_criterion(attention_scores)
            loss += query_specialization_loss
            self.log('query_specialization_loss', query_specialization_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('train_loss', masked_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step: apply mask, normalize, compute loss and log signals.

        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        X = batch["input"]
        channel_locations = batch["channel_locations"]
        channel_names = batch.get("channel_names", None)
        mask = self.generate_mask(X.shape[0], X.shape[1], X.shape[2])

        if self.normalize:
            X = self.normalize_fct(X)

        x_reconstructed, x_original, attention_scores = self.model(X, mask, channel_locations, channel_names)

        masked_loss, unmasked_loss = self.criterion(x_reconstructed, x_original, mask)
        loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss

        if self.hparams.query_specialization_criterion is not None:
            query_specialization_loss = self.query_specialization_criterion(attention_scores)
            loss += query_specialization_loss
            self.log('query_specialization_loss', query_specialization_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True) 
        
        # Fixed indices for logging signals
        random_indices = [6, 16, 30]

        # Log signals with mask only for the first validation batch
        if batch_idx == 0:
            self.log_signals_with_mask(
                x_original.float(),
                x_reconstructed.float(),
                mask,
                batch_indices=random_indices,
                indice_batch=batch_idx
            )
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler based on parameters.

        Returns:
            dict: Dictionary with optimizer and scheduler for PyTorch Lightning.
        """
        if self.hparams.optimizer.optim == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.optimizer.lr, momentum=0.9)
        elif self.hparams.optimizer.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.optimizer.lr, weight_decay=0.01)
        elif self.hparams.optimizer.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.optimizer.lr)
        elif self.hparams.optimizer.optim == 'LAMB':
            optimizer = torch_optim.Lamb(self.model.parameters(), lr=self.hparams.optimizer.lr)
        else:
            raise NotImplementedError("No valid optim name")

        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer, total_training_opt_steps=self.trainer.estimated_stepping_batches)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)
        
    def log_signals_with_mask(self, original, reconstructed, mask=None, batch_indices=None, indice_batch=None):
        """
        Log original and reconstructed signals highlighting masked regions.

        Args:
            original (torch.Tensor): Original signals.
            reconstructed (torch.Tensor): Signals reconstructed by the model.
            mask (torch.BoolTensor, optional): Applied mask.
            batch_indices (list[int], optional): Batch indices to log.
            indice_batch (int, optional): Current batch index.
        """
        patch_H, patch_W = self.patch_size
        batch_size, C, T = original.shape

        for batch_idx in batch_indices:
            original_signal = original[batch_idx]
            reconstructed_signal = reconstructed[batch_idx]

            fig, ax = plt.subplots(1, 1, figsize=(15, 6))

            # Limit visualization to the first patch_H channels
            original_signal_c2 = original_signal[:patch_H, :]
            reconstructed_signal_c2 = reconstructed_signal[:patch_H, :]

            ax.plot(original_signal_c2[0].cpu().numpy(), label='Original Channel 0', color='blue', alpha=0.7)
            ax.plot(reconstructed_signal_c2[0].cpu().numpy(), label='Reconstructed Channel 0', color='orange', alpha=0.7)

            if mask is not None:
                mask_c2 = mask[batch_idx, :patch_H, :]
                indices = []

                # Highlight masked regions with a light gray transparent band
                for i in range(patch_H):
                    for j in range(T // patch_W):
                        if mask_c2[i, j * patch_W:(j + 1) * patch_W].all():
                            ax.axvspan(j * patch_W, (j + 1) * patch_W, color='lightgray', alpha=0.1)
                            indices.append(j)

            # Remove duplicates and sort highlighted indices
            indices_array = np.array(indices)
            indices_array = np.unique(indices) 

            ax.set_title(f"Signal Reconstruction - batch_ {batch_idx}")
            ax.legend()

            # Log the figure on TensorBoard with batch and index in the title
            self.logger.experiment.add_figure(f'Original and Reconstructed Signals with Mask (batch_0_ {batch_idx}, F1 = 0)', fig, self.current_epoch)
            plt.close(fig)
