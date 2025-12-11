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
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch_optimizer as torch_optim
from omegaconf import DictConfig

from util.train_utils import MinMaxNormalization


class MaskTask(pl.LightningModule):
    """
    PyTorch Lightning module for training a model with masked reconstruction.

    Args:
        hparams (DictConfig): Parameters and configurations loaded via Hydra.
    """

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)
        self.patch_size = self.hparams.masking.patch_size
        self.masking_ratio = self.hparams.masking.masking_ratio
        self.unmasked_loss_coeff = self.hparams.masking.unmasked_loss_coeff

        # Enable normalization if specified in parameters
        self.normalize = False
        if "input_normalization" in self.hparams and self.hparams.input_normalization.normalize:
            self.normalize = True
            self.normalize_fct = MinMaxNormalization()

    def generate_mask(self, batch_size, C, T):
        """
        Generate per-sample patch-level boolean masks (MAE-style).

        Returns:
            mask_full (torch.BoolTensor): Shape (B, C, T)
                True = masked element
        """
        patch_H, patch_W = self.patch_size
        num_patches_H = C // patch_H
        num_patches_W = T // patch_W
        N = num_patches_H * num_patches_W

        # Number of patches to mask per sample
        num_to_mask = int(N * self.masking_ratio)

        # Generate patch-level mask (B, N) - vectorized
        mask_patches = torch.zeros(batch_size, N, dtype=torch.bool, device=self.device)

        for b in range(batch_size):
            selected = torch.randperm(N, device=self.device)[:num_to_mask]
            mask_patches[b, selected] = True

        # unpatchify using reshape and repeat_interleave
        # (B, N) -> (B, num_patches_H, num_patches_W)
        mask_patches_2d = mask_patches.reshape(batch_size, num_patches_H, num_patches_W)

        # Expand to full shape using repeat_interleave
        # (B, num_patches_H, num_patches_W) -> (B, C, T)
        mask_full = mask_patches_2d.repeat_interleave(patch_H, dim=1).repeat_interleave(patch_W, dim=2)

        return mask_full

    def unpatchify(self, x_patches: torch.Tensor, in_chans: int) -> torch.Tensor:
        """
        Convert patch embeddings (B, N, P) back to waveform (B, C, T)

        Args:
            x_patches: (B, N, P)
            in_chans: number of channels C
        Returns:
            x_reconstructed: (B, C, T)
        """
        B, N, P = x_patches.shape
        num_patches_per_chan = N // in_chans
        x_recon = x_patches.reshape(B, in_chans, num_patches_per_chan * P)
        return x_recon

    def training_step(self, batch, batch_idx):
        """
        Training step: apply mask, normalize and compute loss.

        Args:
            batch (torch.Tensor): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        X = batch
        mask = self.generate_mask(X.shape[0], X.shape[1], X.shape[2])

        if self.normalize:
            X = self.normalize_fct(X)

        x_reconstructed, x_original = self.model(X, mask=mask)  # x_reconstructed: (B, N, P)

        # unpatchify to original signal shape (B, C, T)
        x_reconstructed_unpatched = self.unpatchify(x_reconstructed, self.hparams.model.in_chans)

        # Compute loss on masked parts and unmasked parts (with coefficient)
        masked_loss, unmasked_loss = self.criterion(x_reconstructed_unpatched, x_original, mask)
        loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss

        self.log(
            "train_loss",
            masked_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
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
        X = batch
        mask = self.generate_mask(X.shape[0], X.shape[1], X.shape[2])

        if self.normalize:
            X = self.normalize_fct(X)

        x_reconstructed, x_original = self.model(X, mask=mask)  # x_reconstructed: (B, N, P)

        # unpatchify to original signal shape (B, C, T)
        x_reconstructed_unpatched = self.unpatchify(x_reconstructed, self.hparams.model.in_chans)

        # Compute loss on masked parts and unmasked parts (with coefficient)
        masked_loss, unmasked_loss = self.criterion(x_reconstructed_unpatched, x_original, mask)
        loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # Fixed indices for logging signals
        random_indices = [6, 16, 30]

        # Log signals with mask only for the first validation batch
        if batch_idx == 0:
            self.log_signals_with_mask(
                x_original.float(),
                x_reconstructed_unpatched.float(),
                mask,
                batch_indices=random_indices,
                indice_batch=batch_idx,
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
        elif self.hparams.optimizer.optim == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay,
            )
        elif self.hparams.optimizer.optim == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay,
            )
        elif self.hparams.optimizer.optim == "LAMB":
            optimizer = torch_optim.Lamb(
                self.model.parameters(),
                lr=self.hparams.optimizer.lr,
            )
        else:
            raise NotImplementedError("No valid optim name")

        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

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

            ax.plot(
                original_signal_c2[0].cpu().numpy(),
                label="Original Channel 0",
                color="blue",
                alpha=0.7,
            )
            ax.plot(
                reconstructed_signal_c2[0].cpu().numpy(),
                label="Reconstructed Channel 0",
                color="orange",
                alpha=0.7,
            )

            if mask is not None:
                mask_c2 = mask[batch_idx, :patch_H, :]
                indices = []

                # Highlight masked regions with a light gray transparent band
                for i in range(patch_H):
                    for j in range(T // patch_W):
                        if mask_c2[i, j * patch_W : (j + 1) * patch_W].all():
                            ax.axvspan(
                                j * patch_W,
                                (j + 1) * patch_W,
                                color="lightgray",
                                alpha=0.1,
                            )
                            indices.append(j)

            # Remove duplicates and sort highlighted indices
            indices_array = np.array(indices)
            indices_array = np.unique(indices)

            ax.set_title(f"Signal Reconstruction - batch_ {batch_idx}")
            ax.legend()

            # Log the figure on TensorBoard with batch and index in the title
            self.logger.experiment.add_figure(
                f"Original and Reconstructed Signals with Mask (batch_0_ {batch_idx}, F1 = 0)",
                fig,
                self.current_epoch,
            )
            plt.close(fig)
