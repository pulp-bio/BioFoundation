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
# * Author: Matteo Fasulo                                                     *
# *----------------------------------------------------------------------------*
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from biofoundation.core.batch import as_signal_batch
from util.train_utils import MinMaxNormalization


class MaskTask(pl.LightningModule):
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
        Fully independent random masking across both time and channels.

        Returns:
            mask_full (torch.BoolTensor): Shape (B, C, T)
                True = masked element
        """
        patch_H, patch_W = self.patch_size
        num_patches_H = C // patch_H
        num_patches_W = T // patch_W

        # Total number of patches per sample (e.g., 16 channels * 50 time patches = 800)
        num_patches_total = num_patches_H * num_patches_W

        # Number of patches to mask per sample
        num_to_mask = int(num_patches_total * self.masking_ratio)

        # Generate a flat mask over ALL patches (B, num_patches_total)
        mask_flat = torch.zeros(
            batch_size, num_patches_total, dtype=torch.bool, device=self.device
        )

        for b in range(batch_size):
            selected = torch.randperm(num_patches_total, device=self.device)[
                :num_to_mask
            ]
            mask_flat[b, selected] = True

        # Reshape the flat mask back into the 2D grid of patches (B, num_patches_H, num_patches_W)
        mask_patches_2d = mask_flat.view(batch_size, num_patches_H, num_patches_W)

        # Expand to full signal shape using repeat_interleave
        # (B, num_patches_H, num_patches_W) -> (B, C, T)
        mask_full = mask_patches_2d.repeat_interleave(patch_H, dim=1).repeat_interleave(
            patch_W, dim=2
        )

        return mask_full

    def unpatchify(self, x_patches: torch.Tensor, in_chans: int) -> torch.Tensor:
        """
        Convert patch embeddings (B, N, P) back to waveform (B, C, T)
        """
        B, N, P = x_patches.shape
        num_patches_per_chan = N // in_chans
        x_recon = x_patches.reshape(B, in_chans, num_patches_per_chan * P)
        return x_recon

    def _step(self, X):
        X = as_signal_batch(X)["input"]
        B, C, T = X.shape

        # Detect zero-padded channels (Shape: B, C. True if channel is 0.0 padded)
        pad_mask_ch = (X.abs().max(dim=-1).values == 0)

        # Generate symmetrical time mask
        mask = self.generate_mask(B, C, T)

        # Remove padded channels from the mask
        # Broadcast pad_mask_ch (B, C) to (B, C, T) and set mask to False
        mask[pad_mask_ch.unsqueeze(-1).expand(-1, -1, T)] = False

        if self.normalize:
            X = self.normalize_fct(X)

        # Pass pad_mask_ch to the model so the attention can ignore them.
        x_reconstructed = self.model(X, mask=mask, pad_mask_ch=pad_mask_ch)

        return {
            "x_original": X,
            "x_reconstructed": x_reconstructed,
            "mask": mask,
        }


    def training_step(self, X, batch_idx):
        batch = as_signal_batch(X)
        out = self._step(batch)
        x_original = out["x_original"]
        mask = out["mask"]

        x_reconstructed = self.unpatchify(out["x_reconstructed"], in_chans=self.model.in_chans)
        masked_loss, unmasked_loss = self.criterion(x_reconstructed, x_original, mask)
        loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss

        losses = {
            "loss": loss,
            "masked_loss": masked_loss,
            "unmasked_loss": unmasked_loss,
        }

        self.log_dict({f"train_{k}": v for k, v in losses.items()}, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, X, batch_idx):
        batch = as_signal_batch(X)
        out = self._step(batch)
        x_original = out["x_original"]
        x_reconstructed = self.unpatchify(out["x_reconstructed"], in_chans=self.model.in_chans)
        mask = out["mask"]

        masked_loss, unmasked_loss = self.criterion(x_reconstructed, x_original, mask)
        loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss

        losses = {
            "loss": loss,
            "masked_loss": masked_loss,
            "unmasked_loss": unmasked_loss,
        }

        self.log_dict({f"val_{k}": v for k, v in losses.items()}, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Fixed indices for logging signals
        random_indices = [6, 16, 30]

        # Log signals with mask only for the first validation batch
        if batch_idx == 0:
            self.log_signals_with_mask(
                x_original.float(),
                x_reconstructed.float(),
                mask,
                batch_indices=random_indices,
            )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )
        scheduler_in_epochs = self.hparams.scheduler.t_in_epochs
        total_training_opt_steps = (
            self.hparams.scheduler.total_training_opt_steps
            if scheduler_in_epochs
            else self.trainer.estimated_stepping_batches
        )
        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer=optimizer,
            total_training_opt_steps=total_training_opt_steps,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch" if scheduler_in_epochs else "step",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler, metric):
        if self.hparams.scheduler.t_in_epochs:
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step(self.global_step)

    def log_signals_with_mask(self, original, reconstructed, mask=None, batch_indices=None):
        """
        Log original and reconstructed signals highlighting masked regions.

        Args:
            original (torch.Tensor): Original signals.
            reconstructed (torch.Tensor): Signals reconstructed by the model.
            mask (torch.BoolTensor, optional): Applied mask.
            batch_indices (list[int], optional): Batch indices to log.
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
                            ax.axvspan(j * patch_W, (j + 1) * patch_W, color='lightgray', alpha=0.3)
                            indices.append(j)

            ax.set_title(f"Signal Reconstruction - batch_ {batch_idx}")
            ax.legend()

            # Log the figure to WandB
            if self.trainer.is_global_zero:
                wandb_logger = None
                for logger in self.trainer.loggers:
                    if isinstance(logger, WandbLogger):
                        wandb_logger = logger
                        break

                if wandb_logger:
                    wandb_logger.experiment.log({
                        "reconstruction/channel0": wandb.Image(
                            fig,
                            caption=f"epoch={self.current_epoch}, batch={batch_idx}"
                        )
                    }, step=self.global_step)

            plt.close(fig)
