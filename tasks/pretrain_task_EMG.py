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
#* Author:  Matteo Fasulo                                                     *
#*----------------------------------------------------------------------------*
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch_optimizer as torch_optim
from einops import rearrange

from util.train_utils import MinMaxNormalization


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
        self.patch_size = self.hparams.masking.patch_size
        self.masking_ratio = self.hparams.masking.masking_ratio
        self.unmasked_loss_coeff = self.hparams.masking.unmasked_loss_coeff
        # Enable normalization if specified in parameters
        if (
            self.hparams.input_normalization is not None
            and self.hparams.input_normalization.normalize
        ):
            self.normalize = True
            self.normalize_fct = MinMaxNormalization()

    def generate_token_mask_and_ids(
        self,
        batch_size: int,
        C: int,
        T: int,
        patch_size: int,
        mask_ratio: float,
        device=None,
    ):
        """
        Generate token-level mask following the same pattern as `random_masking` in your model.

        Returns:
            ids_shuffle:    (B, N) LongTensor  -- indices used to shuffle tokens (argsort of noise)
            ids_restore:    (B, N) LongTensor  -- indices to restore original ordering (argsort of ids_shuffle)
            mask:           (B, N) BoolTensor  -- 1 indicates masked token (same semantics as your implementation)
        """
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and device is None
            else (device or "cpu")
        )

        # Number of tokens N = C * S where S = T // P
        assert T % patch_size == 0, "T must be divisible by patch_size"
        S = T // patch_size
        N = C * S

        B = batch_size
        len_keep = int(N * (1.0 - mask_ratio))

        noise = torch.rand(B, N, device=device)  # noise in [0,1)
        ids_shuffle = torch.argsort(noise, dim=1)  # shape (B, N)
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # shape (B, N)

        # build mask as in random_masking: first len_keep are kept (0), rest masked (1)
        mask = torch.ones(B, N, device=device, dtype=torch.uint8)  # 1 = masked
        mask[:, :len_keep] = 0
        # unshuffle to place mask in original token order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # convert to bool for downstream usage
        mask_bool = mask.bool()

        return ids_shuffle, ids_restore, mask_bool

    def token_mask_to_full_signal_mask(
        self, token_mask: torch.Tensor, C: int, T: int, patch_size: int
    ):
        """
        token_mask: (B, N) bool where N == C * (T // patch_size)
        returns: (B, C, T) bool
        """
        B, N = token_mask.shape
        S = T // patch_size
        assert N == C * S, f"token_mask N ({N}) != C*S ({C*S})"
        # expand each token to its patch samples then rearrange
        token_mask_expanded = token_mask.unsqueeze(-1).repeat(
            1, 1, patch_size
        )  # (B, N, P)
        full_mask = rearrange(
            token_mask_expanded, "b (c s) p -> b c (s p)", c=C, p=patch_size
        )
        return full_mask.bool()

    def unpatchify_to_signal(
        self, pred_patches: torch.Tensor, C: int, T: int, patch_size: int
    ):
        """
        pred_patches: (B, N, P) where N == C * (T // patch_size) and P == patch_size
        returns: (B, C, T)
        """
        B, N, P = pred_patches.shape
        S = T // patch_size
        assert N == C * S, f"N ({N}) != C * (T//P) ({C}*{S})"
        assert P == patch_size, f"P ({P}) != patch_size ({patch_size})"
        # tokens are ordered as (C blocks of S patches): (B, (C S), P)
        return rearrange(pred_patches, "b (c s) p -> b c (s p)", c=C)

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
        B, C, T = X.shape

        # Generate token-level mask and ids
        ids_shuffle, ids_restore, token_mask = self.generate_token_mask_and_ids(
            B, C, T, self.model.patch_size, self.masking_ratio, device=self.device
        )  # token_mask: (B, N) bool (True == masked)

        # normalize input if specified
        if self.normalize:
            X = self.normalize_fct(X)

        # patchify to tokens/embeddings
        x_tokens = self.model.patch_embedding(X)  # (B, N, D)

        # build masked tokens by inserting mask token where token_mask == True
        mask_token_expanded = self.model.mask_token.to(x_tokens.dtype).repeat(
            B, x_tokens.shape[1], 1
        )  # (B, N, D)
        x_masked = torch.where(
            token_mask.unsqueeze(-1), mask_token_expanded, x_tokens
        )  # (B, N, D)

        # full forward pass (direct token input)
        x_reconstructed, _ = self.model(x_masked, directly_input_tokens=True)

        pred_unpatchified = self.unpatchify_to_signal(
            x_reconstructed, C=C, T=T, patch_size=self.model.patch_size
        )  # (B, C, T)

        token_mask_full = self.token_mask_to_full_signal_mask(
            token_mask, C=C, T=T, patch_size=self.model.patch_size
        )  # (B, C, T)

        assert (
            pred_unpatchified.shape == X.shape
        ), f"recon {pred_unpatchified.shape} vs orig {X.shape}"
        assert (
            token_mask_full.shape == X.shape
        ), f"mask {token_mask_full.shape} vs orig {X.shape}"

        # Compute loss only on masked parts
        masked_loss, unmasked_loss = self.criterion(
            pred_unpatchified, X, token_mask_full
        )
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
        B, C, T = X.shape

        # Generate token-level mask and ids
        ids_shuffle, ids_restore, token_mask = self.generate_token_mask_and_ids(
            B, C, T, self.model.patch_size, self.masking_ratio, device=self.device
        )  # token_mask: (B, N) bool (True == masked)

        # normalize input if specified
        if self.normalize:
            X = self.normalize_fct(X)

        # patchify to tokens/embeddings
        x_tokens = self.model.patch_embedding(X)  # (B, N, D)

        # build masked tokens by inserting mask token where token_mask == True
        mask_token_expanded = self.model.mask_token.to(x_tokens.dtype).repeat(
            B, x_tokens.shape[1], 1
        )  # (B, N, D)
        x_masked = torch.where(
            token_mask.unsqueeze(-1), mask_token_expanded, x_tokens
        )  # (B, N, D)

        # full forward pass (direct token input)
        x_reconstructed, _ = self.model(x_masked, directly_input_tokens=True)

        pred_unpatchified = self.unpatchify_to_signal(
            x_reconstructed, C=C, T=T, patch_size=self.model.patch_size
        )  # (B, C, T)

        token_mask_full = self.token_mask_to_full_signal_mask(
            token_mask, C=C, T=T, patch_size=self.model.patch_size
        )  # (B, C, T)

        assert (
            pred_unpatchified.shape == X.shape
        ), f"recon {pred_unpatchified.shape} vs orig {X.shape}"
        assert (
            token_mask_full.shape == X.shape
        ), f"mask {token_mask_full.shape} vs orig {X.shape}"

        # Compute loss only on masked parts
        masked_loss, unmasked_loss = self.criterion(
            pred_unpatchified, X, token_mask_full
        )
        loss = masked_loss + self.unmasked_loss_coeff * unmasked_loss

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # Fixed indices for logging signals
        random_indices = [6, 16, 30]

        # Log signals with mask only for the first validation batch
        if batch_idx == 0:
            self.log_signals_with_mask(
                original=X.cpu().float(),
                reconstructed=pred_unpatchified.cpu().float(),
                mask=token_mask_full.cpu(),
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
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.hparams.optimizer.lr, momentum=0.9
            )
        elif self.hparams.optimizer.optim == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.hparams.optimizer.lr, weight_decay=0.01
            )
        elif self.hparams.optimizer.optim == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.hparams.optimizer.lr
            )
        elif self.hparams.optimizer.optim == "AdamW_finetune":
            # Fine-tuning with separate lr for linear_out layer
            linear_out_params = (
                self.model.linear_out.parameters()
                if not self.hparams.multi_gpu
                else self.model.module.linear_out.parameters()
            )
            ignored_params = list(map(id, linear_out_params))
            base_params = filter(
                lambda p: id(p) not in ignored_params, self.model.parameters()
            )

            optimizer = torch.optim.AdamW(
                [
                    {"params": base_params},
                    {"params": linear_out_params, "lr": self.hparams.optimizer.lr},
                ],
                lr=self.hparams.optimizer.lr * 0.1,
            )
        elif self.hparams.optimizer.optim == "LAMB":
            optimizer = torch_optim.Lamb(
                self.model.parameters(), lr=self.hparams.optimizer.lr
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

    def log_signals_with_mask(
        self, original, reconstructed, mask=None, batch_indices=None, indice_batch=None
    ):
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
