#*----------------------------------------------------------------------------*
#* Copyright (C) 2026 ETH Zurich, Switzerland                                 *
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
#* Author:  Glenn Anta Bucagu                                                 *
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

from typing import Any, Dict, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch_optimizer as torch_optim

from biofoundation.core.batch import BatchRequirements, as_signal_batch, require_batch_fields
from biofoundation.core.checkpoints import SafetensorsCheckpointMixin
from biofoundation.model_registry import get_model_spec
from models.modules.patching import patchify


def extract_encoder_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Pull encoder weights out of a Lightning checkpoint.

    Args:
        checkpoint: Checkpoint dictionary containing a ``state_dict`` entry.

    Returns:
        State dict for the encoder, with the ``model.`` prefix removed and any
        head parameters dropped.
    """
    state_dict = checkpoint["state_dict"]
    return {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.") and not key.startswith("model_head.")
    }


class MaskedAutoencoderPretrainingTask(SafetensorsCheckpointMixin, pl.LightningModule):
    """SimMIM-style masked reconstruction pre-training for the CEReBrO encoder.

    Waveforms are patched and embedded, a random subset of real (non-padded) tokens
    is replaced by a learned mask token, and the encoder sees the full sequence of
    visible and masked tokens. A linear decoder reconstructs every patch and the loss
    is taken over the masked positions. Padded channels are replaced by a learned pad
    token, are excluded from masking, and are masked out of attention.

    Args:
        hparams: Full experiment configuration, used to instantiate the encoder,
            decoder, criterion, optimiser and scheduler.
        masking_ratio: Fraction of real tokens replaced by the mask token.
    """

    def __init__(self, hparams, masking_ratio: float = 0.5):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.model_head = hydra.utils.instantiate(self.hparams.model_head)
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)

        family = self.hparams.get("model_family", None)
        self.batch_requirements = (
            get_model_spec(family).batch_requirements if family else BatchRequirements()
        )

        self.masking_ratio = masking_ratio
        self.patch_size = self.hparams.model.patch_size
        self.num_channels = self.hparams.model.num_channels
        self.embed_dim = self.hparams.model.embed_dim
        self.mask_token = self.model.mask_token
        self.pad_token = self.model.pad_token
        self.strict_loading = False

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:
        """Reshape raw waveforms into patches once the batch is on device."""
        x = batch["input"]
        if x.dim() == 3:
            batch_size, channels, _ = x.shape
            batch["input"] = x.reshape(batch_size, channels, -1, self.patch_size)
        return batch

    def _shared_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run masking, encoding, decoding and loss for one batch."""
        require_batch_fields(batch, self.batch_requirements)
        x = batch["input"]
        batch_size, channels = x.shape[0], x.shape[1]

        tokens = self.model.patch_embed(x)
        tokens, token_mask, attn_mask = self.prepare_tokens(
            tokens, num_padded_channels=batch.get("num_padded_channels")
        )

        latent = self.model(
            tokens,
            channel_positions=batch["channel_coords"],
            directly_input_tokens=True,
            attn_mask=attn_mask,
        )
        pred = self.model_head(latent)

        batch["token_mask"] = token_mask
        batch["attn_mask"] = attn_mask
        batch["target"] = patchify(x.reshape(batch_size, channels, -1), patch_size=self.patch_size)

        loss, _ = self.criterion(pred, batch)
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute and log the training reconstruction loss."""
        loss = self._shared_step(as_signal_batch(batch))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute and log the validation reconstruction loss."""
        loss = self._shared_step(as_signal_batch(batch))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def prepare_tokens(
        self, tokens: torch.Tensor, num_padded_channels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Insert pad tokens, then mask a random subset of the real tokens.

        Args:
            tokens: Token embeddings of shape ``(batch, num_tokens, embed_dim)``.
            num_padded_channels: Per-sample count of trailing padded channels.

        Returns:
            Tuple of tokens reshaped to ``(batch, num_channels, num_patches, embed_dim)``,
            a boolean ``(batch, num_tokens)`` mask marking masked tokens, and an
            integer ``(batch, num_tokens)`` attention mask (``None`` when nothing is padded).
        """
        batch_size, num_tokens, embed_dim = tokens.shape
        channels = self.num_channels
        patches = num_tokens // channels

        attn_mask = None
        if num_padded_channels is not None:
            num_real_channels = channels - num_padded_channels
            channel_indices = torch.arange(channels, device=tokens.device).unsqueeze(0)
            padded = channel_indices >= num_real_channels.unsqueeze(1)
            padded = padded.repeat_interleave(patches, dim=1)
            attn_mask = (~padded).int()
            tokens = torch.where(padded.unsqueeze(-1), self.pad_token.to(tokens.dtype), tokens)

        tokens, token_mask = self.mask_tokens(tokens, attn_mask)
        return tokens.reshape(batch_size, channels, patches, embed_dim), token_mask, attn_mask

    def mask_tokens(
        self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace a random subset of real tokens with the learned mask token.

        Tokens are ranked by uniform noise, with padded positions pushed to the end so
        they are never selected. The first ``1 - masking_ratio`` fraction of each
        sample's real tokens is kept and the remainder is masked in place, so the
        sequence order the encoder sees is unchanged.

        Args:
            tokens: Token embeddings of shape ``(batch, num_tokens, embed_dim)``.
            attn_mask: Optional ``(batch, num_tokens)`` mask, 1 for real tokens.

        Returns:
            Tuple of the masked tokens and a boolean ``(batch, num_tokens)`` mask
            where True marks a masked token.
        """
        batch_size, num_tokens, _ = tokens.shape
        device = tokens.device

        noise = torch.rand(batch_size, num_tokens, device=device)
        if attn_mask is not None:
            noise = noise.masked_fill(attn_mask == 0, 2.0)
            valid_length = attn_mask.sum(dim=1)
        else:
            valid_length = torch.full((batch_size,), num_tokens, device=device, dtype=torch.long)

        rank = torch.argsort(torch.argsort(noise, dim=1), dim=1)
        num_keep = (valid_length * (1 - self.masking_ratio)).to(torch.long)
        token_mask = (rank >= num_keep.unsqueeze(1)) & (rank < valid_length.unsqueeze(1))

        masked = torch.where(token_mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)
        return masked, token_mask

    def configure_optimizers(self) -> Dict[str, Any]:
        """Build the optimiser and the per-step learning-rate scheduler."""
        params = list(self.model.parameters()) + list(self.model_head.parameters())
        optimizer_name = str(self.hparams.optimizer.optim).lower()
        lr = float(self.hparams.optimizer.lr)
        weight_decay = float(getattr(self.hparams.optimizer, "weight_decay", 0.0))

        if optimizer_name == "adamw":
            betas = tuple(getattr(self.hparams.optimizer, "betas", (0.9, 0.999)))
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = float(getattr(self.hparams.optimizer, "momentum", 0.9))
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "lamb":
            optimizer = torch_optim.Lamb(params, lr=lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.hparams.optimizer.optim}")

        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler,
            optimizer=optimizer,
            total_training_opt_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def lr_scheduler_step(self, scheduler, metric) -> None:
        """Advance the timm-style scheduler once per optimiser step."""
        scheduler.step_update(num_updates=self.global_step)

    def load_from_checkpoint(
        self, checkpoint_path, map_location=None, hparams_file=None, strict=None, **kwargs
    ) -> "MaskedAutoencoderPretrainingTask":
        """Load encoder weights from a checkpoint, skipping the decoder.

        Shape-mismatched tensors are left at their initialised values so that an
        encoder pre-trained at one channel count or window length can seed another.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        incoming = extract_encoder_state_dict(checkpoint)
        current = self.model.state_dict()

        loaded, skipped = [], []
        for key, value in incoming.items():
            if key in current and value.shape == current[key].shape:
                current[key] = value
                loaded.append(key)
            else:
                skipped.append(key)

        self.model.load_state_dict(current, strict=False)
        print(f"[load] encoder tensors loaded: {len(loaded)}, skipped: {len(skipped)}")
        if skipped:
            print(f"[load] skipped keys (first 10): {skipped[:10]}")
        return self
