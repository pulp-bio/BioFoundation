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

from typing import Any, Dict

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from biofoundation.core.batch import BatchRequirements, as_signal_batch, require_batch_fields
from biofoundation.core.checkpoints import SafetensorsCheckpointMixin
from biofoundation.model_registry import get_model_spec
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, R2Score

from tasks.classification_task import split_checkpoint_state_dict


class NormalizedRootMeanSquaredError(Metric):
    """Root mean squared error divided by the spread of the targets.

    Normalising by the target standard deviation (or mean) makes the error
    comparable across datasets with different target scales. Statistics are
    accumulated over the whole evaluation split rather than per batch.

    Args:
        normalization: ``std`` to divide by the target standard deviation, ``mean`` to
            divide by the absolute target mean.
    """

    def __init__(self, normalization: str = "std"):
        super().__init__()
        if normalization not in {"std", "mean"}:
            raise ValueError(f"normalization must be 'std' or 'mean', got '{normalization}'")
        self.normalization = normalization
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_observations", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate error and target statistics for one batch."""
        preds = preds.float().flatten()
        target = target.float().flatten()
        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.sum_target += torch.sum(target)
        self.sum_squared_target += torch.sum(target ** 2)
        self.num_observations += target.numel()

    def compute(self) -> torch.Tensor:
        """Return the normalised RMSE over everything accumulated so far."""
        rmse = torch.sqrt(self.sum_squared_error / self.num_observations)
        mean_target = self.sum_target / self.num_observations

        if self.normalization == "mean":
            norm = torch.abs(mean_target)
        else:
            variance = (self.sum_squared_target / self.num_observations) - mean_target ** 2
            norm = torch.sqrt(torch.clamp(variance, min=0.0))

        return rmse / torch.clamp(norm, min=1e-8)


class RegressionTask(SafetensorsCheckpointMixin, pl.LightningModule):
    """Fine-tuning task for scalar EEG regression.

    Each sample is one window of shape ``(num_channels, num_timesteps)`` with a single
    continuous target. Used for SEED-VIG vigilance (PERCLOS) estimation. Reported
    metrics are RMSE, normalised RMSE, R² and Pearson correlation.

    Args:
        hparams: Full experiment configuration.
        freeze_backbone: Train only the head, the patch embedding and the embeddings.
        layerwise_lr_decay: Per-block learning-rate decay; ``1.0`` disables it.
    """

    METRIC_NAMES = ("rmse", "nrmse", "r2", "pearson")

    def __init__(self, hparams, freeze_backbone: bool = False, layerwise_lr_decay: float = 1.0):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.model_head = hydra.utils.instantiate(self.hparams.model_head)
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)

        family = self.hparams.get("model_family", None)
        self.batch_requirements = (
            get_model_spec(family).batch_requirements if family else BatchRequirements()
        )

        self.freeze_backbone = freeze_backbone
        self.layerwise_lr_decay = layerwise_lr_decay
        self.patch_size = int(self.hparams.model.patch_size)
        self.strict_loading = False

        self.train_metrics = self._build_metrics()
        self.val_metrics = self._build_metrics()
        self.test_metrics = self._build_metrics()

        if self.freeze_backbone:
            self._apply_backbone_freeze()

    def _metrics(self, split: str) -> nn.ModuleDict:
        """Return the metric set for ``train``, ``val`` or ``test``."""
        return getattr(self, f"{split}_metrics")

    @staticmethod
    def _build_metrics() -> nn.ModuleDict:
        """Create one metric set for a single evaluation split."""
        return nn.ModuleDict({
            "rmse": MeanSquaredError(squared=False),
            "nrmse": NormalizedRootMeanSquaredError(normalization="std"),
            "r2": R2Score(),
            "pearson": PearsonCorrCoef(num_outputs=1),
        })

    def _apply_backbone_freeze(self) -> None:
        """Freeze encoder blocks while leaving tokenisation and embeddings trainable."""
        trainable = ("patch_embed", "channel_embedding", "positional_embedding")
        for name, param in self.model.named_parameters():
            param.requires_grad = any(prefix in name for prefix in trainable)

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:
        """Reshape raw waveforms into patches once the batch is on device."""
        x = batch["input"]
        if x.dim() != 3:
            raise ValueError(f"Expected input with 3 dimensions, got {x.dim()}")
        batch_size, channels, _ = x.shape
        batch["input"] = x.reshape(batch_size, channels, -1, self.patch_size)
        return batch

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        """Encode a batch and return scalar predictions."""
        encoded = self.model(
            x, channel_positions=channel_positions, directly_input_tokens=False, attn_mask=None
        )
        return self.model_head(encoded)

    def _shared_step(self, batch: Dict[str, Any], split: str) -> torch.Tensor:
        """Compute the loss and update the metrics for one batch."""
        require_batch_fields(batch, self.batch_requirements)
        predictions = self(batch["input"], batch["channel_coords"])
        targets = batch["label"].to(predictions.dtype).reshape(predictions.shape)
        batch["label"] = targets

        loss = self.criterion(predictions, batch)

        for metric in self._metrics(split).values():
            metric(predictions, targets)

        self.log(
            f"{split}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=targets.shape[0],
        )
        return loss

    def _log_epoch_metrics(self, split: str) -> None:
        """Log and reset every metric for one split at epoch end."""
        for name, metric in self._metrics(split).items():
            self.log(
                f"{split}_{name}", metric, prog_bar=True, logger=True, sync_dist=True,
                on_step=False, on_epoch=True,
            )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one training step."""
        if self.freeze_backbone:
            self.model.eval()
        return self._shared_step(as_signal_batch(batch), "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one validation step."""
        return self._shared_step(as_signal_batch(batch), "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one test step."""
        return self._shared_step(as_signal_batch(batch), "test")

    def on_train_epoch_end(self) -> None:
        """Log aggregated training metrics."""
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        """Log aggregated validation metrics."""
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        """Log aggregated test metrics."""
        self._log_epoch_metrics("test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Build parameter groups with layer-wise decay, then the optimiser and scheduler."""
        base_lr = float(self.hparams.optimizer.lr)
        base_weight_decay = float(getattr(self.hparams.optimizer, "weight_decay", 0.0))
        betas = tuple(getattr(self.hparams.optimizer, "betas", (0.9, 0.999)))
        head_lr = float(getattr(self.hparams.optimizer, "head_lr", base_lr))
        head_weight_decay = float(getattr(self.hparams.optimizer, "head_weight_decay", base_weight_decay))
        depth = int(self.hparams.model.depth)

        param_groups = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            lr = base_lr
            if self.layerwise_lr_decay != 1.0 and name.startswith("blocks."):
                block_idx = int(name.split(".")[1])
                lr = base_lr * (self.layerwise_lr_decay ** (depth - 1 - block_idx))
            weight_decay = 0.0 if self._excluded_from_weight_decay(name, param) else base_weight_decay
            param_groups.append({"params": [param], "lr": lr, "weight_decay": weight_decay})

        head_params = [p for p in self.model_head.parameters() if p.requires_grad]
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": head_weight_decay})

        optimizer_name = str(self.hparams.optimizer.optim).lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=base_weight_decay, betas=betas)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=base_lr, weight_decay=base_weight_decay, betas=betas)
        elif optimizer_name == "sgd":
            momentum = float(getattr(self.hparams.optimizer, "momentum", 0.9))
            optimizer = torch.optim.SGD(
                param_groups, lr=base_lr, weight_decay=base_weight_decay, momentum=momentum
            )
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

    @staticmethod
    def _excluded_from_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
        """Return True for parameters that should not be weight-decayed."""
        if name.endswith(".bias") or param.ndim == 1:
            return True
        lowered = name.lower()
        return any(
            key in lowered
            for key in ("norm", "positional_embedding", "channel_embedding", "mask_token", "pad_token")
        )

    def lr_scheduler_step(self, scheduler, metric) -> None:
        """Advance the timm-style scheduler once per optimiser step."""
        scheduler.step_update(num_updates=self.global_step)

    def load_from_checkpoint(
        self, checkpoint_path, map_location=None, hparams_file=None, strict=None, **kwargs
    ) -> "RegressionTask":
        """Load encoder weights from a checkpoint, skipping the head."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dicts = split_checkpoint_state_dict(checkpoint)
        current = self.model.state_dict()

        loaded, skipped = [], []
        for key, value in state_dicts["model"].items():
            if key in current and value.shape == current[key].shape:
                current[key] = value
                loaded.append(key)
            else:
                skipped.append(key)

        self.model.load_state_dict(current, strict=False)
        print(f"[load:model] loaded={len(loaded)} skipped={len(skipped)} total_target={len(current)}")
        if not loaded:
            print("[load:model] WARNING: no tensors were loaded from this checkpoint")

        if self.freeze_backbone:
            self._apply_backbone_freeze()
        return self
