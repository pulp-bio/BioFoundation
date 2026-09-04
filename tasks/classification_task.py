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
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

from typing import Any, Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from biofoundation.core.batch import BatchRequirements, as_signal_batch, require_batch_fields
from biofoundation.core.checkpoints import SafetensorsCheckpointMixin
from biofoundation.model_registry import get_model_spec
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    CohenKappa,
    F1Score,
    Precision,
    Recall,
)


def split_checkpoint_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Separate a Lightning checkpoint into encoder and head state dicts.

    Args:
        checkpoint: Object returned by ``torch.load``; either a Lightning checkpoint
            with a ``state_dict`` entry or a bare state dict.

    Returns:
        Mapping with keys ``model`` and ``model_head``, each a prefix-stripped state dict.
    """
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        return {
            "model": {
                k[len("model."):]: v
                for k, v in state_dict.items()
                if k.startswith("model.") and not k.startswith("model_head.")
            },
            "model_head": {
                k[len("model_head."):]: v for k, v in state_dict.items() if k.startswith("model_head.")
            },
        }
    return {"model": dict(checkpoint), "model_head": {}}


def freeze_pretraining_only_parameters(encoder: nn.Module) -> list:
    """Disable gradients for encoder parameters that no fine-tuning forward pass uses.

    The mask and pad tokens exist for masked pre-training. A fine-tuning step never
    substitutes them, so they receive no gradient. Under DistributedDataParallel with
    find_unused_parameters=False the reducer waits for a gradient from every parameter
    it tracks, so leaving them trainable hangs the first training step of a multi-GPU
    run with no error message. Freezing them removes them from DDP's set entirely,
    which is cheaper than enabling find_unused_parameters and walking the graph on
    every step.

    LUNA disables its mask token for classification for the same reason; see
    models/LUNA.py.

    Returns:
        Names of the parameters that were frozen.
    """
    frozen = []
    for name in ("mask_token", "pad_token"):
        parameter = getattr(encoder, name, None)
        if parameter is not None and parameter.requires_grad:
            parameter.requires_grad = False
            frozen.append(name)
    return frozen


class ClassificationTask(SafetensorsCheckpointMixin, pl.LightningModule):
    """Fine-tuning task for EEG classification.

    Supports two input layouts:

    * **Window classification** (TUAB, CHB-MIT, Neonate, PhysioNet-MI, SHU-MI, STEW,
      Mumtaz, MentalArithmetic, SEED-V): each sample is one window of shape
      ``(num_channels, num_timesteps)`` with a single label. Use with
      :class:`~models.model_heads.mlp_classification_head.MlpClassificationHead`.
    * **Sequence classification** (ISRUC): each sample is a sequence of consecutive
      epochs of shape ``(sequence_length, num_channels, num_timesteps)`` with one label
      per epoch. The sequence axis is folded into the batch so the encoder processes
      epochs independently, and per-epoch labels are flattened to match. Use with
      :class:`~models.model_heads.sequence_classification_head.SequenceClassificationHead`,
      which restores the sequence axis internally.

    Args:
        hparams: Full experiment configuration.
        freeze_backbone: Train only the head, the patch embedding and the embeddings.
        layerwise_lr_decay: Per-block learning-rate decay; ``1.0`` disables it. Earlier
            blocks receive ``lr * decay ** (depth - 1 - block_idx)``.
        head_lr_multiplier: Multiplier applied to the head learning rate when the
            optimiser config does not set ``head_lr`` explicitly.
    """

    LABEL_METRICS = ("acc", "balanced_acc", "f1_score", "precision", "cohen_kappa")
    SCORE_METRICS = ("auroc", "aupr")

    def __init__(
        self,
        hparams,
        freeze_backbone: bool = False,
        layerwise_lr_decay: float = 1.0,
        head_lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.model_head = hydra.utils.instantiate(self.hparams.model_head)
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)

        self.num_classes = int(self.hparams.model_head.num_classes)
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        self.task_type = "binary" if self.num_classes == 2 else "multiclass"

        self.freeze_backbone = freeze_backbone
        self.layerwise_lr_decay = layerwise_lr_decay
        self.head_lr_multiplier = head_lr_multiplier
        self.patch_size = int(self.hparams.model.patch_size)
        self.softmax = nn.Softmax(dim=1)
        self.strict_loading = False

        family = self.hparams.get("model_family", None)
        self.batch_requirements = (
            get_model_spec(family).batch_requirements if family else BatchRequirements()
        )

        self.train_metrics = self._build_metrics()
        self.val_metrics = self._build_metrics()
        self.test_metrics = self._build_metrics()

        freeze_pretraining_only_parameters(self.model)

        if self.freeze_backbone:
            self._apply_backbone_freeze()

    def _metrics(self, split: str) -> nn.ModuleDict:
        """Return the metric set for ``train``, ``val`` or ``test``."""
        return getattr(self, f"{split}_metrics")

    def _build_metrics(self) -> nn.ModuleDict:
        """Create one metric set for a single evaluation split."""
        return nn.ModuleDict({
            "acc": Accuracy(task=self.task_type, num_classes=self.num_classes),
            "balanced_acc": Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
            "f1_score": F1Score(task="multiclass", num_classes=self.num_classes, average="weighted"),
            "precision": Precision(task="multiclass", num_classes=self.num_classes, average="micro"),
            "cohen_kappa": CohenKappa(task=self.task_type, num_classes=self.num_classes),
            "auroc": AUROC(task=self.task_type, num_classes=self.num_classes, average="macro"),
            "aupr": AveragePrecision(task=self.task_type, num_classes=self.num_classes, average="macro"),
        })

    def _apply_backbone_freeze(self) -> None:
        """Freeze encoder blocks while leaving tokenisation and embeddings trainable."""
        trainable = ("patch_embed", "channel_embedding", "positional_embedding")
        for name, param in self.model.named_parameters():
            param.requires_grad = any(prefix in name for prefix in trainable)

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:
        """Patch the waveforms and, for sequence data, fold the sequence into the batch."""
        x = batch["input"]

        if x.dim() == 3:
            batch_size, channels, _ = x.shape
            batch["input"] = x.reshape(batch_size, channels, -1, self.patch_size)
            return batch

        if x.dim() == 4:
            batch_size, sequence_length, channels, _ = x.shape
            batch["input"] = x.reshape(batch_size * sequence_length, channels, -1, self.patch_size)
            batch["label"] = batch["label"].reshape(batch_size * sequence_length)
            batch["channel_coords"] = (
                batch["channel_coords"]
                .unsqueeze(1)
                .expand(-1, sequence_length, -1, -1, -1)
                .reshape(batch_size * sequence_length, channels, 2, 3)
            )
            return batch

        raise ValueError(f"Expected input with 3 or 4 dimensions, got {x.dim()}")

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        """Encode a batch and return classification logits."""
        encoded = self.model(
            x, channel_positions=channel_positions, directly_input_tokens=False, attn_mask=None
        )
        return self.model_head(encoded)

    def _shared_step(self, batch: Dict[str, Any], split: str) -> torch.Tensor:
        """Compute the loss and update the metrics for one batch."""
        require_batch_fields(batch, self.batch_requirements)
        logits = self(batch["input"], batch["channel_coords"])
        labels = batch["label"]
        if labels.dim() == 2:
            labels = labels.argmax(1)
        batch["label"] = labels

        loss = self.criterion(logits, batch)

        predictions = torch.argmax(logits, dim=1)
        probabilities = self.softmax(logits)
        scores = probabilities[:, 1] if self.num_classes == 2 else probabilities

        metrics = self._metrics(split)
        for name in self.LABEL_METRICS:
            metrics[name](predictions, labels)
        for name in self.SCORE_METRICS:
            metrics[name](scores, labels)

        self.log(
            f"{split}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=labels.shape[0],
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
        """Build parameter groups with layer-wise decay, then the optimiser and scheduler.

        Encoder blocks receive geometrically decayed learning rates so that layers
        closer to the input change least. Biases, normalisation weights and embedding
        tables are excluded from weight decay. The head forms its own group.
        """
        base_lr = float(self.hparams.optimizer.lr)
        base_weight_decay = float(getattr(self.hparams.optimizer, "weight_decay", 0.0))
        betas = tuple(getattr(self.hparams.optimizer, "betas", (0.9, 0.999)))
        head_lr = float(getattr(self.hparams.optimizer, "head_lr", base_lr * self.head_lr_multiplier))
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
        self,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        include_head: bool = False,
        **kwargs,
    ) -> "ClassificationTask":
        """Load encoder weights, and optionally head weights, from a checkpoint.

        Tensors whose shapes do not match the current model are skipped rather than
        forced, so an encoder pre-trained at a different channel count can seed
        fine-tuning. The number of loaded and skipped tensors is printed so a silent
        no-op load is visible in the logs.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dicts = split_checkpoint_state_dict(checkpoint)

        self._partial_load(self.model, state_dicts["model"], "model")
        if include_head and state_dicts["model_head"]:
            self._partial_load(self.model_head, state_dicts["model_head"], "model_head")

        if self.freeze_backbone:
            self._apply_backbone_freeze()
        return self

    @staticmethod
    def _partial_load(module: nn.Module, incoming: Dict[str, torch.Tensor], label: str) -> None:
        """Load only the tensors whose names and shapes match ``module``."""
        current = module.state_dict()
        loaded, skipped, unexpected = [], [], []

        for key, value in incoming.items():
            if key not in current:
                unexpected.append(key)
            elif value.shape != current[key].shape:
                skipped.append(key)
            else:
                current[key] = value
                loaded.append(key)

        module.load_state_dict(current, strict=False)
        print(
            f"[load:{label}] loaded={len(loaded)} shape_mismatch={len(skipped)} "
            f"unexpected={len(unexpected)} total_target={len(current)}"
        )
        if skipped:
            print(f"[load:{label}] shape mismatch (first 10): {skipped[:10]}")
        if unexpected:
            print(f"[load:{label}] unexpected (first 10): {unexpected[:10]}")
        if not loaded:
            print(f"[load:{label}] WARNING: no tensors were loaded from this checkpoint")
