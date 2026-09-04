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

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class PretrainingDataModule(pl.LightningDataModule):
    """Data module for self-supervised pre-training on a union of EEG corpora.

    The configured corpora are concatenated and split once into a training and a
    validation portion with a seeded generator, so the split is identical across
    processes in distributed training and reproducible across runs. The split is over
    windows rather than recordings, so the validation loss measures reconstruction
    quality in distribution rather than generalisation to unseen subjects.

    All corpora must share a window length in timesteps, otherwise batches drawn
    across corpora cannot be collated.

    Args:
        train: Mapping of corpus name to dataset; ``None`` entries are skipped.
        cfg: Holds ``batch_size`` and ``num_workers``.
        val_ratio: Fraction of windows held out for validation.
        seed: Seed for the train/validation split.
        name: Label for logging.
    """

    def __init__(
        self,
        train: Dict[str, Optional[Dataset]],
        cfg=None,
        val_ratio: float = 0.2,
        seed: int = 42,
        name: str = "",
        **kwargs,
    ):
        super().__init__()
        datasets = [dataset for dataset in train.values() if dataset is not None]
        if not datasets:
            raise ValueError("No pre-training datasets were configured")

        combined = ConcatDataset(datasets)
        generator = torch.Generator().manual_seed(seed)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            combined, [1.0 - val_ratio, val_ratio], generator=generator
        )
        print(
            f"[PretrainingDataModule] corpora={len(datasets)} windows={len(combined)} "
            f"train={len(self.train_dataset)} val={len(self.val_dataset)}"
        )

        self.cfg = cfg
        self.name = name

    def train_dataloader(self) -> DataLoader:
        """Shuffled loader over the training split."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Deterministic loader over the validation split."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )
