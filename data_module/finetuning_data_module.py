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

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class FinetuningDataModule(pl.LightningDataModule):
    """Data module for fine-tuning on a dataset with fixed train, validation and test splits.

    The three splits are supplied as already-constructed datasets, so the split is
    defined entirely by the preprocessing step and is identical for every run and every
    model. No resampling, subject-level cross-validation or automatic splitting happens
    here.

    Args:
        train: Training dataset.
        val: Validation dataset, used for model selection and early stopping.
        test: Test dataset, evaluated once at the end of the run.
        cfg: Holds ``batch_size`` and ``num_workers``.
        name: Label for logging.
    """

    def __init__(
        self,
        train: Dataset,
        val: Dataset,
        test: Optional[Dataset] = None,
        cfg=None,
        name: str = "",
        **kwargs,
    ):
        super().__init__()
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.cfg = cfg
        self.name = name

    def _loader(self, dataset: Dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        """Build a loader with the shared worker and pinning settings."""
        if dataset is None:
            raise ValueError("Requested a dataloader for a split that was not configured")
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled loader over the training split, dropping the last partial batch."""
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Deterministic loader over the full validation split."""
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Deterministic loader over the full test split."""
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)
