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
# * Author:  Anna Tegon                                                        *
# * Author:  Thorir Mar Ingolfsson                                             *
# *----------------------------------------------------------------------------*
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class FinetuneDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for fine-tuning, managing training, validation,
    test, and prediction datasets, providing the corresponding DataLoaders.

    Args:
        train (Dataset): Dataset for training.
        val (Dataset): Dataset for validation.
        test (Dataset, optional): Dataset for testing.
        cfg (Config): Configuration object containing batch_size and num_workers.
        name (str, optional): Identifier name for the DataModule.
        **kwargs: Additional optional arguments.
    """

    def __init__(self, train, val, test=None, cfg=None, name="", **kwargs):
        super().__init__()
        # Store datasets and config
        self.train = train
        self.val = val
        self.test = test
        self.name = name
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets based on the current stage.

        Args:
            stage (str, optional): One of 'fit', 'validate', 'test', 'predict', or None.
        """
        if stage == "fit" or stage is None:
            # Setup train and validation datasets for fitting
            self.train_dataset = self.train
            self.val_dataset = self.val
        elif stage == "validate":
            # Setup validation dataset for validation step
            self.val_dataset = self.val
        elif stage == "test":
            # Setup test dataset for testing step
            self.test_dataset = self.test
        elif stage == "predict":
            # Setup validation dataset for prediction step
            self.val_dataset = self.val

    def train_dataloader(self):
        """
        Returns the DataLoader for training with shuffling enabled.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,  # MFASULO: Changed to False to include all samples
            pin_memory=True,
            persistent_workers=True,  # MFASULO: Enable persistent workers for efficiency
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for validation with shuffling disabled.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Returns the DataLoader for testing with shuffling disabled.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        """
        Returns the DataLoader for prediction with shuffling disabled.
        Uses the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=True,
        )
