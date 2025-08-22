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
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    Dataset,
)
import torch

class PretrainDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for pretraining that manages multiple datasets,
    splitting them into training and validation subsets, and provides train/val
    dataloaders with configurable batch size and worker count.

    Args:
        datasets (dict of Dataset): Dictionary of datasets to be concatenated and split.
        test (Dataset, optional): Optional test dataset.
        cfg (Config): Configuration object containing batch_size and num_workers.
        name (str, optional): Name identifier for this data module.
        train_val_split_ratio (float, optional): Ratio for train/validation split (default=0.8).
        **kwargs: Additional arguments.

    """

    def __init__(
        self,
        datasets: [torch.utils.data.Dataset], 
        test=None,
        cfg=None,
        name="",
        train_val_split_ratio=0.8,
        **kwargs
    ):
        super().__init__()

        # Filter out None datasets and collect available datasets to concatenate
        datasets_list = [
            datasets[dataset_name]
            for dataset_name in datasets
            if datasets[dataset_name] is not None
        ]

        print("datasets list:", datasets_list)

        # Initialize lists to hold split datasets
        self.train, self.val = [], []

        # For each dataset, split into training and validation sets according to the ratio
        for dataset in datasets_list:
            train_size = int(train_val_split_ratio * len(dataset))
            val_size = len(dataset) - train_size
            train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
            self.train.append(train)
            self.val.append(val)

        # Concatenate all training splits into a single training dataset
        self.train = ConcatDataset(self.train)
        # Concatenate all validation splits into a single validation dataset
        self.val = ConcatDataset(self.val)

        self.test = test
        self.name = name
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        
        print(len(self.train), len(self.val))

    def setup(self, stage: Optional[str] = None):
        """
        Prepare datasets for different stages: 'fit', 'validate', or 'test'.

        Args:
            stage (str, optional): Stage name. Options: 'fit', 'validate', 'test', or None.
        """

        if stage == "fit" or stage is None:
            # Assign train and validation datasets for training phase
            self.train_dataset = self.train
            self.val_dataset = self.val
        elif stage == "validate":
            # Assign validation dataset for validation phase
            self.val_dataset = self.val
        elif stage == "test":
            # Assign validation dataset for test phase (could be adjusted if test set available)
            self.test_dataset = self.val

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Raises:
            ValueError: If setup() hasn't been called before this.

        Returns:
            DataLoader: DataLoader with shuffling enabled for training.
        """
        if not hasattr(self, "train_dataset"):
            raise ValueError(
                "Setup method must be called before accessing train_dataloader."
            )
        return DataLoader(
            self.train_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Raises:
            ValueError: If setup() hasn't been called before this.

        Returns:
            DataLoader: DataLoader without shuffling for validation.
        """
        if not hasattr(self, "val_dataset"):
            raise ValueError(
                "Setup method must be called before accessing val_dataloader."
            )
        return DataLoader(
            self.val_dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=True
        )
