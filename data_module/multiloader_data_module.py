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
#* Author:  Berkay Döner                                                      *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    Dataset,
)
import torch
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm

class SequentialLoader:
    def __init__(self, dataloaders: DataLoader):
        self.dataloaders = dataloaders
    def __len__(self):
        return sum(len(d) for d in self.dataloaders)
    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader

class VaryingChannelsDataModule(pl.LightningDataModule):

    def __init__(
        self,
        datasets: [torch.utils.data.Dataset],
        cfg=None,
        name="",
        train_val_split_ratio=0.8,
        subset_ratio = 0.2,
        **kwargs
    ):
        super().__init__()

        # Concatenate multiple datasets for training
        datasets_list = [
            datasets[dataset_name]
            for dataset_name in datasets
            if datasets[dataset_name] is not None
        ]
        self.train, self.val = {}, {}
        self.subset_ratio = subset_ratio
        for dataset in datasets_list:
            # Load a subset of each dataset
            num_channels = dataset.num_channels
            if subset_ratio is not None:
                subset_size = int(subset_ratio * len(dataset))  # Adjust the fraction as needed
                indices = torch.randperm(len(dataset))[:subset_size]
                subset = torch.utils.data.Subset(dataset, indices)
                
                train_size = int(train_val_split_ratio * len(subset))
                val_size = len(subset) - train_size
                train, val = torch.utils.data.random_split(subset, [train_size, val_size])
            else:
                train_size = int(train_val_split_ratio * len(dataset))
                val_size = len(dataset) - train_size
                train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
            if num_channels not in self.train:
                self.train[num_channels] = []
                self.val[num_channels] = []
            self.train[num_channels].append(train)
            self.val[num_channels].append(val)
        
        self.train = [ConcatDataset(group) for group in self.train.values()]
        self.val = [ConcatDataset(group) for group in self.val.values()]
        self.name = name
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
       
    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self.train
            self.val_dataset = self.val
        elif stage == "validate":
            self.val_dataset = self.val
        elif stage == "test":
            self.test_dataset = self.val

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            raise ValueError(
                "Setup method must be called before accessing train_dataloader."
            )

        loaders = [
            DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                #shuffle=False,
                num_workers=self.cfg.num_workers,
                drop_last=True,
                pin_memory=True,
                sampler=DistributedSampler(ds, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=True)
            )
            for ds in self.train_dataset
        ]
        combined_loader = SequentialLoader(loaders)
        return combined_loader

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            raise ValueError(
                "Setup method must be called before accessing val_dataloader."
            )
        loaders = [
            DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                #shuffle=False,
                num_workers=self.cfg.num_workers,
                drop_last=True,
                pin_memory=True,
                sampler=DistributedSampler(ds, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=True)
            )
            for ds in self.val_dataset
        ]
        combined_loader = SequentialLoader(loaders)

        return combined_loader
