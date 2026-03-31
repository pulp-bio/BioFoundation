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
        tuab_loader: [torch.utils.data.Dataset] = None,
        tuar_loader: [torch.utils.data.Dataset] = None,
        max_samples=2000,
        cfg=None,
        name="",
        train_val_split_ratio=0.8,
        subset_ratio=None,
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


        self.tuab_loader_cfg = tuab_loader
        self.tuar_loader_cfg = tuar_loader
        self.max_samples = max_samples # Max samples for balanced subset for t-SNE

    def _init_test_loader(self, loader_cfg: dict, num_classes: int, dataset_name: str, num_workers: int = 0):
        print(f"Setting up {dataset_name} test loader...")
        try:
            train_ds = loader_cfg.get("train")
            val_ds = loader_cfg.get("val")
            test_ds = loader_cfg.get("test")

            # TUAB has only "test", TUAR has train+val+test
            datasets_to_concat = [ds for ds in [train_ds, val_ds, test_ds] if ds is not None]

            full_dataset = ConcatDataset(datasets_to_concat)

            base_loader = DataLoader(
                full_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=self.cfg.num_workers
            )
            print(f"{dataset_name} base dataloader created.")

            # Balanced subset for t-SNE
            balanced_subset = get_balanced_subset(
                base_loader,
                num_classes=num_classes,
                total_samples=self.max_samples
            )

            subset_loader = DataLoader(
                balanced_subset,
                batch_size=64,
                shuffle=False,
                drop_last=False,
                num_workers=self.cfg.num_workers
            )

            print(f"{dataset_name} t-SNE loader length:", len(subset_loader))
            return subset_loader

        except Exception as e:
            print(f"Failed to initialize {dataset_name} test loader:", e)
            return None

    def _build_tsne_loaders(self):
        if self.tuar_loader_cfg is not None:
            self.tuar_test_loader = self._init_test_loader(
                self.tuar_loader_cfg,
                num_classes=6,
                dataset_name="TUAR",
                num_workers=0 
            )

        if self.tuab_loader_cfg is not None:
            self.tuab_test_loader = self._init_test_loader(
                self.tuab_loader_cfg,
                num_classes=2,
                dataset_name="TUAB",
                num_workers=0
        )

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self.train
            self.val_dataset = self.val
            # --------- t-SNE loader init (rank 0 only) ---------
            if self.trainer.global_rank == 0:
                self._build_tsne_loaders()
            else:
                self.tuab_test_loader = None
                self.tuar_test_loader = None

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

    def test_tuab_dataloader(self):
        """
        Returns the DataLoader for testing with shuffling disabled.
        """
        print("Returning DataLoader for TUAB testing...")
        return self.tuab_test_loader

    def test_tuar_dataloader(self):
        """
        Returns the DataLoader for testing with shuffling disabled.
        """
        print("Returning DataLoader for TUAR testing...")
        return self.tuar_test_loader

def get_balanced_subset(dataloader, total_samples, num_classes):

    samples_per_class = total_samples // num_classes
    collected = {c: [] for c in range(num_classes)}

    for batch in dataloader:
        inputs = batch["input"]
        labels = batch["label"]
        ch_locs = batch["channel_locations"]
        ch_names = batch["channel_names"]

        for i in range(len(labels)):
            label = int(labels[i].item())

            if len(collected[label]) < samples_per_class:
                sample_dict = {
                    "input": inputs[i].cpu(),
                    "label": labels[i].cpu(),
                    "channel_locations": ch_locs[i].cpu(),
                    "channel_names": ch_names[i].cpu(),
                }
                collected[label].append(sample_dict)

        # Stop when all classes are filled
        if all(len(collected[c]) >= samples_per_class for c in range(num_classes)):
            break

    # Flatten list
    balanced_subset = [item for c in collected for item in collected[c]]

    print(f"Balanced subset created: {len(balanced_subset)} samples")
    for c in collected:
        print(f" - Class {c}: {len(collected[c])}")

    return balanced_subset