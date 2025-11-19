import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold

from datasets.emg_finetune_dataset import EMGDataset


class EMGCVDataModule(pl.LightningDataModule):
    """
    5-Fold Cross-Validation Data Module for EMG datasets.
    Splits the training data into 5 folds and uses one fold for validation.
    """

    def __init__(
        self,
        hdf5_file: str,
        cfg=None,
        name="",
        n_splits: int = 5,
        batch_size: int = 32,
        num_workers: int = 4,
        squeeze: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hdf5_file = hdf5_file
        self.cfg = cfg
        self.name = name
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.squeeze = squeeze
        self.current_fold = 0

        # Load full dataset to get size
        with h5py.File(hdf5_file, "r") as f:
            self.total_samples = f["data"].shape[0]

        # Generate fold indices
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.fold_indices = list(self.kf.split(np.arange(self.total_samples)))

    def set_fold(self, fold_idx: int):
        """Set which fold to use for validation"""
        if fold_idx >= self.n_splits:
            raise ValueError(f"Fold index {fold_idx} exceeds n_splits={self.n_splits}")
        self.current_fold = fold_idx

    def setup(self, stage: str = None):
        """Setup train/val splits for current fold"""
        train_idx, val_idx = self.fold_indices[self.current_fold]

        self.train_dataset = EMGDataset(
            self.hdf5_file,
            squeeze=self.squeeze,
            finetune=True,
            fold_indices=train_idx,
        )

        self.val_dataset = EMGDataset(
            self.hdf5_file,
            squeeze=self.squeeze,
            finetune=True,
            fold_indices=val_idx,
        )

        self.test_dataset =

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
