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
# * Author:  Matteo Fasulo                                                     *
# *----------------------------------------------------------------------------*
from collections import deque
from typing import Tuple, Union

import h5py
import torch


class EMGDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading EMG data from HDF5 files.

    This dataset supports both classification and regression tasks. It provides
    two loading modes: pre-loading the entire dataset into RAM for speed, or
    lazy-loading from disk with an LRU cache for large datasets that don't fit
    in memory.

    Attributes:
        hdf5_file (str): Path to the HDF5 source file.
        finetune (bool): If True, returns (data, label). If False, returns data only.
        unsqueeze (bool): If True, adds a channel dimension to the input.
        cache_size (int): Max number of samples to keep in the LRU cache.
        use_cache (bool): Enables LRU caching for lazy loading.
        regression (bool): If True, labels are treated as floats. Else, longs.
        preload_in_memory (bool): If True, loads the full HDF5 content into RAM on init.
        num_samples (int): Total number of samples in the dataset.
    """
    def __init__(
        self,
        hdf5_file: str,
        finetune: bool = True,
        unsqueeze: bool = False,
        cache_size: int = 1500,
        use_cache: bool = False,
        regression: bool = False,
        preload_in_memory: bool = True,
    ):
        self.hdf5_file = hdf5_file
        self.finetune = finetune
        self.unsqueeze = unsqueeze
        self.cache_size = cache_size
        self.use_cache = use_cache
        self.regression = regression
        self.preload_in_memory = preload_in_memory

        self.data = None
        self.X_ds = None
        self.Y_ds = None
        self.X_tensor = None
        self.Y_tensor = None

        if self.preload_in_memory:
            with h5py.File(self.hdf5_file, "r") as f:
                X_np = f["data"][:]
                Y_np = f["label"][:] if self.finetune else None

            self.X_tensor = torch.from_numpy(X_np).float().contiguous()
            if self.finetune:
                if self.regression:
                    self.Y_tensor = torch.from_numpy(Y_np).float().contiguous()
                else:
                    self.Y_tensor = torch.from_numpy(Y_np).long().contiguous()

            self.num_samples = self.X_tensor.shape[0]
        else:
            self.data = h5py.File(self.hdf5_file, "r")
            self.X_ds = self.data["data"]
            self.Y_ds = self.data["label"] if self.finetune else None
            self.num_samples = self.X_ds.shape[0]

        if self.use_cache:
            self.cache: dict[int, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = {}
            self.cache_queue = deque()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieves the EMG data and optional label at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If finetune=True: (EMG tensor, Label tensor)
                - If finetune=False: EMG tensor
        """
        # Check Cache
        if self.use_cache and index in self.cache:
            return self._process_data(self.cache[index])

        if self.preload_in_memory:
            X = self.X_tensor[index]
            if self.finetune:
                Y = self.Y_tensor[index]
                data_item = (X, Y)
            else:
                data_item = X
        else:
            # Read Data, HDF5 slicing returns numpy array
            X_np = self.X_ds[index]
            X = torch.from_numpy(X_np).float()

            if self.finetune:
                Y_np = self.Y_ds[index]
                if self.regression:
                    Y = torch.from_numpy(Y_np).float()
                else:
                    Y = torch.tensor(Y_np, dtype=torch.long)
                data_item = (X, Y)
            else:
                data_item = X

        # Update Cache
        if self.use_cache:
            # If cache is full, remove oldest item from dict AND queue
            if len(self.cache) >= self.cache_size:
                oldest_index = self.cache_queue.popleft()
                del self.cache[oldest_index]

            self.cache[index] = data_item
            self.cache_queue.append(index)

        return self._process_data(data_item)

    def _process_data(self, data_item: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Applies final transformations like unsqueezing.

        Args:
            data_item (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): Raw data/label tuple.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Processed data.
        """
        if self.finetune:
            X, Y = data_item
            if self.unsqueeze:
                X = X.unsqueeze(0)
            return X, Y

        X = data_item
        if self.unsqueeze:
            X = X.unsqueeze(0)
        return X


    def __del__(self):
        if self.data is not None:
            self.data.close()
