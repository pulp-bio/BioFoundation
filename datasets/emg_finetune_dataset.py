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

    def __init__(
        self,
        hdf5_file: str,
        unsqueeze: bool = False,
        finetune: bool = True,
        cache_size: int = 1500,
        use_cache: bool = True,
        regression: bool = False,
    ):
        self.hdf5_file = hdf5_file
        self.unsqueeze = unsqueeze
        self.cache_size = cache_size
        self.finetune = finetune
        self.use_cache = use_cache
        self.regression = regression

        self.data = None
        self.X_ds = None
        self.Y_ds = None

        # Open once to get length, then close immediately
        with h5py.File(self.hdf5_file, "r") as f:
            self.num_samples = f["data"].shape[0]

        if self.use_cache:
            self.cache: dict[
                int, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            ] = {}
            self.cache_queue = deque()

    def _open_file(self) -> None:
        # 'rdcc_nbytes' to increase the raw data chunk cache size
        self.data = h5py.File(self.hdf5_file, "r", rdcc_nbytes=1024 * 1024 * 4)
        if self.data is not None:
            self.X_ds = self.data["data"]
            self.Y_ds = self.data["label"]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):
        # Check Cache
        if self.use_cache and index in self.cache:
            return self._process_data(self.cache[index])

        # Open file (Lazy Loading for Multiprocessing)
        if self.data is None:
            self._open_file()

        # Read Data, HDF5 slicing returns numpy array
        X_np = self.X_ds[index]
        X = torch.from_numpy(X_np).float()

        if self.finetune:
            Y_np = self.Y_ds[index]
            if self.regression:
                Y = torch.from_numpy(Y_np).float()
            else:
                # Ensure scalar is converted properly
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

    def _process_data(self, data_item):
        """Helper to handle squeezing/returning uniformly."""
        if self.finetune:
            X, Y = data_item
        else:
            X = data_item
            Y = None

        if self.unsqueeze:
            X = X.unsqueeze(0)

        if self.finetune:
            return X, Y
        else:
            return X

    def __del__(self):
        if self.data is not None:
            self.data.close()
