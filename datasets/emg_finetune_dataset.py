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
from typing import Tuple, Union

import h5py
import numpy as np
import torch


class EMGDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading EMG data from HDF5 files.

    Attributes:
        hdf5_file (str): Path to the HDF5 source file.
        finetune (bool): If True, returns (data, label). If False, returns data only.
        regression (bool): If True, labels are treated as floats. Else, longs.
    """
    def __init__(
        self,
        hdf5_file: str,
        finetune: bool = True,
        regression: bool = False,
        verbose: bool = False,
    ):
        self.hdf5_file = hdf5_file
        self.finetune = finetune
        self.regression = regression

        with h5py.File(self.hdf5_file, "r") as f:
            X_np = f["data"][:]
            Y_np = f["label"][:] if self.finetune else None

        self.X_tensor = torch.from_numpy(X_np).float().contiguous()
        if self.finetune:
            if self.regression:
                self.Y_tensor = torch.from_numpy(Y_np).float().contiguous()
            else:
                self.Y_tensor = torch.from_numpy(Y_np).long().contiguous()
                if verbose:
                    uniq, cnt = np.unique(Y_np, return_counts=True)
                    print(
                        f"[EMGDataset] {self.hdf5_file}: label min={Y_np.min()}, max={Y_np.max()}, classes={len(uniq)}"
                    )
                    print(f"[EMGDataset] {self.hdf5_file}: class hist={dict(zip(uniq.tolist(), cnt.tolist()))}")

        self.num_samples = self.X_tensor.shape[0] # [N, C, T]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieves the EMG data and optional label at the specified index."""

        X = self.X_tensor[index]

        if self.finetune:
            Y = self.Y_tensor[index]
            return X, Y

        return X
