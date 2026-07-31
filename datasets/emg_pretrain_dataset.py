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
import os
from bisect import bisect_right

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class EMGPretrainDataset(Dataset):
    """Lazy HDF5 Dataset for large-scale EMG pretraining.

    Only HDF5 metadata is read during initialization. Each DataLoader worker
    opens its own persistent HDF5 handle on first access, avoiding full-dataset
    RAM preloading while allowing the VAST filesystem to serve reads directly.

    Attributes:
        hdf5_file_path (str): Path to the single HDF5 source file.
        pad_up_to_max_chans (Optional[int]): If set, zero-pads channels to this count.
        total_len (int): Total number of samples across all HDF5 groups.
        group_keys (list[str]): HDF5 groups containing signal samples.
    """

    def __init__(
        self,
        hdf5_file: str,
        pad_up_to_max_chans: int | None = None,
    ):
        """Initializes the shared memory dataset and loads data if needed.

        Args:
            hdf5_file (str): Path to the HDF5 file.
            pad_up_to_max_chans (Optional[int]): Target channel count for padding.
        """
        super().__init__()
        self.pad_up_to_max_chans = pad_up_to_max_chans

        self.hdf5_file_path = hdf5_file
        if not os.path.exists(self.hdf5_file_path) or not self.hdf5_file_path.endswith(".h5"):
            raise ValueError(f"Expected hdf5_file to be a path to a single HDF5 file, but got {self.hdf5_file_path}")

        self._h5_file = None
        self._h5_pid = None

        # Read only metadata. Keep group offsets for O(log n) index lookup.
        with h5py.File(self.hdf5_file_path, "r") as hf:
            group_keys = sorted(hf.keys())
            if not group_keys:
                raise ValueError(f"HDF5 file {self.hdf5_file_path} contains no data groups.")

            self.group_keys = group_keys
            self.group_offsets = [0]
            total_samples = 0
            for key in group_keys:
                if "X" not in hf[key]:
                    raise ValueError(f"Group {key} in {self.hdf5_file_path} has no X dataset.")
                num_in_group = hf[key]["X"].shape[0]
                total_samples += num_in_group
                self.group_offsets.append(total_samples)

        self.total_len = total_samples

    def _get_h5_file(self):
        """Return a handle owned by the current worker process."""
        pid = os.getpid()
        if self._h5_file is None or self._h5_pid != pid:
            if self._h5_file is not None:
                self._h5_file.close()
            self._h5_file = h5py.File(self.hdf5_file_path, "r")
            self._h5_pid = pid
        return self._h5_file

    def __getstate__(self):
        """Prevent an open HDF5 handle from being serialized to workers."""
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_h5_pid"] = None
        return state

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return self.total_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves a single sample as a float32 tensor.

        Args:
            idx (int): Global index across all groups.

        Returns:
            torch.Tensor: Padded EMG tensor of shape (C, T).
        """
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_len}")

        group_idx = bisect_right(self.group_offsets, idx) - 1
        local_idx = idx - self.group_offsets[group_idx]
        h5_file = self._get_h5_file()
        X_np = h5_file[self.group_keys[group_idx]]["X"][local_idx]
        X = torch.from_numpy(np.asarray(X_np)).float().contiguous()

        if self.pad_up_to_max_chans is not None:
            C = X.shape[0]
            to_pad = self.pad_up_to_max_chans - C
            if to_pad > 0: X = F.pad(X.T, (0, to_pad)).T
        return X


    def __del__(self):
        if hasattr(self, "_h5_file") and self._h5_file is not None:
            try: self._h5_file.close()
            except Exception: pass
