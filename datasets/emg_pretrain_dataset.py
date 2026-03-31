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
import fcntl
from multiprocessing import shared_memory

import h5py
from tqdm import tqdm
import numpy as np
import torch
from joblib import Parallel, delayed
import torch.nn.functional as F
from torch.utils.data import Dataset

class EMGPretrainDataset(Dataset):
    """Shared-memory optimized Dataset for large-scale EMG pretraining.

    This dataset loads HDF5 data into a shared RAM block (POSIX shared memory)
    to allow fast access across multiple worker processes without
    the serial overhead of HDF5 reads or the memory duplication of standard
    multiprocessing.

    Attributes:
        hdf5_file_path (str): Path to the single HDF5 source file.
        minmax (bool): Whether to apply min-max scaling to [-1, 1].
        pad_up_to_max_chans (Optional[int]): If set, zero-pads channels to this count.
        total_len (int): Total number of samples across all HDF5 groups.
        ram_data (np.ndarray): View into the shared memory block.
    """

    def __init__(
        self,
        hdf5_file: str,
        minmax: bool = True,
        pad_up_to_max_chans: int | None = None,
        n_jobs: int = 16
    ):
        """Initializes the shared memory dataset and loads data if needed.

        Args:
            hdf5_file (str): Path to the HDF5 file.
            minmax (bool): Enable scaling. Defaults to True.
            pad_up_to_max_chans (Optional[int]): Target channel count for padding.
            n_jobs (int): Number of parallel threads for the initial load.
        """
        super().__init__()
        self.minmax = minmax
        self.pad_up_to_max_chans = pad_up_to_max_chans

        self.rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

        # This class will be instantiated once per file (e.g., train.h5, val.h5)
        self.hdf5_file_path = hdf5_file
        if not os.path.exists(self.hdf5_file_path) or not self.hdf5_file_path.endswith(".h5"):
            raise ValueError(f"Expected hdf5_file to be a path to a single HDF5 file, but got {self.hdf5_file_path}")

        self.ram_data = None
        self.shm_block = None

        file_name = os.path.basename(self.hdf5_file_path)

        # Calculate total shape from all groups
        with h5py.File(self.hdf5_file_path, "r") as hf:
            group_keys = sorted(hf.keys())
            if not group_keys: raise ValueError(f"HDF5 file {file_name} contains no data groups.")

            self.group_offsets = [0]
            total_samples = 0
            for key in group_keys:
                num_in_group = hf[key]['X'].shape[0]
                total_samples += num_in_group
                self.group_offsets.append(total_samples)

            # Get other dimensions from the first group
            _, C, T = hf[group_keys[0]]['X'].shape
            final_shape = (total_samples, C, T)

        self.total_len = total_samples

        # Allocate shared memory and load in parallel
        clean_name = f"{os.path.splitext(file_name)[0]}_{final_shape[0]}"
        shm_name = f"emg_shm_{clean_name}"
        target_dtype = np.float16 # cast to fp16 to fit in RAM
        num_bytes = int(np.prod(final_shape)) * np.dtype(target_dtype).itemsize

        lock_path = f"/tmp/{shm_name}.lock"
        ready_path = f"/tmp/{shm_name}.ready"
        file_mtime = os.path.getmtime(self.hdf5_file_path)
        ready_token = f"{os.path.abspath(self.hdf5_file_path)}|{file_mtime}|{num_bytes}"

        with open(lock_path, "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)

            shm = None
            shm_needs_load = True

            if os.path.exists(ready_path):
                try:
                    with open(ready_path, "r") as rf:
                        token = rf.read().strip()
                    if token == ready_token:
                        existing = shared_memory.SharedMemory(name=shm_name)
                        if existing.size == num_bytes:
                            shm = existing
                            shm_needs_load = False
                        else:
                            existing.close()
                            try:
                                existing.unlink()
                            except FileNotFoundError:
                                pass
                except Exception:
                    shm_needs_load = True

            if shm_needs_load:
                try:
                    stale = shared_memory.SharedMemory(name=shm_name)
                    stale.close()
                    try:
                        stale.unlink()
                    except FileNotFoundError:
                        pass
                except FileNotFoundError:
                    pass

                print(f"[PID {os.getpid()}] Allocating {num_bytes / 1e9:.2f} GB of Shared RAM for {file_name}...")
                shm = shared_memory.SharedMemory(create=True, name=shm_name, size=num_bytes)
                shm_arr = np.ndarray(final_shape, dtype=target_dtype, buffer=shm.buf)

                def load_group(group_idx):
                    key = group_keys[group_idx]
                    start_offset = self.group_offsets[group_idx]
                    end_offset = self.group_offsets[group_idx + 1]
                    with h5py.File(self.hdf5_file_path, "r") as local_hf:
                        data_chunk = local_hf[key]['X'][:].astype(target_dtype)
                        shm_arr[start_offset:end_offset] = data_chunk

                print(f"[PID {os.getpid()}] Parallel loading groups from {file_name} using {n_jobs} cores...")
                Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(load_group)(i) for i in tqdm(range(len(group_keys)), desc=f"Loading {file_name}")
                )
                with open(ready_path, "w") as wf:
                    wf.write(ready_token)
                print(f"[PID {os.getpid()}] Finished loading {file_name}!")

        self.shm_block = shm
        self.ram_data = np.ndarray(final_shape, dtype=target_dtype, buffer=shm.buf)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return self.total_len

    def _minmax_scale(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Scales EMG signal to [-1, 1] range."""
        maxv = x.amax(dim=-1, keepdim=True)
        minv = x.amin(dim=-1, keepdim=True)
        x = (x - minv) / (maxv - minv + eps)
        return (x - 0.5) * 2

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves a single sample as a float32 tensor.

        Args:
            idx (int): Global index across all groups.

        Returns:
            torch.Tensor: Normalized and padded EMG tensor of shape (C, T).
        """
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_len}")

        # Direct, instant access from the single RAM array
        X = torch.tensor(self.ram_data[idx], dtype=torch.float32).contiguous()

        if self.minmax: X = self._minmax_scale(X)
        if self.pad_up_to_max_chans is not None:
            C = X.shape[0]
            to_pad = self.pad_up_to_max_chans - C
            if to_pad > 0: X = F.pad(X.T, (0, to_pad)).T
        return X


    def __del__(self):
        if hasattr(self, 'shm_block') and self.shm_block is not None:
            try: self.shm_block.close()
            except Exception: pass