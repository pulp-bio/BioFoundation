#*----------------------------------------------------------------------------*
#* Copyright (C) 2026 ETH Zurich, Switzerland                                 *
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
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


class TUEGDataset(Dataset):
    """LMDB-backed reader for the TUEG pre-training corpus.

    TUEG is large enough that per-sample pickling is a measurable overhead, so each
    value is stored as a fixed-size raw byte blob holding the padded waveform followed
    by the electrode coordinates. Sample keys live in a companion text file, one per
    line, which avoids a full LMDB scan at start-up.

    Recordings are padded to ``max_channels`` offline. The number of padded channels is
    recovered by finding channels whose coordinates are all zero, so pre-training can
    exclude them from masking and from attention.

    Args:
        lmdb_path: Path to the LMDB directory.
        keys_path: Path to the newline-separated key list; defaults to
            ``{lmdb_path without extension}_keys.txt``.
        max_channels: Channel count every record was padded to.
        sampling_freq: Sampling frequency in Hz.
        slice_duration: Window length in seconds.
        use_cache: Keep decoded samples in an in-process LRU cache.
        cache_size: Maximum number of cached samples.
    """

    def __init__(
        self,
        lmdb_path: str,
        keys_path: Optional[str] = None,
        max_channels: int = 64,
        sampling_freq: int = 200,
        slice_duration: int = 30,
        use_cache: bool = False,
        cache_size: int = 10000,
    ):
        self.lmdb_path = lmdb_path
        self.keys_path = keys_path or f"{os.path.splitext(lmdb_path)[0]}_keys.txt"
        self.max_channels = max_channels
        self.num_timesteps = sampling_freq * slice_duration

        self.waveform_bytes = max_channels * self.num_timesteps * np.dtype(np.float32).itemsize
        self.coords_bytes = max_channels * 2 * 3 * np.dtype(np.float32).itemsize
        self.record_bytes = self.waveform_bytes + self.coords_bytes

        if not os.path.isfile(self.keys_path):
            raise FileNotFoundError(f"Key list not found: {self.keys_path}")
        with open(self.keys_path, "r") as handle:
            self.keys: List[bytes] = [line.strip().encode("ascii") for line in handle if line.strip()]

        self.use_cache = use_cache
        self.cache_size = cache_size
        self.cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self.env = None

        print(f"[TUEGDataset] {os.path.basename(self.lmdb_path)}: {len(self.keys)} windows")

    def _ensure_env(self) -> None:
        """Open the LMDB environment lazily so it is not inherited across worker forks."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=True, map_async=True
            )

    def __len__(self) -> int:
        """Number of windows."""
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Decode one window and its electrode coordinates."""
        if self.use_cache and idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]

        self._ensure_env()
        key = self.keys[idx]
        with self.env.begin() as txn:
            blob = txn.get(key)

        if blob is None:
            raise KeyError(f"Key {key!r} is missing from {self.lmdb_path}")
        if len(blob) != self.record_bytes:
            raise ValueError(
                f"Record {key!r} has {len(blob)} bytes, expected {self.record_bytes}. "
                "Check that max_channels, sampling_freq and slice_duration match the "
                "values used during preprocessing."
            )

        waveform = np.frombuffer(blob[: self.waveform_bytes], dtype=np.float32)
        waveform = waveform.reshape(self.max_channels, self.num_timesteps).copy()
        coords = np.frombuffer(blob[self.waveform_bytes :], dtype=np.float32)
        coords = coords.reshape(self.max_channels, 2, 3).copy()

        channel_coords = torch.from_numpy(coords)
        num_padded = int((channel_coords.view(self.max_channels, -1).abs().sum(dim=1) == 0).sum())

        sample = {
            "input": torch.from_numpy(waveform),
            "channel_coords": channel_coords,
            "num_padded_channels": num_padded,
            "num_padded_timesteps": 0,
        }

        if self.use_cache:
            self.cache[idx] = sample
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        return sample

    def __del__(self):
        """Close the LMDB environment."""
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
