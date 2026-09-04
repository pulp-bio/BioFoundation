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

import glob
import json
import os
import pickle
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
import torch.nn.functional as F


def resolve_lmdb_path(path: str, split: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Resolve a dataset directory and split name to an LMDB file and its index file.

    Args:
        path: Either a directory containing ``{split}.lmdb`` files or a path ending in
            ``.lmdb``.
        split: One of ``train``, ``val``, ``test``, or ``None`` to auto-select.

    Returns:
        Tuple of the LMDB path and the matching ``{stem}.meta.json`` path, or ``None``
        when no index file exists. A split-specific index is never substituted with a
        pooled one, so a missing ``train.meta.json`` falls back to scanning
        ``train.lmdb`` rather than silently indexing it with pooled keys.
    """
    if path.endswith(".lmdb"):
        stem = os.path.splitext(os.path.basename(path))[0]
        meta = os.path.join(os.path.dirname(path), f"{stem}.meta.json")
        return path, meta if os.path.isfile(meta) else None

    if not os.path.isdir(path):
        raise ValueError(f"Dataset path is neither a directory nor an .lmdb file: {path}")

    available = {
        os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(path, "*.lmdb"))
    }
    if not available:
        raise FileNotFoundError(f"No .lmdb files found in {path}")

    if split in {"train", "val", "test"}:
        if split not in available:
            raise FileNotFoundError(
                f"Expected {split}.lmdb in {path}; found {sorted(available)}. "
                "Re-run the preprocessing script for this dataset to build the splits."
            )
        stem = split
    elif len(available) == 1:
        stem = next(iter(available))
    else:
        raise ValueError(f"Multiple LMDBs in {path}; set split to one of {sorted(available)}")

    meta = os.path.join(path, f"{stem}.meta.json")
    return available[stem], meta if os.path.isfile(meta) else None


class LMDBDataset(torch.utils.data.Dataset):
    """LMDB-backed EEG dataset for windowed and sequence samples.

    Each LMDB value is a pickled dictionary with keys ``eeg``, ``channel_coords`` and
    optionally ``label`` and ``subject_id``.

    Two sample layouts are supported:

    * ``window``: ``eeg`` has shape ``(num_channels, num_timesteps)`` with one scalar label.
    * ``sequence``: ``eeg`` has shape ``(sequence_length, num_channels, num_timesteps)``
      with one label per element. Every entry in a sequence dataset must share the same
      ``sequence_length``, otherwise batches cannot be collated.

    Args:
        path: Dataset directory or ``.lmdb`` file.
        dataset_kind: ``window`` or ``sequence``.
        split: ``train``, ``val`` or ``test``.
        apply_minmax: Scale each channel to ``[-1, 1]`` independently.
        apply_zero_padding: Zero-pad channels and timesteps up to ``max_channels`` and
            ``max_timesteps``, reporting how much padding was added.
        max_channels: Channel count to pad to; required if ``apply_zero_padding``.
        max_timesteps: Timestep count to pad to; optional.
        label_mode: ``classification``, ``regression``, or ``auto`` to infer from dtype.
        use_cache: Keep decoded samples in an in-process LRU cache.
        cache_size: Maximum number of cached samples.
    """

    def __init__(
        self,
        path: str,
        dataset_kind: str = "window",
        split: Optional[str] = None,
        apply_minmax: bool = True,
        apply_zero_padding: bool = False,
        max_channels: Optional[int] = None,
        max_timesteps: Optional[int] = None,
        label_mode: str = "auto",
        use_cache: bool = False,
        cache_size: int = 1000,
    ):
        if dataset_kind not in {"window", "sequence"}:
            raise ValueError(f"dataset_kind must be 'window' or 'sequence', got '{dataset_kind}'")
        if label_mode not in {"auto", "classification", "regression"}:
            raise ValueError(f"Invalid label_mode '{label_mode}'")

        self.dataset_kind = dataset_kind
        self.apply_minmax = apply_minmax
        self.apply_zero_padding = apply_zero_padding
        self.max_channels = max_channels
        self.max_timesteps = max_timesteps
        self.label_mode = label_mode
        self.use_cache = use_cache
        self.cache_size = cache_size

        self.lmdb_path, self.meta_path = resolve_lmdb_path(path, split)
        self.env = None
        self.keys = self._load_keys()
        self.cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()

        print(f"[LMDBDataset] {os.path.basename(self.lmdb_path)}: {len(self.keys)} entries")

    def _ensure_env(self) -> None:
        """Open the LMDB environment lazily, on first read.

        Construction deliberately leaves ``env`` unset. LMDB refuses to open the same
        path twice in one process, and ``run_train.py`` re-instantiates the data module
        before the rank-zero test pass while the original may still be referenced. It
        also avoids handing an open handle to forked dataloader workers. This mirrors
        how :class:`~datasets.tueg_dataset.TUEGDataset` manages its environment.
        """
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=True,
                max_readers=64, map_async=True,
            )

    def _load_keys(self) -> List[bytes]:
        """Read the sample keys from the index file, or scan the LMDB if there is none."""
        if self.meta_path is not None:
            with open(self.meta_path, "r") as handle:
                meta = json.load(handle)
            keys = meta.get("keys", [])
            if not keys:
                raise ValueError(f"Index file {self.meta_path} contains no keys")
            return [key.encode() if isinstance(key, str) else key for key in keys]

        # No index file: scan once through a temporary environment, then close it so
        # construction still leaves no handle open.
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True)
        try:
            with env.begin(buffers=True) as txn:
                return [bytes(key) for key in txn.cursor().iternext(keys=True, values=False)]
        finally:
            env.close()

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.keys)

    def _min_max_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each channel independently to ``[-1, 1]``."""
        flat = x.reshape(-1, x.shape[-1])
        minimum = flat.min(dim=1, keepdim=True).values
        maximum = flat.max(dim=1, keepdim=True).values
        scaled = (flat - minimum) / (maximum - minimum + 1e-6)
        return ((scaled - 0.5) * 2).reshape(x.shape)

    def _pad(self, x: torch.Tensor, channel_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Zero-pad the channel and time axes, returning the amount of padding added."""
        channels, timesteps = x.shape[-2], x.shape[-1]
        pad_time = max(0, (self.max_timesteps or timesteps) - timesteps)
        pad_channels = max(0, (self.max_channels or channels) - channels)
        padded = F.pad(x, (0, pad_time, 0, pad_channels), value=0.0)

        coord_padding = max(0, (self.max_channels or channel_coords.shape[0]) - channel_coords.shape[0])
        padded_coords = F.pad(channel_coords, (0, 0, 0, 0, 0, coord_padding), value=0.0)
        return padded, padded_coords, pad_channels, pad_time

    def _is_regression(self, label: Any) -> bool:
        """Decide whether a stored label is a regression target."""
        if self.label_mode != "auto":
            return self.label_mode == "regression"
        if isinstance(label, (float, np.floating)):
            return True
        if isinstance(label, (list, np.ndarray)):
            array = np.asarray(label)
            return array.size > 0 and np.issubdtype(array.dtype, np.floating)
        return False

    def _build_label(self, label: Any, is_regression: bool) -> torch.Tensor:
        """Convert a stored label into a tensor with the right dtype and shape."""
        dtype = torch.float32 if is_regression else torch.long
        if self.dataset_kind == "sequence":
            array = np.asarray(label, dtype=np.float64 if is_regression else np.int64)
            return torch.as_tensor(array.reshape(-1), dtype=dtype)
        return torch.tensor(float(label) if is_regression else int(label), dtype=dtype)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Decode, normalise and pad one sample."""
        if self.use_cache and idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]

        self._ensure_env()
        key = self.keys[idx]
        with self.env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(
                f"Key {key!r} is missing from {self.lmdb_path}. The index file and the "
                "LMDB are out of sync; rebuild the dataset."
            )
        entry = pickle.loads(raw)

        x = torch.as_tensor(entry["eeg"], dtype=torch.float32)
        channel_coords = torch.as_tensor(entry["channel_coords"], dtype=torch.float32)

        expected_dims = 2 if self.dataset_kind == "window" else 3
        if x.dim() != expected_dims:
            raise ValueError(
                f"dataset_kind='{self.dataset_kind}' expects {expected_dims}-dimensional eeg, "
                f"got shape {tuple(x.shape)}"
            )

        if self.apply_minmax:
            x = self._min_max_normalize(x)

        pad_channels, pad_time = 0, 0
        if self.apply_zero_padding:
            x, channel_coords, pad_channels, pad_time = self._pad(x, channel_coords)

        sample: Dict[str, Any] = {
            "input": x,
            "channel_coords": channel_coords,
            "num_padded_channels": pad_channels,
            "num_padded_timesteps": pad_time,
        }

        raw_label = entry.get("label")
        if raw_label is not None:
            is_regression = self._is_regression(raw_label)
            sample["label"] = self._build_label(raw_label, is_regression)

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
