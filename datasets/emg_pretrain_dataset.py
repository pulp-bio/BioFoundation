import os
import threading
from collections import deque

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# thread‐local storage for per‐worker file handle
_thread_local = threading.local()


def _get_h5_handle(path):
    h5f = getattr(_thread_local, "h5f", None)
    if h5f is None or h5f.filename != path:
        h5f = h5py.File(path, "r")
        _thread_local.h5f = h5f
    return h5f


class EMGPretrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        squeeze: bool = False,
        cache_size: int = 1500,
        use_cache: bool = True,
        pad_up_to_max_chans: int | None = None,
        max_samples: int | None = None,
    ):
        super().__init__()
        self.squeeze = squeeze
        self.cache_size = cache_size
        self.use_cache = use_cache
        self.pad_up_to_max_chans = pad_up_to_max_chans

        # discover all .h5 files
        self.file_paths = sorted(
            os.path.join(data_dir, fn)
            for fn in os.listdir(data_dir)
            if fn.endswith(".h5")
        )
        if not self.file_paths:
            raise RuntimeError(f"No .h5 files in {data_dir!r}")

        # build index of (file_path, sample_idx)
        self.index_map = []
        for fp in self.file_paths:
            with h5py.File(fp, "r") as h5f:
                n = h5f["data"].shape[0]
            for i in range(n):
                self.index_map.append((fp, i))
        if max_samples is not None:
            self.index_map = self.index_map[:max_samples]

        # Cache to store recently accessed samples
        if use_cache:
            self.cache = {}
            self.cache_queue = deque(maxlen=self.cache_size)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        if self.use_cache and index in self.cache:
            cached_data = self.cache[index]
            X = cached_data
        else:
            fp, local_idx = self.index_map[index]
            h5f = _get_h5_handle(fp)
            np_x = h5f["data"][local_idx]  # shape (C, T)
            X = torch.from_numpy(np_x).float()

            if self.use_cache:
                self.cache[index] = X
                self.cache_queue.append(index)

        # squeeze if requested
        if self.squeeze:
            X = X.unsqueeze(0)

        # pad channels if requested
        if self.pad_up_to_max_chans is not None:
            C = X.shape[0]
            to_pad = self.pad_up_to_max_chans - C
            if to_pad > 0:
                X = X.T
                X = F.pad(X, (0, to_pad))
                X = X.T

        return X

    def __del__(self):
        h5f = getattr(_thread_local, "h5f", None)
        if h5f is not None:
            h5f.close()
