from collections import deque

import h5py
import torch


class EMGDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        hdf5_file,
        squeeze=False,
        finetune=True,
        cache_size=1500,
        use_cache=True,
        regression: bool = False,
        fold_indices=None,
    ):
        self.hdf5_file = hdf5_file
        self.squeeze = squeeze
        self.cache_size = cache_size
        self.finetune = finetune
        self.use_cache = use_cache
        self.data = h5py.File(self.hdf5_file, "r")
        self.keys = list(self.data.keys())
        self.regression = regression
        self.fold_indices = fold_indices

        self._data = self.data["data"][:]
        if self.finetune:
            self._labels = self.data["label"][:]

        self.num_samples = self._data.shape[0]

        # Apply fold indices if provided
        if self.fold_indices is not None:
            self._data = self._data[self.fold_indices]
            if self.finetune:
                self._labels = self._labels[self.fold_indices]
            self.num_samples = len(self.fold_indices)

        # Cache to store recently accessed samples
        if self.use_cache:
            self.cache = {}
            self.cache_queue = deque(maxlen=self.cache_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.use_cache and index in self.cache:
            cached_data = self.cache[index]
            if self.finetune:
                X, Y = cached_data
            else:
                X = cached_data
        else:
            X = self._data[index]
            X = torch.FloatTensor(X)

            if self.finetune:
                Y = self._labels[index]
                if self.regression:
                    Y = torch.FloatTensor([Y]).squeeze()
                else:
                    Y = torch.LongTensor([Y]).squeeze()
                if self.use_cache:
                    self.cache[index] = (X, Y)
            else:
                if self.use_cache:
                    self.cache[index] = X

            if self.use_cache:
                self.cache_queue.append(index)

        if self.squeeze:
            X = X.unsqueeze(0)

        if self.finetune:
            return X, Y
        else:
            return X

    def __del__(self):
        self.data.close()
