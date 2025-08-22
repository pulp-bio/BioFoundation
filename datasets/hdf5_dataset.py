#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
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
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import torch
import h5py
from collections import deque

class HDF5Loader(torch.utils.data.Dataset):
    def __init__(self, hdf5_file, squeeze=False, finetune=True, cache_size=1500, use_cache=True):
        self.hdf5_file = hdf5_file
        self.squeeze = squeeze
        self.cache_size = cache_size
        self.finetune = finetune
        self.use_cache = use_cache
        self.data = h5py.File(self.hdf5_file, 'r')
        self.keys = list(self.data.keys())

        self.index_map = []
        for key in self.keys:
            group_size = len(self.data[key]['X'])  # Always assume 'X' is present
            self.index_map.extend([(key, i) for i in range(group_size)])
        
        # Cache to store recently accessed samples
        if self.use_cache:
            self.cache = {}
            self.cache_queue = deque(maxlen=self.cache_size)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        if self.use_cache and index in self.cache:
            cached_data = self.cache[index]
            if self.finetune:
                X, Y = cached_data
            else:
                X = cached_data
        else:
            group_key, sample_idx = self.index_map[index]
            grp = self.data[group_key]
            X = grp["X"][sample_idx]
            X = torch.FloatTensor(X)

            if self.finetune:
                Y = grp["y"][sample_idx]
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