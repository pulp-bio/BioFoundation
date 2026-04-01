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
#* Author:  Danaé Broustail                                                   *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import torch
import h5py
import numpy as np
from models.modules.channel_embeddings import get_channel_indices, get_channel_locations

MoBI_CHANNEL_ORDER = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 
 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4',
   'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 
   'F6', 'FT7', 'FC3', 'FC4', 'FT8', 'C5', 'C1', 'C2', 'C6', 'TP7', 
 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']


CHANNEL_SUBSET_19 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'C3', 'Cz', 'C4',
    'P3', 'Pz', 'P4',
    'T7', 'T8', 'P7', 'P8',
    'O1', 'O2']

class MoBI_Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file, num_channels=60, finetune=True, channel_subset=False):
        self.hdf5_file = hdf5_file
        self.channel_subset = channel_subset
        self.data = h5py.File(self.hdf5_file, "r")
        self.keys = list(self.data.keys())
        self.finetune = finetune
        self.index_map = []

        for key in self.keys:
            if key=='index_map':
                continue
            group_size = len(self.data[key]["X"])
            self.index_map.extend([(key, i) for i in range(group_size)])

        # -----------------------------
        # Channel selection
        # -----------------------------
        if self.channel_subset:
            self.channel_names = CHANNEL_SUBSET_19
        else:
            self.channel_names = MoBI_CHANNEL_ORDER[:num_channels]

        self.num_channels = len(self.channel_names)

        # ------------------------------------------------
        # Indices into the data tensor X
        # ------------------------------------------------
        self.data_channel_indices = torch.tensor(
            [MoBI_CHANNEL_ORDER.index(ch) for ch in self.channel_names],
            dtype=torch.long
        )
        self.num_channels = len(self.channel_names)

        # ------------------------------------------------
        # Locations for embeddings
        # ------------------------------------------------
        self.channel_locations = np.stack(get_channel_locations(self.channel_names), axis=0)
        self.channel_locations = torch.from_numpy(self.channel_locations).to(torch.float)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]

        X = torch.FloatTensor(grp["X"][sample_idx])
        if self.channel_subset:
            X = X[self.data_channel_indices]

        return_dict = {"input": X, 'channel_locations': self.channel_locations}
        if self.finetune:
            if 'y' in grp:
                y = torch.FloatTensor(grp["y"][sample_idx])
                return_dict['label'] = y
            else:
                print(f"Warning: No labels found for sample {group_key}/{sample_idx} in finetune mode.")
        return return_dict

    def __del__(self):
        self.data.close()