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
#* Author:  Berkay Döner                                                      *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

import torch
import h5py
import numpy as np
from models.modules.channel_embeddings import get_channel_indices, get_channel_locations

CHN_ORDER = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "A1-T3",
    "T4-A2",
]


class TUH_Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file, num_channels=None, finetune=False):
        self.hdf5_file = hdf5_file
        self.num_channels = num_channels
        self.channel_names = CHN_ORDER[: self.num_channels]
        self.data = h5py.File(self.hdf5_file, "r")
        self.keys = list(self.data.keys())
        self.finetune = finetune
        self.index_map = []
        for key in self.keys:
            if key=='index_map':
                continue
            group_size = len(self.data[key]["X"])
            self.index_map.extend([(key, i) for i in range(group_size)])
        self.channel_indices = get_channel_indices(self.channel_names)
        self.channel_indices = torch.tensor(self.channel_indices).to(torch.long)
        self.channel_locations = np.stack(get_channel_locations(self.channel_names), axis=0)
        self.channel_locations = torch.from_numpy(self.channel_locations).to(torch.float)
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]

        X = torch.FloatTensor(grp["X"][sample_idx])
        chns = self.channel_indices

        return_dict = {"input": X, 'channel_names': chns, 'channel_locations': self.channel_locations}
        if self.finetune:
            if 'y' in grp:
                y = int(grp["y"][sample_idx])
                return_dict['label'] = y
            else:
                print(f"Warning: No labels found for sample {group_key}/{sample_idx} in finetune mode.")
        return return_dict

    def __del__(self):
        self.data.close()
