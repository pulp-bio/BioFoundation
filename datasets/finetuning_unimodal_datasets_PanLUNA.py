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
#* Author:  Marija Zelic                                                      *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import torch
import h5py
import numpy as np
from models.modules.lead_positions import (
    get_channel_indices,
    get_channel_locations,
    map_lead_labels_to_angles,
)

class FinetuningUnimodal_Dataset(torch.utils.data.Dataset):
    """
    Unified class for unimodal finetuning datasets. 
    
    Args:
        hdf5_file: Path to the .h5 file.
        channels: List of channels.
        location_fn: "eeg" or "ecg".
        sensor_type: 0 (ECG), 1(EEG), 2(PPG).
        num_channels: Number of channels taken from total channel list. 
    """
    def __init__(
        self, 
        hdf5_file: str,
        channels: list[str],
        location_fn: str,
        sensor_type: int | list[int],
        num_channels: int | None = None, 
    ):
        super().__init__()
        channel_names = channels[:num_channels] if num_channels else channels
        self.num_channels = len(channel_names)

        if location_fn == "eeg":
            locs = np.stack(get_channel_locations(channel_names), axis=0)
            self.channel_locations = torch.from_numpy(locs).float()
        else:
            self.channel_locations = torch.FloatTensor(map_lead_labels_to_angles(channel_names))
            
        self.channel_indices = torch.tensor(get_channel_indices(channel_names), dtype=torch.long)
        
        if isinstance(sensor_type, list):
            self.sensor_type = torch.tensor(sensor_type, dtype=torch.long)
        else:
            self.sensor_type = torch.full((self.num_channels, ), sensor_type, dtype=torch.long)
            
        self.data = h5py.File(hdf5_file, "r")
        self.keys = list(self.data.keys())
        self.index_map = []
        
        for key in self.keys:
            if key == 'index_map':
                continue
            group_size = len(self.data[key]["X"])
            self.index_map.extend([(key, i) for i in range(group_size)])
            
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]
        X = torch.FloatTensor(grp["X"][sample_idx])
        label = torch.tensor(grp["y"][sample_idx], dtype=torch.long)
        
        item = {
            "input": X,
            "channel_names": self.channel_indices,
            "channel_locations": self.channel_locations,
            "sensor_type": self.sensor_type,
            "label": label
        }
        
        return item
    
    def __del__(self):
        if hasattr(self, "data"):
            self.data.close()