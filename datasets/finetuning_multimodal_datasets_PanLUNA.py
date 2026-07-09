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
from typing import Optional

from models.modules.lead_positions import (
    map_lead_labels_to_angles,
    get_channel_indices,
    get_channel_locations
)

MODALITY_TO_SENSOR_TYPE = {"ecg": 0, "eeg": 1, "ppg": 2}
KNOWN_MODALITIES = {"eeg", "ecg", "ppg"}

def _compute_channel_location(
    channel_names: list[str],
    channel_modalities: list[str]
):
    """
    Build a (n_channels, max_dim) location array.
    Handles mixed EEG/ECG by zero-padding to the largest feature dimension.
    """
    raw_locs = []
    for ch, mod in zip(channel_names, channel_modalities):
        if mod == "eeg":
            loc = np.stack(get_channel_locations([ch]), axis=0)
        elif mod == "ecg" or mod == "ppg":
            loc = map_lead_labels_to_angles([ch])
        raw_locs.append(loc)
        
    max_dim = max(loc.shape[1] for loc in raw_locs)
    padded = [
        np.pad(loc, ((0, 0), (0, max_dim - loc.shape[1])), mode="constant")
        if loc.shape[1] < max_dim else loc
        for loc in raw_locs
    ]
    
    return np.vstack(padded)

class FinetuningMultimodal_Dataset(torch.utils.data.Dataset):
    """
    Unified class for mulitmodal finetuning datasets. Handles channel selection and channel location padding with support of hydra configuration.
    
    Args:
        hadf5_file: Path to the .h5 file.
        channel_groups: All channels in the dataset organized as in the finetune_data_module_multimodal.yaml
        channel_start: Starting index for channel slicing. Allows taking all or fraction of modalities.
        channel_end: Ending index for channel slicing.
    """
    def __init__(
        self,
        hdf5_file: str, 
        channel_groups: dict[str, list[str]],
        channel_start: Optional[int] = None,
        channel_end: Optional[int] = None, 
    ):
        super().__init__()
        self._x_slice = slice(channel_start, channel_end)
        
        # Flatten channel_groups into parallel list of names and modalities
        self.channel_names = []
        flat_modalities = []
        
        for modality, names in channel_groups.items():
            self.channel_names.extend(names)
            if modality in KNOWN_MODALITIES:
                flat_modalities.extend([modality] * len(names))
                
        self.channel_indices = torch.tensor(get_channel_indices(self.channel_names), dtype=torch.long)
        
        # Obtains channel locations
        locs = _compute_channel_location(self.channel_names, flat_modalities)
        self.channel_locations = torch.from_numpy(locs).float()
        self.sensor_type = torch.tensor([MODALITY_TO_SENSOR_TYPE[m] for m in flat_modalities], dtype=torch.long)
        
        # Open HDF5 and build flat
        self.data = h5py.File(hdf5_file, "r")
        self.keys = list(self.data.keys())
        self.index_map = []
        
        for key in self.keys:
            group_size = len(self.data[key]['X'])
            self.index_map.extend([(key, i) for i in range(group_size)])
            
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        
        group_key, sample_idx = self.index_map[index]
        grp = self.data[group_key]

        X = torch.FloatTensor(grp["X"][sample_idx])[self._x_slice, :]
        label = torch.tensor(grp["y"][sample_idx], dtype=torch.long)
        
        return_dict = {
            "input": X,
            "channel_names": self.channel_indices[self._x_slice],
            "channel_locations": self.channel_locations[self._x_slice],
            "sensor_type": self.sensor_type[self._x_slice],
            "label": label
        }
        
        return return_dict
        
        
        