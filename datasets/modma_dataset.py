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
import mne

MODMA_CHN_ORDER = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10',
                    'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20',
                      'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30',
                        'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40',
                          'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50',
                            'E51', 'E52', 'E53', 'E54', 'E55', 'E56', 'E57', 'E58', 'E59', 'E60', 
                            'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69', 'E70',
                              'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80',
                                'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88', 'E89', 'E90',
                                  'E91', 'E92', 'E93', 'E94', 'E95', 'E96', 'E97', 'E98', 'E99', 'E100',
                                    'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110', 
                                    'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120',
                                      'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128']  # Cz is dropped during preprocessing, so we only have E1 to E128


RELEVANT_SUBSET = ['E3', 'E4', 'E9', 'E11', # FRONTAL LOBE
                   'E36', 'E37', 'E45', # TEMPORAL REGION
                   'E83', 'E92', 'E94', # PARIETAL LOBE
                   'E104', 'E108', 'E116' # OCCIPITAL LOBE
                   ]

def get_channel_locations(channel_names):
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    ch_pos = montage.get_positions()['ch_pos']
    locations = {ch: ch_pos[ch] for ch in channel_names if ch in ch_pos}
    # return as ordered list of locations
    ordered_locations = [locations[ch] for ch in channel_names if ch in locations]
    return np.stack(ordered_locations, axis=0)

class MODMA_Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file, num_channels=128, finetune=True, channel_subset=False):
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
            self.channel_names = RELEVANT_SUBSET
        else:
            self.channel_names = MODMA_CHN_ORDER[:num_channels]
        self.num_channels = len(self.channel_names)

        # ------------------------------------------------
        # Indices into the data tensor X
        # ------------------------------------------------
        self.data_channel_indices = torch.tensor(
            [MODMA_CHN_ORDER.index(ch) for ch in self.channel_names],
            dtype=torch.long
        )

        # ------------------------------------------------
        # Locations for embeddings
        # ------------------------------------------------
        self.channel_locations = get_channel_locations(self.channel_names)
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
                y = int(grp["y"][sample_idx])
                return_dict['label'] = y
            else:
                print(f"Warning: No labels found for sample {group_key}/{sample_idx} in finetune mode.")
        return return_dict

    def __del__(self):
        self.data.close()