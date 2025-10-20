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
from torcheeg.datasets import SEEDVDataset
import os
import uuid
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_CHANNEL_LIST
import numpy as np
from models.modules.channel_embeddings import get_channel_indices, get_channel_locations

class CustomSEEDDataset(SEEDVDataset):
    def __init__(
        self,
        root_path,
        num_workers,
        num_channels=None,
        io_path=None,
        train_size=5,
        val_size=5,
        test_size=5,
        *args,
        **kwargs
    ):
        # Set up a unique path for caching if none is provided
        self.num_channels = num_channels
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        # os.makedirs(io_path, exist_ok=True)
        super().__init__(
            root_path=root_path,
            io_path=io_path,
            chunk_size=200,
            online_transform=transforms.ToTensor(),
            label_transform=transforms.Select("emotion"),  # Replace if necessary
            num_worker=num_workers,
            *args,
            **kwargs
        )
        self.channel_locations = np.stack(get_channel_locations(SEED_CHANNEL_LIST), axis=0)
        self.channel_locations = torch.from_numpy(self.channel_locations).to(torch.float)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        eeg_signal, label = super().__getitem__(int(index))
        # Rescale labels from (-1, 0, 1) to (0, 1, 2)
        return {"input": eeg_signal, "label": label, 'channel_locations': self.channel_locations}
