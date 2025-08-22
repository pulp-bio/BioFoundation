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
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import os
import pickle
import numpy as np
import h5py
from tqdm import tqdm

# Check the hostname to set the DATA_PATH accordingly
hostname = os.uname().nodename

prepath = '#CHANGEME'

def create_hdf5(source_dir, target_file, finetune=True,group_size=1000):
    files = os.listdir(source_dir)
    data_group = []
    with h5py.File(target_file, 'w') as h5f:
        for i, file in enumerate(tqdm(files)):
            with open(os.path.join(source_dir, file), 'rb') as f:
                sample = pickle.load(f)
                data_group.append(sample)
                if (i + 1) % group_size == 0 or i == len(files) - 1:
                    grp = h5f.create_group(f"data_group_{i // group_size}")
                    X_data = np.array([s['X'] for s in data_group])
                    if(finetune):
                        y_data = np.array([s['y'] for s in data_group])
                    grp.create_dataset("X", data=X_data)
                    if(finetune):
                        grp.create_dataset("y", data=y_data)
                    data_group = []

# Check first if the .h5 files already exist
# Pretraining data first:
folders = ['TUAR_data', 'TUSL_data',]
to_do = ['train', 'val']
for folder in folders:
    for td in to_do:
        if os.path.exists(prepath + "/" + folder + "/" + td + ".h5"):
            print(f"{folder} {td}.h5 already exists")
            print("Skipping...")
        else:
            print(f"Creating {folder} {td}.h5")
            create_hdf5(prepath + "/" + folder + "/processed/" + td, prepath + "/" + folder + "/" + td + ".h5", finetune=False)

# Now the TUAB data
folders = ['TUAB_data']
to_do = ['train', 'val', 'test']
for folder in folders:
    for td in to_do:
        if os.path.exists(prepath + "/" + folder + "/" + td + ".h5"):
            print(f"{folder} {td}.h5 already exists")
            print("Skipping...")
        else:
            print(f"Creating {folder} {td}.h5")
            create_hdf5(prepath + "/" + folder + "/processed/" + td, prepath + "/" + folder + "/" + td + ".h5", finetune=True)