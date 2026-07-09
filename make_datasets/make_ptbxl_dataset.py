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
import pandas as pd
import numpy as np 

import wfdb
import ast
import argparse
import os
import pickle
import h5py

from process_raw_ecg import preprocess_signal
from itertools import chain
from tqdm import tqdm

def create_hdf5(source_dir, target_file, finetune=True, group_size=1000):
    """
    Lists all the pickle files in the source directory and writes them in the .h5 file.

    Args:
        source_dir (str): Source directory where pickle files are found. 
        target_file (str): Path to the target .h5 file. 
        finetune (bool): Whether data has label or not. 
        group_size (int): Group size in HDF5 saving. 
    """
    # List all the files in the folder
    files = sorted(os.listdir(source_dir))
    data_group = []

    
    with h5py.File(target_file, 'w') as h5f:
        for i, file in enumerate(tqdm(files)):
            with open(os.path.join(source_dir, file), 'rb') as f:
                
                sample = pickle.load(f)
                data_group.append(sample)
                
                if (i + 1) % group_size == 0 or i == len(files) - 1:
                    
                    grp = h5f.create_group(f"data_group_{i // group_size}")
                    X_data = np.array([s['X'] for s in data_group])
                    grp.create_dataset("X", data=X_data)
                    
                    if(finetune):
                        y_data = np.array([s['y'] for s in data_group])
                        grp.create_dataset("y", data=y_data)

                    data_group = []

def process_csv_files(args, csv_file, setup, split_type):
    """
    Process csv files and dump into pickle files. The csv files are taken from MERL ICML 2024 repository (https://github.com/cheliu-computation/MERL-ICML2024/tree/main/finetune/data_split).

    Args:
        args: Parsed arguments from argparse.
        csv_file (str): Path to the csv file.
        setup (str): One of ['super_class', 'sub_class', 'form' and 'rhythm'].
        split (str): One of ['train', 'val', 'test'].
    """
    # Make output directory for this setup if it doesn't exist
    os.makedirs(os.path.join(args.output_path, setup, split_type), exist_ok=True)
    
    # Read the csv file
    print(f"Processing {csv_file}...")
    df = pd.read_csv(csv_file)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        # Load the signal, we always use the 500 Hz version
        file_path = os.path.join(args.prepath, row['filename_hr'])
        signal, _ = wfdb.rdsamp(file_path)

        # Preprocess the signal 
        processed_signal = preprocess_signal(signal.T, args.sampling_rate, 0.5, 120, args.downsampling_fs, None)

        # Get the labels
        if setup == 'super_class': # superclass setup has 5 classes / last 5 columns of every row are labels
            y = row.iloc[-5:].values.astype(np.int8)
        elif setup == 'sub_class': # subclass setup has 23 classes / last 23 columns of every row are labels
            y = row.iloc[-23:].values.astype(np.int8)
        elif setup == 'form': # form setup has 19 classes / last 19 columns of every row are labels
            y = row.iloc[-19:].values.astype(np.int8)
        elif setup == 'rhythm': # rhythm setup has 12 classes / last 12 columns of every row are labels
            y = row.iloc[-12:].values.astype(np.int8)
        else:
            raise ValueError("Invalid setup provided.")
                
        data_dict = {"X": processed_signal, "y": y}
        dump_path = os.path.join(args.output_path, setup, split_type, f'ptb-xl-{idx}.pkl')
        
        with open(dump_path, "wb") as f:
            pickle.dump(data_dict, f)
        
def main_splitted(args, setup):
    """
    Main function for processing PTB_XL dataset and saving as pickle file.
    Uses splits provided by MERL paper for consistency in comparison.
    Processing is done with the respect to the the pipeline my code expects.
    Filtering, downsampling and segmenting in 5s pieces is done here.  
    
    Args:
        args: Parsed arguments from argparse.
        setup (str): One of ['super_class', 'sub_class', 'form' and 'rhythm'].
    """
    train_csv = os.path.join(args.csv_files_dir, setup, f'ptbxl_{setup}_train.csv')
    val_csv = os.path.join(args.csv_files_dir, setup, f'ptbxl_{setup}_val.csv')
    test_csv = os.path.join(args.csv_files_dir, setup, f'ptbxl_{setup}_test.csv')

    # Create output directory for this setup if it doesn't exist
    os.makedirs(os.path.join(args.output_path, setup), exist_ok=True)

    # Write each csv into pickle files with X and y
    process_csv_files(args, train_csv, setup, split_type='train')
    process_csv_files(args, val_csv, setup, split_type='val')
    process_csv_files(args, test_csv, setup, split_type='test')

    # Finally, write to HDF5
    to_do = ['train', 'val', 'test']
    for td in to_do:
        if os.path.exists(args.output_path + '/' + setup + "/" + td + "/" + ".h5"):
            print(f"File {td}.h5 already exists!")
        else:
            print(f"Creating file {td}.h5.")
            create_hdf5(args.output_path + "/" + setup + "/" + td, args.output_path + "/" + setup + "/" + td + ".h5")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process PTB-XL dataset files and save as pickle.")
    parser.add_argument('--prepath', type=str, default='#CHANGEME')
    parser.add_argument('--output_path', type=str, default='#CHANGEME')
    parser.add_argument('--csv_files_dir', type=str, default='#CHANGEME')
    parser.add_argument('--sampling_rate', type=int, default=500)
    parser.add_argument('--downsampling_fs', type=int, default=256)
    parser.add_argument('--setup', type=str, default=5, help="Either super_class, sub_class, form or rhythm.")
    
    args = parser.parse_args()
    setup = args.setup
    
    # Run main function for 
    main_splitted(args, setup=setup)
