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
import wfdb 
import numpy as np
import pandas as pd
import pickle
import argparse
import tqdm
import ast
import os 
import h5py
from scipy.io import loadmat

from process_raw_ecg import preprocess_signal

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
        for i, file in enumerate(tqdm.tqdm(files)):
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

def process_csv_files(args, csv_file, split_type):
    """
    Process CPSC2018 ECG dataset based on a CSV file containing file paths. 
    
    Args:
        args: Command line arguments. 
        csv_files (str): Path to the CSV file containing file paths. 
        split_type (str): Type of split ('train', 'val', 'test').
    """
    # Make directory for this split if it doesn't exist
    os.makedirs(os.path.join(args.output_path, split_type), exist_ok=True)
    
    # Read the csv file
    print(f"Processing {csv_file}...")
    df = pd.read_csv(csv_file)
    
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        
        # Load the signal
        try:
            file_path = os.path.join(args.prepath, f"{row['filename']}.mat")
            signal = loadmat(file_path)['val'][:, :2500]
            ecg = np.array(signal)

            sampling_rate = 500 # hopefully, it's 500Hz
            
            # Preprocess the signal and time-segment if needed
            processed_ecg = preprocess_signal(ecg, sampling_rate, 0.5, 120, args.downsample_fs, None)
            
            # Get the labels - there's 9 classes in total and they take last 9 columns of the csv
            y = row.iloc[-9:].values.astype(np.int8)
            
            # Write splits to pickle files
            data_dict = {"X": processed_ecg, "y": y}
            dump_path = os.path.join(args.output_path, split_type, f'cpsc2018-{idx}.pkl')
            
            with open(dump_path, "wb") as f:
                pickle.dump(data_dict, f)
                
        except Exception as e:
            print(f"Skipped file {file_path}. Error occured: {e}")
            continue
            
def main_splitted(args):
    """
    Main function for processing CPASC2018 dataset and saving as pickle file. 
    Uses splits provided by MERL ICML 2024 paper for consistency in comparison (https://github.com/cheliu-computation/MERL-ICML2024/tree/main/finetune/data_split). 
    """
    train_csv = os.path.join(args.csv_files_dir, "icbeb_train.csv")
    val_csv = os.path.join(args.csv_files_dir, "icbeb_val.csv")
    test_csv = os.path.join(args.csv_files_dir, "icbeb_test.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Write each .csv file into pickle files with X and y
    process_csv_files(args, train_csv, split_type="train")
    process_csv_files(args, val_csv, split_type="val")
    process_csv_files(args, test_csv, split_type="test")
    
    # Finally write to HDF5
    to_do = [f'train', f'val', f'test']
    for td in to_do:
        if os.path.exists(args.output_path + '/' + td + ".h5"):
            print(f"File {td}.h5 already exists!")
        else:
            print(f"Creating file {td}.h5.")
            create_hdf5(args.output_path+ "/" + td, args.output_path + "/" + td + ".h5")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process CPSC2018 ECG dataset and save as pickle files.")
    parser.add_argument('--prepath', type=str, default="#CHANGEME", help="Directory containing CPSC2018 WFDB files.")
    parser.add_argument('--output_path', type=str, default='#CHANGEME', help="Directory to save processed pickle files.")
    parser.add_argument("--csv_files_dir", type=str, default='#CHANGEME')
    parser.add_argument("--downsample_fs", type=int, default=256, help="Desired downsampling frequency. If None, no downsampling is done.")
    
    args = parser.parse_args()
    
    main_splitted(args)
