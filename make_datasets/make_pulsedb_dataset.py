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
import scipy.io
import pickle
import argparse
import multiprocessing
import tqdm
import h5py
import ast
import os

from process_raw_ecg import preprocess_signal, time_segmenting
from pathlib import Path
from sklearn.model_selection import train_test_split
from mat73 import loadmat

def dump_data(data_slice, output_dir, data_group_name, file_idx, channel_info, dataset):

    data_group = data_slice
    output_path = os.path.join(output_dir, f"{dataset}_2_channels_{file_idx}.h5")
    
    with h5py.File(output_path, "a") as h5f:
        
        if h5f.attrs.get("channel_names") is None:
            h5f.attrs["channel_names"] = channel_info
            
        grp = h5f.create_group(data_group_name)
        X_data = np.array(data_group, dtype=np.float16)
        grp.create_dataset("X", data=X_data, dtype='float16')
        
    print(f"Saved {len(data_slice)} samples to {output_path}.")

def process_single_file(file_paths, output_dir, sampling_rate, upsample_fs, split_signal, dataset, file_idx, worker_id):
    """
    Process a single ,mat file from VitalDB dataset. Friendly for multiprocessing.
    """
    
    with tqdm.tqdm(total=len(file_paths), desc=f"Worker {worker_id}", position=worker_id) as pbar:
        for file_path in file_paths:
            channel_info = ['II', 'PPG']
            
            # Handle case if file is corrupted
            try:
                session_name = file_path.parts[-1][:-4]
                data = loadmat(file_path)
                raw_ecg = np.array(data['Subj_Wins']['ECG_Raw'])
                raw_ppg = np.array(data['Subj_Wins']['PPG_Raw'])
        
            except Exception as e:
                print(f"Standard loading failed for {session_name}. Falling back to scipy.io.loadmat().")
                try:
                    session_name = file_path.parts[-1][:-4]
                    data = scipy.io.loadmat(file_path)
                    unpacked = data['Subj_Wins'][0, 0]
                    raw_ecg = np.array(unpacked['ECG_Raw']).T
                    raw_ppg = np.array(unpacked['PPG_Raw']).T 
                
                except Exception as scipy_e:
                    print(f"Error loading {file_path} even with scipy.io. {scipy_e}.")
                    return None 

            # Preprocess signals and concatenate
            processed_ecg = preprocess_signal(raw_ecg, sampling_rate, 0.5, 120, None, upsample_fs)
            processed_ppg = preprocess_signal(raw_ppg, sampling_rate, 0.5, 12, None, upsample_fs)
            
            if processed_ecg.ndim < 3 or processed_ppg.ndim < 3:
                processed_ecg = processed_ecg.reshape(1, 1, -1)
                processed_ppg = processed_ppg.reshape(1, 1, -1)
                
            signals = np.concatenate((processed_ecg, processed_ppg), axis=1)
            
            merged_splits = signals
            
            # Time splitting
            if split_signal is not None:

                signal_splitted = time_segmenting(signals, split_signal, sampling_rate, None, upsample_fs)
                merged_splits = np.concatenate(signal_splitted, axis=0)
                
            dump_data(merged_splits, output_dir, session_name, file_idx, channel_info, dataset)

            pbar.update(1)

def loading_and_processing_parallel(files, output_dir, sampling_rate, upsampling_fs, split_signal, dataset, max_sessions_per_file=300):
    """
    Load and process all .mat files in the input directory in parallel.
    
    Args:
        files (list): List of .mat filenames to process.
        output_dir (str): Directory to save processed pickle files.
        sampling_rate (int): Original sampling rate of the signals.
        upsampling_fs (int): Desired upsampling frequency.
        split_signal (int or None): Length of segments to split the signals into (in seconds). If None, no splitting is done.
        dataset (str): Dataset name. 
    """
    num_workers = 4
    worker_args = []
    
    # Prepare arguments for parallel processing
    for i in range(0, len(files), max_sessions_per_file):
        worker_args.append((files[i:i+max_sessions_per_file], output_dir, sampling_rate, upsampling_fs, split_signal, dataset, int(i / max_sessions_per_file)))
        
    print(f"Processing {len(worker_args)} groups of files:")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use multiprocessing to parallelize the process
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(process_single_file, [(args[0], args[1], args[2], args[3], args[4], args[5], args[6], worker_id) for worker_id, args in enumerate(worker_args)])      
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process VitalDB .mat files and save as pickle files.")
    parser.add_argument('--dataset', type=str, default='VitalDB', help='Either VitalDB or MimicDB.')
    parser.add_argument('--prepath', type=str, default='#CHANGEME', help='Directory containing VitalDB .mat files.')
    parser.add_argument('--output_path', type=str, default='#CHANGEME', help='Directory to save processed pickle files.')
    parser.add_argument('--sampling_rate', type=int, default=125, help='Original sampling rate of the signals.')
    parser.add_argument('--upsampling_fs', type=int, default=256, help='Desired upsampling frequency.')
    parser.add_argument('--split_signal', type=int, default=5, help='Length of segments to split the signals into (in seconds). If None, no splitting is done.')

    args = parser.parse_args()
    dataset = args.dataset
    input_dir = Path(args.prepath)
    output_dir = Path(args.output_path)
    sampling_rate = args.sampling_rate
    upsampling_fs = args.upsampling_fs
    split_signal = args.split_signal

    # List all .mat files in the input directory 
    files = [f for f in input_dir.rglob('*.mat')]
    loading_and_processing_parallel(files, output_dir, sampling_rate, upsampling_fs, split_signal, dataset)
    
