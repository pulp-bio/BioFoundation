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
import h5py 
import argparse
import multiprocessing
import tqdm
import os

import numpy as np
from process_raw_ecg import preprocess_signal, time_segmenting

sampling_freq = 400
CODE15_channels = ['I', "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def dump_data(data_slice, output_dir, data_group_name, file_idx):
    """
    Write the data in H5 file. 
    """
    
    data_group = data_slice
    output_path = os.path.join(output_dir, f"CODE15_{file_idx}.h5")

    with h5py.File(output_path, "a") as h5f:

        if h5f.attrs.get("channel_names") is None:
            h5f.attrs['channel_names'] = CODE15_channels
        
        grp = h5f.create_group(data_group_name)
        X_data = np.array(data_group, dtype=np.float16)
        grp.create_dataset("X", data=X_data, dtype='float16')

    
def process_single_file(file_path, output_dir, downsample_fs, split_signal, file_idx, worker_id):
    """
    Process a single HDF5 file of Code15 dataset.

    Each sample has shape (4096, 12). Samples are either of length 10s or 7s.
    Sampling frequency is 400Hz. When length is 7s long, there's 2800 sample and to pad to 4096 zeros are addded.
    In case of 10s, there's no additional padding, but it's actually a signal.
    To have consistency, I first strip 96 samples, (48 at the beginning and 48 at the end) - it's going to be around 0.1s for full signals and just zeros for 7s long.
    And then process it.

    Args:
        file_path (str): Path to the file.
        output_dir (str): Output directory to which we are writing the H5 files.
        downsample_fs (int): Downsampling frequency.
        split_signal (int): Lenght of the split to segment the signal. Assumes no padding, no overlap.
    """
    
    batch = 1024
    with h5py.File(file_path, "r") as f:
        
        # Extract dataset storing the ECGs
        # We can't extract all the data in one np.array as it is too much
        ecgs = f['tracings']
        N = ecgs.shape[0]
        
        # We need to do it in batches
        for start in tqdm.tqdm(range(0, N, batch), desc=f"Worker {worker_id}", position=worker_id):
            session_name = f"code15_{start}"
            end = min(start+batch, N)
            
            # Extract batch_size samples and convert to np.array
            ecg_batch = np.array(ecgs[start:end])

            # Transpose and take middle 4000 samples (discarding first and last 48)
            ecg_batch = np.transpose(ecg_batch, axes=(0,2,1))[:, :, 48:4048]

            # Process and split the signal
            processed_ecg = preprocess_signal(ecg_batch, sampling_freq, 0.5, 120, downsample_fs, None)
            reshaped_ecg = processed_ecg

            # Split if split_signal provided
            if split_signal is not None:

                reshaped_ecg = np.expand_dims(processed_ecg, axis=0)
                signal_splitted = time_segmenting(reshaped_ecg, split_signal, sampling_freq, downsample_fs, None)
                merged_splits = np.concatenate(signal_splitted, axis=0)
                reshaped_ecg = merged_splits.reshape(-1, merged_splits.shape[2], merged_splits.shape[3])

            dump_data(reshaped_ecg, output_dir, session_name, file_idx)              

def loading_and_processing_parallel(files, output_dir, downsampling_fs, split_signal):
    """
    Load and process all the HDF5 files from Code15 dataset in the input directory in parallel.
    
    Args:
        files (list): List of HDF5 file paths to process. 
        output_dir (str): Directory to save processed pickle files. 
        downsampling_fs (int): Desired downsampling frequency.
        split_signal (int or None): Length or segments to split the signals into (in seconds). If None, no splitting is done. 
    """
    num_workers = 18
    worker_args = []
    
    # Prepare arguments for parallel processing
    for idx, file in enumerate(files):
        worker_args.append((file, output_dir, downsampling_fs, split_signal, idx))
        
    print(f"Processing {len(worker_args)} groups of files:")
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use multiprocessing to parallelize the process
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(process_single_file, [(args[0], args[1], args[2], args[3], args[4], worker_id) for worker_id, args in enumerate(worker_args)]) 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process Code15 HDF5 files and save as .h5 files.")
    parser.add_argument('--prepath', type=str, default='#CHANGEME', help='Directory contraining MIMIC-IV-ECG files.')
    parser.add_argument('--output_path', type=str, default='#CHANGEME', help='Directory to save processed .h5 files.')
    parser.add_argument('--downsampling_fs', type=int, default=256, help='Desired downsampling frequency.')
    parser.add_argument('--split_signal', type=int, default=5, help='Length of segments to split the signals into (in seconds). If None, no splitting is done.')
    
    args = parser.parse_args()
    input_dir = args.prepath
    output_dir = args.output_path
    downsampling_fs = args.downsampling_fs
    split_signal = args.split_signal
    
    # List all HDF5 files that are stored in the directory
    files = os.listdir(input_dir)
    path_files = [os.path.join(input_dir, file) for file in files]
    
    # Parallel writing
    loading_and_processing_parallel(path_files, output_dir, downsampling_fs, split_signal)
