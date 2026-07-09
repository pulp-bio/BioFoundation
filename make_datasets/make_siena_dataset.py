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
import mne
import argparse
import tqdm
import h5py
import os

import numpy as np

from pathlib import Path

standard = {
    'EEG FP1', 'EEG F3', 'EEG C3', 'EEG P3', 'EEG O1', 'EEG F7', 'EEG T3', 'EEG T5', 'EEG FC1', 'EEG FC5', 
    'EEG CP1', 'EEG CP5', 'EEG F9', 'EEG FZ', 'EEG CZ', 'EEG PZ', 'EEG FP2', 'EEG F4', 'EEG C4', 'EEG P4', 'EEG O2',
    'EEG F8', 'EEG T4', 'EEG T6', 'EEG FC2', 'EEG FC6', 'EEG CP2', 'EEG CP6', 'EEG F10'
}

all_channels = list(standard)
sampling_freq = 256 # target sampling frequency
segment_length = 5 # segment length in seconds

def create_hdf5(sliced_data, output_dir, session_name):
    """
    Create HDF5 file from sliced data. 
    
    Args:
        sliced_data (np.ndarray): Sliced data of shape (n_channels, n_intervals, interval_size).
        output_dir (str or Path): Directory to save the HDF5 file.
        session_name (str): Name of the session/file.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, f"SIENA_29_channels.h5")

    with h5py.File(target_path, "a") as hdf5_file:
        group = hdf5_file.create_group(session_name)
        X_data = np.array(sliced_data, dtype=np.float16)
        group.create_dataset("X", data=X_data, dtype='float16')

def process_and_save_files_to_hdf5(file_paths, output_dir):
    """
    Process EDF files, extract standard EEG channels, filter, downsample, segment and save to HDF5.

    Args:
        file_paths (list of Path): List of paths to EDF files.
        output_dir (str or Path): Directory to save processed HDF5 files.
    """

    for file_path in tqdm.tqdm(file_paths):

        print(f"Processing file: {file_path.name}")
        session_name = file_path.parts[-1][:-4]
        # Load raw EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # First map all channel names to upper-case to have consistency
        upper_mapping = {ch: ch.upper() for ch in raw.ch_names}
        raw.rename_channels(upper_mapping)
        raw.pick(all_channels)

        # Strip EEG part from mapping
        mapping = {ch: ch.split(' ')[1] for ch in raw.ch_names}
        raw.rename_channels(mapping)

        # Filtering (bandpass and notch)
        raw.filter(l_freq=0.1, h_freq=75.0, verbose="ERROR")
        raw.notch_filter(50, verbose="ERROR")

        # Downsample 
        if raw.info['sfreq'] != sampling_freq:
            raw.resample(sampling_freq)
        data = raw.get_data(units='uV')
        
        # Check for NaN and Infs (just for notification)
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"Warning: NaN or Inf values found in file {file_path.name}")
        
        n_channels, n_times = data.shape
        print(f"Data shape (channels x timepoints): {data.shape}")
        interval_size = segment_length * sampling_freq
        num_intervals = n_times // interval_size

        new_sliced_data = data[:, :num_intervals * interval_size].reshape(num_intervals, n_channels, interval_size)

        create_hdf5(new_sliced_data, output_dir, session_name)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prepath', type=str, default='#CHANGEME', help='Input directory containing raw EDF files')
    parser.add_argument('--output_path', type=str, default='#CHANGEME')

    # Parse arguments
    args = parser.parse_args()
    input_dir = args.prepath
    output_dir = args.output_path

    # Find all EDF files in the input directory
    input_dir = Path(input_dir)
    file_paths = [f for f in input_dir.rglob('*.edf')]
    print(f"Found {len(file_paths)} EDF files.")

    process_and_save_files_to_hdf5(file_paths, output_dir)



