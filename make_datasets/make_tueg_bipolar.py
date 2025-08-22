
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
import argparse
import numpy as np
import mne
import os
from pathlib import Path
from tqdm import tqdm
import h5py
import multiprocessing
import pickle
import matplotlib.pyplot as plt

TCP_MONTAGE_01_TCP_AR = {
    "FP1-F7": ["EEG FP1-REF", "EEG F7-REF"],
    "F7-T3": ["EEG F7-REF", "EEG T3-REF"],
    "T3-T5": ["EEG T3-REF", "EEG T5-REF"],
    "T5-O1": ["EEG T5-REF", "EEG O1-REF"],
    "FP2-F8": ["EEG FP2-REF", "EEG F8-REF"],
    "F8-T4": ["EEG F8-REF", "EEG T4-REF"],
    "T4-T6": ["EEG T4-REF", "EEG T6-REF"],
    "T6-O2": ["EEG T6-REF", "EEG O2-REF"],
    "T3-C3": ["EEG T3-REF", "EEG C3-REF"],
    "C3-CZ": ["EEG C3-REF", "EEG CZ-REF"],
    "CZ-C4": ["EEG CZ-REF", "EEG C4-REF"],
    "C4-T4": ["EEG C4-REF", "EEG T4-REF"],
    "FP1-F3": ["EEG FP1-REF", "EEG F3-REF"],
    "F3-C3": ["EEG F3-REF", "EEG C3-REF"],
    "C3-P3": ["EEG C3-REF", "EEG P3-REF"],
    "P3-O1": ["EEG P3-REF", "EEG O1-REF"],
    "FP2-F4": ["EEG FP2-REF", "EEG F4-REF"],
    "F4-C4": ["EEG F4-REF", "EEG C4-REF"],
    "C4-P4": ["EEG C4-REF", "EEG P4-REF"],
    "P4-O2": ["EEG P4-REF", "EEG O2-REF"],
    "A1-T3": ["EEG A1-REF", "EEG T3-REF"], #
    "T4-A2": ["EEG T4-REF", "EEG A2-REF"], #
}

TCP_MONTAGE_02_TCP_LE = {
    "FP1-F7": ["EEG FP1-LE", "EEG F7-LE"],
    "F7-T3": ["EEG F7-LE", "EEG T3-LE"],
    "T3-T5": ["EEG T3-LE", "EEG T5-LE"],
    "T5-O1": ["EEG T5-LE", "EEG O1-LE"],
    "FP2-F8": ["EEG FP2-LE", "EEG F8-LE"],
    "F8-T4": ["EEG F8-LE", "EEG T4-LE"],
    "T4-T6": ["EEG T4-LE", "EEG T6-LE"],
    "T6-O2": ["EEG T6-LE", "EEG O2-LE"],
    "T3-C3": ["EEG T3-LE", "EEG C3-LE"],
    "C3-CZ": ["EEG C3-LE", "EEG CZ-LE"],
    "CZ-C4": ["EEG CZ-LE", "EEG C4-LE"],
    "C4-T4": ["EEG C4-LE", "EEG T4-LE"],
    "FP1-F3": ["EEG FP1-LE", "EEG F3-LE"],
    "F3-C3": ["EEG F3-LE", "EEG C3-LE"],
    "C3-P3": ["EEG C3-LE", "EEG P3-LE"],
    "P3-O1": ["EEG P3-LE", "EEG O1-LE"],
    "FP2-F4": ["EEG FP2-LE", "EEG F4-LE"],
    "F4-C4": ["EEG F4-LE", "EEG C4-LE"],
    "C4-P4": ["EEG C4-LE", "EEG P4-LE"],
    "P4-O2": ["EEG P4-LE", "EEG O2-LE"],
    "A1-T3": ["EEG A1-LE", "EEG T3-LE"],
    "T4-A2": ["EEG T4-LE", "EEG A2-LE"],
}

TCP_MONTAGE_03_TCP_AR_A = {
    "FP1-F7": ["EEG FP1-REF", "EEG F7-REF"],
    "F7-T3": ["EEG F7-REF", "EEG T3-REF"],
    "T3-T5": ["EEG T3-REF", "EEG T5-REF"],
    "T5-O1": ["EEG T5-REF", "EEG O1-REF"],
    "FP2-F8": ["EEG FP2-REF", "EEG F8-REF"],
    "F8-T4": ["EEG F8-REF", "EEG T4-REF"],
    "T4-T6": ["EEG T4-REF", "EEG T6-REF"],
    "T6-O2": ["EEG T6-REF", "EEG O2-REF"],
    "T3-C3": ["EEG T3-REF", "EEG C3-REF"],
    "C3-CZ": ["EEG C3-REF", "EEG CZ-REF"],
    "CZ-C4": ["EEG CZ-REF", "EEG C4-REF"],
    "C4-T4": ["EEG C4-REF", "EEG T4-REF"],
    "FP1-F3": ["EEG FP1-REF", "EEG F3-REF"],
    "F3-C3": ["EEG F3-REF", "EEG C3-REF"],
    "C3-P3": ["EEG C3-REF", "EEG P3-REF"],
    "P3-O1": ["EEG P3-REF", "EEG O1-REF"],
    "FP2-F4": ["EEG FP2-REF", "EEG F4-REF"],
    "F4-C4": ["EEG F4-REF", "EEG C4-REF"],
    "C4-P4": ["EEG C4-REF", "EEG P4-REF"],
    "P4-O2": ["EEG P4-REF", "EEG O2-REF"]
}

TCP_MONTAGE_04_TCP_LE_A = {
    "FP1-F7": ["EEG FP1-LE", "EEG F7-LE"],
    "F7-T3": ["EEG F7-LE", "EEG T3-LE"],
    "T3-T5": ["EEG T3-LE", "EEG T5-LE"],
    "T5-O1": ["EEG T5-LE", "EEG O1-LE"],
    "FP2-F8": ["EEG FP2-LE", "EEG F8-LE"],
    "F8-T4": ["EEG F8-LE", "EEG T4-LE"],
    "T4-T6": ["EEG T4-LE", "EEG T6-LE"],
    "T6-O2": ["EEG T6-LE", "EEG O2-LE"],
    "T3-C3": ["EEG T3-LE", "EEG C3-LE"],
    "C3-CZ": ["EEG C3-LE", "EEG CZ-LE"],
    "CZ-C4": ["EEG CZ-LE", "EEG C4-LE"],
    "C4-T4": ["EEG C4-LE", "EEG T4-LE"],
    "FP1-F3": ["EEG FP1-LE", "EEG F3-LE"],
    "F3-C3": ["EEG F3-LE", "EEG C3-LE"],
    "C3-P3": ["EEG C3-LE", "EEG P3-LE"],
    "P3-O1": ["EEG P3-LE", "EEG O1-LE"],
    "FP2-F4": ["EEG FP2-LE", "EEG F4-LE"],
    "F4-C4": ["EEG F4-LE", "EEG C4-LE"],
    "C4-P4": ["EEG C4-LE", "EEG P4-LE"],
    "P4-O2": ["EEG P4-LE", "EEG O2-LE"]
}

TCP_MONTAGES = {
    '01_tcp_ar': TCP_MONTAGE_01_TCP_AR,
    '02_tcp_le': TCP_MONTAGE_02_TCP_LE,
    '03_tcp_ar_a': TCP_MONTAGE_03_TCP_AR_A,
    '04_tcp_le_a': TCP_MONTAGE_04_TCP_LE_A
}

# Define the standard channel order that should come first
standard = {
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF',  
    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',  
    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',  
    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF'
}
standard_LE = {
    'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE',
    'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE',
    'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE', 'EEG A1-LE', 'EEG A2-LE',
    'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE', 'EEG T1-LE', 'EEG T2-LE'
}

custom_pos_no_signal = {
    f'EEG {num}-REF' for num in range(33, 129)
}

custom_pos = {
    'EEG 1X10_LAT_01-', 'EEG 1X10_LAT_02-', 'EEG 1X10_LAT_03-', 'EEG 1X10_LAT_04-', 'EEG 1X10_LAT_05-', 
    'EEG 20-LE', 'EEG 20-REF', 'EEG 21-LE', 'EEG 21-REF', 'EEG 22-LE', 'EEG 22-REF', 'EEG 23-LE', 
    'EEG 23-REF', 'EEG 24-LE', 'EEG 24-REF', 'EEG 25-LE', 'EEG 25-REF', 'EEG 26-LE', 'EEG 26-REF', 
    'EEG 27-LE', 'EEG 27-REF', 'EEG 28-LE', 'EEG 28-REF', 'EEG 29-LE', 'EEG 29-REF', 'EEG 30-LE', 
    'EEG 30-REF', 'EEG 31-LE', 'EEG 31-REF', 'EEG 32-LE', 'EEG 32-REF', 'EEG X1-REF'
}

head_loc_pos = {
    'EEG C3-P3', 'EEG C3P-REF',  
    'EEG C3-T3', 'EEG C4-CZ', 'EEG C4-P4', 'EEG C4P-REF',  'EEG CZ-C3', 
    'EEG CZ-PZ',  'EEG F3-C3',  'EEG F4-C4',  
  'EEG F7-T3',  'EEG F8-T4', 'EEG FP1-F7', 
  'EEG FP2-F8',   'EEG FZ-CZ',
     'EEG OZ-LE', 'EEG OZ-REF',  
     'EEG SP1-LE', 'EEG SP1-REF', 'EEG SP2-LE', 'EEG SP2-REF', 
    'EEG T1-T2',  'EEG T2-T4',  
    'EEG T3-T1', 'EEG T3-T5', 'EEG T4-C4',  'EEG T4-T6',  'EEG T5-O1', 
     'EEG T6-O2',  
}

ecg_pos = {'ECG EKG-REF', 'EEG EKG1-REF', 'EEG EKG-LE', 'PULSE RATE'}

resp_pos = {'EEG RESP1-REF', 'EEG RESP2-REF', 'RESP ABDOMEN-REF'}
# The full union of all possible channels
standard_order = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "A1-T3", "T4-A2"]

def make_bipolar(data, channels, montage):
    """
    Takes in a mne raw object and returns a new mne raw object with unipolar channels
    and converts it into TPC bipolar montage. And returns the data.
    """
    TPC_bipolar_montage = TCP_MONTAGES[montage]
    new_data = []
    new_channels = []

    for key in TPC_bipolar_montage.keys():
        ch1, ch2 = TPC_bipolar_montage[key]
        if ch1 in channels and ch2 in channels:
            new_data.append(data[channels.index(ch1)] - data[channels.index(ch2)])
            new_channels.append(key)
        
    new_data = np.array(new_data)

    return new_data, new_channels

# Combine the standard channels with the rest of the union in order
all_channels_union = list(standard) + list(standard_LE)

sampling_freq=256
slice_duration=5
pad_up_to_max_chans = None

montage_name_to_idx = {'01_tcp_ar':0, '02_tcp_le':1, '03_tcp_ar_a':2, '04_tcp_le_a': 3}

def create_hdf5(data_slices, channel_count, target_dir, data_group_name, file_idx, channel_info, montage_name):
    data_group = data_slices # np array of size (num_slices, num_channels, num_samples)
    target_path = os.path.join(target_dir, f"TUEG_{channel_count}_channels_{file_idx}.h5")

    with h5py.File(target_path, "a") as h5f:
        if h5f.attrs.get("channel_names") is None:
            h5f.attrs["channel_names"] = channel_info
        grp = h5f.create_group(data_group_name)#f"data_group_{i // group_size}")
        X_data = np.array(data_group, dtype=np.float16)
        grp.create_dataset("X", data=X_data, dtype='float16')
        grp.attrs["montage_name"] = montage_name
    #print(f"Saved {len(data_slices)} samples with {channel_count} channels to {target_path}")

def get_channel_count_from_file(file_path):
    try:
        #raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        #raw.drop_channels(ch_names=[ch for ch in raw.ch_names if ch not in all_channels_union])
        #channels = raw.get_data().shape[0]
        montage_name = str(file_path.parts[-2])
        channels = len(TCP_MONTAGES[montage_name])
        return channels, file_path
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_montage_name(file_path):
    return str(file_path.parts[-2])

def get_montage_names_parallel(file_paths, num_workers=4):
    montages_count = {}
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(get_montage_name, file_paths), total=len(file_paths)))
    for result in results:
        if result not in montages_count:
            montages_count[result] = 0
        montages_count[result] += 1

    # Plot the counts for each key in montages_count
    plt.figure(figsize=(10, 6))
    plt.bar(montages_count.keys(), montages_count.values())
    plt.xlabel('Montage Name')
    plt.ylabel('Count')
    plt.title('Counts of Each Montage Name')
    plt.xticks(rotation=90)
    plt.tight_layout()
    # Save the plot to a file
    plot_file_path = os.path.join(os.path.dirname(__file__), 'montages_count_plot.png')
    print(montages_count)
    plt.savefig(plot_file_path)
    print(f"Montage count plot saved to {plot_file_path}")
    
def get_channel_names_from_file(file_path):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        return raw.ch_names
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_channel_names_parallel(file_paths, num_workers=4):
    channel_names = {}
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(get_channel_names_from_file, file_paths), total=len(file_paths)))
    for result in results:
        if result is not None:
            for name in result:
                if name not in channel_names:
                    channel_names[name] = 0
                channel_names[name] += 1
    plt.figure(figsize=(10, 6))
    plt.bar(channel_names.keys(), channel_names.values())
    plt.xlabel('Channel Name')
    plt.ylabel('Count')
    plt.title('Counts of Each Channel Name')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_file_path = os.path.join(os.path.dirname(__file__), 'channel_names_count_plot.png')
    plt.savefig(plot_file_path)
    print(f"Channel names count plot saved to {plot_file_path}")

def group_samples_by_channel_count_parallel(file_paths, num_workers=4):
    """
    Groups samples by their channel count using parallel processing.
    """
    grouped_samples = {}
    with multiprocessing.Pool(num_workers) as pool:
        # Use `imap_unordered` for parallel processing with progress bar
        results = list(tqdm(pool.imap_unordered(get_channel_count_from_file, file_paths), total=len(file_paths)))
    
    # Aggregate results
    for result in results:
        if result is not None:
            channels, file_path = result
            if channels not in grouped_samples:
                grouped_samples[channels] = []
            grouped_samples[channels].append(file_path)

    with open('/home/msc24h11/grouped_samples3.pkl', 'wb') as f:
        pickle.dump(grouped_samples, f)
    return grouped_samples

def save_files_to_hdf5_parallel(file_paths, channel_count, output_dir, tuab_subjects, file_idx, worker_id):
    with tqdm(total=len(file_paths), desc=f"Worker {worker_id}", position=worker_id) as pbar:
        for file_path in file_paths:
            try:
                session_name = file_path.parts[-1][:-4]
                subject_name = session_name.split('_s')[0]
                montage_name = str(file_path.parts[-2])
                if subject_name in tuab_subjects:
                    #print(f"Skipping this edf file because it corresponds to TUAB subject {subject_name}")
                    continue
                
                # print(session_name)
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False, include=all_channels_union)
                chn_order = standard if montage_name in ['01_tcp_ar', '03_tcp_ar_a'] else standard_LE
                raw.reorder_channels(ch_names=[ch for ch in chn_order if ch in raw.ch_names])

                #raw.reorder_channels(ch_names=[ch for ch in all_channels_union if ch in raw.ch_names])
                raw.filter(l_freq=0.1, h_freq=75.0, verbose="ERROR")
                raw.notch_filter(60,  verbose="ERROR")

                
                if raw.info['sfreq'] != sampling_freq:
                    raw.resample(sampling_freq)
                data = raw.get_data(units='uV')

                data, channel_info = make_bipolar(data, raw.ch_names, montage_name)

                n_channels, n_times = data.shape
                assert n_channels == channel_count, (
                    f"Channel count mismatch in file {file_path}: "
                    f"expected {channel_count}, but got {n_channels}"
                )
                interval_size = sampling_freq*slice_duration
                num_intervals = n_times // interval_size
                
                new_sliced_data = data[:, :num_intervals * interval_size].reshape(num_intervals, n_channels, interval_size)
                # pad here to pad_up_to_max_chans
                if pad_up_to_max_chans is not None:
                    num_real_chans = new_sliced_data.shape[0]
                    if num_real_chans < pad_up_to_max_chans:
                        to_pad = pad_up_to_max_chans - num_real_chans
                        # assert to_pad > 0                
                        new_sliced_data = new_sliced_data.T
                        new_sliced_data = np.pad(new_sliced_data, (0, to_pad), 'constant', constant_values=0.0).T
                    channel_mask = np.zeros(pad_up_to_max_chans)
                    channel_mask[:num_real_chans] = 1
                create_hdf5(new_sliced_data, channel_count, target_dir=output_dir, data_group_name=session_name, file_idx=file_idx, channel_info=channel_info, montage_name=montage_name)
                
            except Exception as e:
                print(f'error happened at file path:{file_path}, {e}')
            pbar.update(1)
            

# Function to retrieve TUAB subject names
def get_tuab_subjects(tuab_path):
    tuab_edf_files = [os.path.basename(file) for file in Path(tuab_path).rglob('*.edf')]
    tuab_subjects = {os.path.basename(f).split('_s')[0] for f in tuab_edf_files}
    return tuab_subjects

def parallel_save_files_to_hdf5(file_paths, output_dir, tuab_subjects, max_sessions_per_file=2000):
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} workers")
    if os.path.exists('#CHANGEME'):
        with open('CHANGEME', 'rb') as f:
            grouped_samples = pickle.load(f)
    else:
        grouped_samples = group_samples_by_channel_count_parallel(file_paths, num_workers)
    print(f"Grouped samples by channel count: {grouped_samples.keys()}")
    # Prepare arguments for parallel processing
    args = []
    for channel_count, file_paths in grouped_samples.items():
        for i in range(0, len(file_paths), max_sessions_per_file):
            args.append((file_paths[i:i + max_sessions_per_file], channel_count, output_dir, tuab_subjects, int(i / max_sessions_per_file)))
    print(f"Processing {len(args)} groups of files")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use multiprocessing to parallelize the process
    with multiprocessing.Manager() as manager:
    # Use tqdm to track progress
          with multiprocessing.Pool(num_workers) as pool:
            # We need to pass the worker id to each worker for unique progress bars
            pool.starmap(save_files_to_hdf5_parallel, [(arg[0], arg[1], arg[2], arg[3], arg[4], worker_id) for worker_id, arg in enumerate(args)])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', default='#CHANGEME/TUEG/v2.0.0/')
    parser.add_argument('--hdf5_file_path', default='#CHANGEME/TUEG/')
    
    
    tuab_path = "#CHANGEME/TUAB/"
    tuab_subjects = get_tuab_subjects(tuab_path)
    print(f"Found {len(tuab_subjects)} TUAB subjects")
    args = parser.parse_args()
    # root_path = args.root_path
    in_path = args.in_path
    hdf5_file_path = args.hdf5_file_path

    hdf5_file_path = Path(hdf5_file_path)
    full_in_path = Path(in_path)
    file_paths =[f for f in full_in_path.rglob('*.edf')]
    print(f"Found {len(file_paths)} edf files in {full_in_path}")
    #get_channel_names_parallel(file_paths, multiprocessing.cpu_count())
    # Save all slices to HDF5 using parallel processing
    parallel_save_files_to_hdf5(file_paths, hdf5_file_path, tuab_subjects)
    