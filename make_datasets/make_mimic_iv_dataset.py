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
import wfdb
import subprocess
import multiprocessing
import h5py

from process_raw_ecg import preprocess_signal, time_segmenting

def dump_data(data_slice, output_dir, data_group_name, file_idx, channel_info):
    output_path = os.path.join(output_dir, f"MIMIC-IV-ECG_12_channels_{file_idx}.h5")
    with h5py.File(output_path, "a") as h5f:
        if h5f.attrs.get("channel_name") is None:
            h5f.attrs['channel_names'] = channel_info
        grp = h5f.create_group(data_group_name)
        X_data = np.array(data_slice, dtype=np.float16)
        grp.create_dataset("X", data=X_data, dtype='float16')

def process_single_file(file_paths, output_dir, downsample_fs, split_signal, file_idx, worker_id, batch_size=300):
    batch_data = []
    batch_idx = 0
    channel_info = None

    def flush_batch():
        nonlocal batch_idx
        if not batch_data:
            return
        concatenated = np.concatenate(batch_data, axis=0)
        group_name = f"{worker_id}_{batch_idx}"
        dump_data(concatenated, output_dir, group_name, file_idx, channel_info)
        batch_data.clear()
        batch_idx += 1

    with tqdm.tqdm(total=len(file_paths), desc=f"Worker {worker_id}", position=worker_id) as pbar:
        for file_path in file_paths:
            record_path = file_path.split('.')[0]
            record = wfdb.rdrecord(record_path)

            raw_ecg = np.array(record.p_signal).T
            sampling_rate = record.fs
            channel_info = record.sig_name

            processed_ecg = preprocess_signal(raw_ecg, sampling_rate, 0.5, 120, downsample_fs, None)

            p_min = processed_ecg.min()
            p_max = processed_ecg.max()
            if (p_max - p_min) == 0:
                pbar.update(1)
                continue

            processed_ecg = (processed_ecg - p_min) / (p_max - p_min)
            merged_splits = processed_ecg.reshape(1, processed_ecg.shape[0], processed_ecg.shape[1])

            if split_signal is not None:
                signal_splitted = time_segmenting(merged_splits, split_signal, sampling_rate, downsample_fs, None)
                merged_splits = np.concatenate(signal_splitted, axis=0)

            batch_data.append(merged_splits)

            if len(batch_data) >= batch_size:
                flush_batch()

            pbar.update(1)

        flush_batch()  # flush remaining records that didn't fill a full batch

def loading_and_processing_parallel(files, output_dir, downsampling_fs, split_signal, max_sessions_per_file=30000):
    num_workers = 27
    worker_args = []

    print(max_sessions_per_file)
    for i in range(0, len(files), max_sessions_per_file):
        worker_args.append((files[i:i+max_sessions_per_file], output_dir, downsampling_fs, split_signal, int(i / max_sessions_per_file)))

    print(f"Processing {len(worker_args)} groups of files:")
    os.makedirs(output_dir, exist_ok=True)

    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(process_single_file, [
                (args[0], args[1], args[2], args[3], args[4], worker_id, 300)
                for worker_id, args in enumerate(worker_args)
            ])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process MIMIC-IV-ECG .hea and .dat files and save as .h5 files.")
    parser.add_argument('--prepath', type=str, default='#CHANGEME', help='Directory containing MIMIC-IV-ECG files.')
    parser.add_argument('--output_path', type=str, default='#CHANGEME', help='Directory to save processed .h5 files.')
    parser.add_argument('--collapse_pickle', type=str, default='#CHANGEME', help='Pickle file with collected samples for faster reading.')
    parser.add_argument('--downsampling_fs', type=int, default=256, help='Desired downsampling frequency.')
    parser.add_argument('--split_signal', type=int, default=None, help='Length of segments to split the signals into (in seconds). If None, no splitting is done.')

    args = parser.parse_args()

    if os.path.exists(args.collapse_pickle):
        with open(args.collapse_pickle, 'rb') as f:
            files = pickle.load(f)
    else:
        # If you run for the first time, this will create the pickle file 
        result = subprocess.run(["find", args.prepath, "-maxdepth", str(5), "-type", "f", "-name", "*.hea"], stdout=subprocess.PIPE, text=True)
        files = result.stdout.splitlines()
        with open(args.collapse_pickle, 'wb') as f:
            pickle.dump(files, f)

    loading_and_processing_parallel(files, args.output_path, args.downsampling_fs, args.split_signal)
