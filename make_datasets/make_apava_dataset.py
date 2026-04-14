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
from util.preprocess_utils import segment_raw, write_segments_to_h5_append
import argparse
import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io as sio

## APAVA-specific constants

APAVA_CH_ORDER = ['C3',
 'C4',
 'F3',
 'F4',
 'F7',
 'F8',
 'Fp1',
 'Fp2',
 'O1',
 'O2',
 'P3',
 'P4',
 'T3',
 'T4',
 'T5',
 'T6']

APAVA_LABEL_MAP = {'Healthy': 0, 'Alzheimer': 1}

ORIGINAL_SFREQ = 256 # Hz

RESAMPLE_SFREQ = 256 # Hz

WINDOW_SEC = 5 # seconds

## Dataset-specific processing functions

def raw_from_np(np_array, sfreq):
    
    # pick only channels in APAVA_CH_ORDER
    ch_names = [name for name in APAVA_CH_ORDER]
    ch_types = ['eeg'] * len(ch_names)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    raw = mne.io.RawArray(np_array, info, verbose=False)
    
    return raw

def preprocess_raw(raw, target_sfreq):
    # Set montage
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        match_alias={'cb1': 'POO7', 'cb2': 'POO8'}
    )

    # Bandpass
    raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)

    # Notch (Europe = 50 Hz, US = 60 Hz)
    raw.notch_filter(freqs=[50], verbose=False)

    # Resample
    if int(raw.info['sfreq']) != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose=False)

    return raw

## Main processing functions

def process_and_convert_to_h5(prepath, output_path):
    # 1. Collect metadata
    data_dict, label_dict, split_dict = collect_metadata(prepath)

    # 2. Processing loop
    out_directory = os.path.join(output_path, "APAVA_Alzheimer_bc_5s/")

    splits = ['test', 'train', 'val']
    for split in splits:
        subjects = split_dict[split]
        labels = [label_dict[i] for i in split_dict[split]]

        for participant_id, label in zip(subjects, labels):
            # get session's data from data_dict
            subject_data = data_dict[participant_id]  # shape (n_sessions, n_channels, n_times)
                
            # iterate through sessions
            segments = []
            for session_idx in range(subject_data.shape[0]):
                session_sample = subject_data[session_idx]  # shape (n_channels, n_times)

                # read raw data from dataframe
                raw = raw_from_np(session_sample, sfreq=ORIGINAL_SFREQ)
                # preprocess raw data
                preprocessed = preprocess_raw(raw, target_sfreq=RESAMPLE_SFREQ)
                # segment and store segments
                segments.append(segment_raw(preprocessed, window_sec=WINDOW_SEC))

            segments = np.concatenate(segments, axis=0)  # shape (n_total_segments, n_channels, n_times)

            h5_path = os.path.join(out_directory, f"{split}.h5")
            
            write_segments_to_h5_append(
                h5_path,
                segments,
                label=label,
                finetune=True,
                group_size=1000
            )

        print("Preprocessing complete. HDF5 files saved to:", output_path)

def collect_metadata(prepath):
    # 1. Collect data
    filenames = []
    for filename in os.listdir(prepath):
        filenames.append(filename)

    data_dict = {}
    for i in range(len(filenames[:])):
        path = os.path.join(prepath, filenames[i])
        mat = sio.loadmat(path)

        # Get epoch/session number for each subject
        data = mat['data'][0, 0]
        channel_names = data['label']
        channel_names = [lbl[0] for lbl in channel_names.squeeze()]
        
        sessions = len(data[2][0])
        temp = np.zeros((sessions, 16, 1280)) # 16 channels, 1280 samples (5 seconds at 256 Hz)

        # Store in temp
        for j in range(sessions):
            temp[j] = data[2][0][j]

        participant_id = i+1
        data_dict[participant_id] = temp

    # 2. Defining labels
    AD_positive = [1,3,6,8,9,11,12,13,15,17,19,21]

    # Add label 1 for AD positive, 0 for healthy
    label_dict = {}
    for participant_id in data_dict.keys():
        if participant_id in AD_positive:
            label_dict[participant_id] = 1
        else:
            label_dict[participant_id] = 0

    # 3. Define splits
    val = [15, 16, 19, 20]
    test = [1, 2, 17, 18]
    all_ids = set(data_dict.keys())
    train = list(all_ids - set(val) - set(test))
    print("Train IDs: ", train)
    print("Validation IDs: ", val)
    print("Test IDs: ", test)

    split_dict = {
        'train': train,
        'val': val,
        'test': test
    }

    return data_dict, label_dict, split_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create HDF5 files from processed .pkl files.")
    parser.add_argument(
        "--prepath",
        type=str,
        required=True,
        help="The root directory containing the processed dataset folders (e.g., TUAR_data, TUSL_data)."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The directory where the output HDF5 files will be saved. Example: './processed_h5_files/'"
    )

    args = parser.parse_args()

    # main processing loop
    process_and_convert_to_h5(args.prepath, args.output_path)