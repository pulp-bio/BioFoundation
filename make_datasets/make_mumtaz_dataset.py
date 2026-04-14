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

## Mumtaz2016-specific constants

MUMTAZ_CHN_ORDER = ['Fp1', 'F3', 'C3', 'P3', 'O1',
 'F7','T3', 'T5', 'Fz', 'Fp2',
 'F4','C4','P4','O2','F8',
 'T4','T6','Cz', 'Pz']

MUMTAZ_LABEL_MAP = {'H': 0, 'MDD': 1}

ORIGINAL_SFREQ = 256 # Hz

RESAMPLE_SFREQ = 256 # Hz

WINDOW_SEC = 5 # seconds

## Dataset-specific processing functions

def raw_from_edf(edf_path, preload=True):
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)

    # change channel names
    # remove "EEG " at start, then remove "-LE" at end if it exists
    new_ch_names = []
    for ch in raw.info['ch_names']:
        if ch.startswith("EEG "):
            ch = ch.replace("EEG ", "")
        if ch.endswith("-LE"):
            ch = ch.replace("-LE", "")
        new_ch_names.append(ch)
    # remove '23A-23R' and '24A-24R'
    raw.rename_channels({old: new for old, new in zip(raw.info['ch_names'], new_ch_names)})
    if '23A-23R' in raw.info['ch_names'] or '24A-24R' in raw.info['ch_names']:
        raw.drop_channels(['23A-23R', '24A-24R'])
    if "A2-A1" in raw.info['ch_names']:
        raw.drop_channels(['A2-A1'])
    
    return raw

def preprocess_raw(raw, target_sfreq):
    # Set montage
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        match_alias={'cb1': 'POO7', 'cb2': 'POO8'}
    )

    # Reorder channels
    raw.pick_channels(MUMTAZ_CHN_ORDER, ordered=True)

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
    
    # 1. Collect metadata and create data splits
    split_dict = collect_metadata(prepath)

    # 2. Process each split and write to HDF5
    out_directory = os.path.join(output_path, "Mumtaz_MDD_bc_5s/")
    raw_data_directory = prepath # with subject level information

    for split in split_dict.keys():
        for condition in ['MDD', 'H']:
            edf_files = split_dict[split][condition]
            label = MUMTAZ_LABEL_MAP[condition]
            print("Condition is {}, with numerical label: {}".format(condition, label))
            # print number of EDF files in this split and condition
            print(f"Number of EDF files in split {split} and condition {condition}: {len(edf_files)}")

            for edf_file in edf_files[:]:
                # get raw path
                raw_path = os.path.join(raw_data_directory, edf_file)
                
                # read raw data from EDF file
                raw = raw_from_edf(raw_path, preload=True)

                # preprocess data
                preprocessed = preprocess_raw(raw, target_sfreq=RESAMPLE_SFREQ)

                # segment into windows
                segments = segment_raw(preprocessed, window_sec=WINDOW_SEC)

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
    # find all .edf files in the data directory
    edf_files = [f for f in os.listdir(prepath) if f.endswith('.edf')]
    print(f"Found {len(edf_files)} .edf files in {prepath}")

    subjects = []
    for f in edf_files:
        parts = f.split(' ') # tolerance for extra spaces in the filename

        subject_type = parts[0]
        subject_id = parts[1]
        session_type = parts[2].split('.')[0]  # remove .edf extension

        # if "H" is in subject_type, replace it with "Healthy", if "MDD" is in subject_type, replace it with "MDD"
        if "H" in subject_type:
            subject_type = "H"
        elif "MDD" in subject_type:
            subject_type = "MDD"

        # filter eyes-open and eyes-closed sessions
        if session_type == "EO" or session_type == "EC": 
            # print(f"Subject type: {subject_type}, Subject ID: {subject_id}, Session type: {session_type}")
            subjects.append((f, subject_type, subject_id, session_type))

    depressed_subjects = list(set([s[2] for s in subjects if s[1] == "MDD"]))
    healthy_subjects = list(set([s[2] for s in subjects if s[1] == "H"]))
    
    num_depressed = len(depressed_subjects)
    num_healthy = len(healthy_subjects)

    print(f"Number of depressed subjects: {num_depressed}") # (!!!) Important: should be 33 (it is 34 including all task types, but we only include EC and EO)
    print(f"Number of healthy subjects: {num_healthy}") #(!!!) Important: should be 30
    # (!!!) Important: rename files to have consistent format: {subject_id}_{session_type}_{subject_type}.edf if not already in that format

    # sorted by number after "S"
    depressed_subjects = sorted(list(set([s[2] for s in subjects if s[1] == "MDD"])), key=lambda x: int(x.split('S')[1]))
    healthy_subjects = sorted(list(set([s[2] for s in subjects if s[1] == "H"])), key=lambda x: int(x.split('S')[1]))

    # create data splits
    train_subjects = {"MDD": depressed_subjects[:23], "H": healthy_subjects[:19]} # 23 MDD and 19 NC
    val_subjects = {"MDD": depressed_subjects[23:28], "H": healthy_subjects[19:23]} # 5 MDD and 4 NC
    test_subjects = {"MDD": depressed_subjects[28:], "H": healthy_subjects[23:27]} # 5 MDD and 4 NC

    # find files corresponding to each split
    # find number of files in each split
    train = {"MDD": [s[0] for s in subjects if s[1] == "MDD" and s[2] in train_subjects['MDD']], 
             "H": [s[0] for s in subjects if s[1] == "H" and s[2] in train_subjects['H']]}
    val = {"MDD": [s[0] for s in subjects if s[1] == "MDD" and s[2] in val_subjects['MDD']], 
           "H": [s[0] for s in subjects if s[1] == "H" and s[2] in val_subjects['H']]}
    test = {"MDD": [s[0] for s in subjects if s[1] == "MDD" and s[2] in test_subjects['MDD']], 
            "H": [s[0] for s in subjects if s[1] == "H" and s[2] in test_subjects['H']]}
    
    split_dict = {
    'train': train,
    'val': val,
    'test': test}

    return split_dict

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