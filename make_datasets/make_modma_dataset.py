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

## MODMA-specific constants

MODMA_CHN_ORDER = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10',
                    'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20',
                      'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30',
                        'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40',
                          'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50',
                            'E51', 'E52', 'E53', 'E54', 'E55', 'E56', 'E57', 'E58', 'E59', 'E60', 
                            'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69', 'E70',
                              'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80',
                                'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88', 'E89', 'E90',
                                  'E91', 'E92', 'E93', 'E94', 'E95', 'E96', 'E97', 'E98', 'E99', 'E100',
                                    'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110', 
                                    'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120',
                                      'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'Cz'] # Cz is reference, will be dropped later

MODMA_LABEL_MAP = {'H': 0, 'MDD': 1}

ORIGINAL_SFREQ = 250 # Hz

RESAMPLE_SFREQ = 250 # Hz; preserved

WINDOW_SEC = 5.12 # seconds

## Dataset-specific processing functions

def raw_from_mat(mat_path):
    mat_data = sio.loadmat(mat_path)
    # get the first key that contains 'rest' (to avoid keys like '__header__', '__version__', etc.)
    key = next(k for k in mat_data.keys() if 'rest' in k or 'mat' in k)  # added 'mat' to catch keys that might not have 'rest' but are still relevant
    data = mat_data[key] # (n_channels, n_times)
    info = mne.create_info(ch_names=MODMA_CHN_ORDER, sfreq=ORIGINAL_SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # drop reference Cz
    raw.drop_channels(['Cz'])
    return raw

def preprocess_raw(raw):
    # Set montage (use GSN-HydroCel-129 montage)
    raw.set_montage(
        mne.channels.make_standard_montage('GSN-HydroCel-129')
    )

    # Bandpass
    raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)

    # Notch (Europe = 50 Hz, US = 60 Hz)
    raw.notch_filter(freqs=[50], verbose=False)

    return raw

## Main processing functions

def process_and_convert_to_h5(prepath, output_path):
    # 1. Collect metadata and create data splits
    split_dict = collect_metadata(prepath)

    # 2. Processing loop
    out_directory = os.path.join(output_path, "MODMA_128_channels_bc_5.12s/")
    raw_data_directory = prepath

    for split in split_dict.keys():
        for condition in ['MDD', 'H']:
            files = [filename for filename, c_id, s_id in split_dict[split][condition]] # get the list of files for this split and condition
            label = MODMA_LABEL_MAP[condition]
          
            for file in files:
                # get raw path
                print("Processing file:", file)
                raw_path = os.path.join(raw_data_directory, file)
                
                # read raw data from dataframe
                raw = raw_from_mat(raw_path)
                # preprocess data
                preprocessed = preprocess_raw(raw)
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

    filenames = [f for f in os.listdir(prepath) if f.endswith('.mat')]

    subject_dict = {'MDD': [], 'H': []}
    for filename in filenames:
        condition_id = filename[:4]
        subject_id = filename[:8]
        subject_id_without_condition = int(subject_id[4:])  # remove the first 4 characters (condition ID)
        if condition_id == '0201':
            subject_dict['MDD'].append((filename, subject_id, subject_id_without_condition))
        else:
            subject_dict['H'].append((filename, subject_id, subject_id_without_condition))

    # print len of each condiiton
    for condition, subjects in subject_dict.items():
        print(f"{condition}: {len(subjects)} subjects")
    # Should see: MDD: 24 subjects, H: 29 subjects

    # sort by subject ID (numerically)
    for condition in subject_dict:
        subject_dict[condition] = sorted(subject_dict[condition], key=lambda x: x[2])
    
    training_subjects = {
        'MDD': subject_dict['MDD'][:14], # 14 depressed subjects for training 
        'H': subject_dict['H'][:15] # 15 healthy subjects for training
    }

    # val takes 
    validation_subjects = {
        'MDD': subject_dict['MDD'][14:19], # 5 subjects for validation
        'H': subject_dict['H'][15:22] # 7 subjects for validation
    }

    test_subjects = {
        'MDD': subject_dict['MDD'][19:], # 5 subjects for testing 
        'H': subject_dict['H'][22:] # 7 subjects for testing
    }

    split_dict = {
    'train': training_subjects,
    'val': validation_subjects,
    'test': test_subjects}

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