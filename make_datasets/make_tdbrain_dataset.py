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

## TDBRAIN-specific constants

TDBRAIN_CH_ORDER = ['Fp1', 'Fp2', 
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC3', 'FCz', 'FC4', 
    'T7', 'C3', 'Cz', 'C4', 'T8', 
    'CP3', 'CPz', 'CP4', 
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2']

TDBRAIN_LABEL_MAP = {'Healthy': 0, 'Parkinson': 1}

ORIGINAL_SFREQ = 500 # Hz

RESAMPLE_SFREQ = 256 # Hz

WINDOW_SEC = 1.25 # seconds

## Dataset-specific processing functions

def raw_from_csv(df, sfreq):

    ch_names = list(df.columns)
    ch_types = []

    for ch in ch_names:
        if ch in ['VPVA', 'VNVB']:
            ch_types.append('eog')
        elif ch in ['HPHL', 'HNHR', 'Erbs', 'OrbOcc', 'Mass']:
            ch_types.append('emg')  # or 'misc'
        else:
            ch_types.append('eeg')

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    data = df.to_numpy().T  # (n_channels, n_times)

    raw = mne.io.RawArray(data, info, verbose=False)
    
    return raw

def preprocess_raw(raw, target_sfreq):
    # Set montage
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        match_alias={'cb1': 'POO7', 'cb2': 'POO8'}
    )

    # Reorder channels
    raw.pick_channels(TDBRAIN_CH_ORDER, ordered=True)

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
    ## 1. Metadata collection
    data_splits, labels_dictionary = collect_metadata(prepath)

    ## 2. Processing loop
    out_directory = os.path.join(output_path, "TDBrain_dataset_Parkinson_bc_1.25s/")
    raw_data_directory = os.path.join(prepath, "derivatives")

    splits = ['test', 'train', 'val']
    for split in splits:

        subjects = [labels_dictionary[i]['participant_id'] for i in data_splits[split]]
        labels = [labels_dictionary[i]['label'] for i in data_splits[split]]
        print(f"{split.capitalize()} subjects:", subjects)
        print(f"{split.capitalize()} labels:", labels)

        for participant_id, label in tqdm(zip(subjects, labels), desc=f"Processing {split} set"):
            
            # get dataframe
            subject_path = os.path.join(raw_data_directory, participant_id, 'ses-1', 'eeg', f"{participant_id}_ses-1_task-restEC_eeg.csv")
            if not os.path.exists(subject_path):
                print(f"File not found: {subject_path}")
                continue
            df = pd.read_csv(subject_path)
            
            # get raw object
            raw = raw_from_csv(df, sfreq=ORIGINAL_SFREQ)

            # run minimal preprocessing
            preprocessed = preprocess_raw(raw, target_sfreq=RESAMPLE_SFREQ)

            # segment sample into windows
            segments = segment_raw(preprocessed, window_sec=WINDOW_SEC)

            h5_path = os.path.join(out_directory, f"{split}.h5")
            
            # append segments to h5 file
            write_segments_to_h5_append(
                h5_path,
                segments,
                label=label,
                finetune=True,
                group_size=1000
            )
    print("Preprocessing complete. HDF5 files saved to:", output_path)

def collect_metadata(prepath):
    # read participants_metadata.csv
    participants_metadata_tsv_path = os.path.join(prepath, "TDBRAIN_participants_V2.tsv")
    participants_metadata_df = pd.read_csv(participants_metadata_tsv_path, sep='\t')
    
    # get the list of participant ids in eyes-closed condition
    mask_formal_status_EC_healthy = (participants_metadata_df['indication'] == 'HEALTHY') & (participants_metadata_df['EC'] == 1.0)
    healthy_patients_df = participants_metadata_df[mask_formal_status_EC_healthy]
    list_of_healthy_patients_EC = healthy_patients_df['participants_ID'].to_list()
    print("Length of Healthy patients with EC", len(list_of_healthy_patients_EC))

    # parkinson's disease patients in eyes-closed condition
    mask_formal_status_EC_pd = (participants_metadata_df['indication'] == 'PARKINSON') & (participants_metadata_df['EC'] == 1.0)
    pd_patients_df = participants_metadata_df[mask_formal_status_EC_pd]
    list_of_pd_patients_EC_indication_status = pd_patients_df['participants_ID'].to_list()
    print("Length of Parkinson's patients with EC", len(list_of_pd_patients_EC_indication_status))    

    # deduplicate the lists
    parkinsons_subjects = list(set(list_of_pd_patients_EC_indication_status))
    healthy_subjects = list(set(list_of_healthy_patients_EC))

    # Assign BioMamba IDs
    parkinsons_subjects_BioMamba_id = [i+1 for i in range(len(parkinsons_subjects))]
    healthy_subjects_BioMamba_id = [i+1 for i in range(25, 25+len(healthy_subjects))]

    # create a dictionary mapping BioMamba ID to participant subject ID
    labels_dictionary = {}
    for i, subject in enumerate(parkinsons_subjects):
        labels_dictionary[parkinsons_subjects_BioMamba_id[i]] = {'participant_id': subject, 'label': 1}
    for i, subject in enumerate(healthy_subjects):
        labels_dictionary[healthy_subjects_BioMamba_id[i]] = {'participant_id': subject, 'label': 0}

    # Replicating the BioMamba IDs needed for the dataset splits
    validation_set = [18, 19, 20, 21, 46, 47, 48, 49]
    test_set = [22, 23, 24, 25, 50, 51, 52, 53]
    train_set = list(range(1, 18)) + list(
                range(29, 46)
            )
    
    # create a dictionary for the splits
    data_splits = {
        'train': train_set,
        'val': validation_set,
        'test': test_set
    }

    print("Train set IDs:", len(train_set), "subjects", train_set)
    print("Validation set IDs:", len(validation_set), "subjects", validation_set)
    print("Test set IDs:", len(test_set), "subjects", test_set)

    return data_splits, labels_dictionary


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