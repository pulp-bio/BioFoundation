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
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

import os
import pickle
import argparse
from multiprocessing import Pool
import numpy as np
import mne
import tqdm

# --- Standard Channel Configuration ---
CH_ORDER_STANDARD = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
]

# --- Bipolar Montage Definition ---
# Defines the TCP bipolar montage used for re-referencing the signals.
TCP_BIPOLAR_MONTAGE = {
    "FP1-F7": ["EEG FP1-REF", "EEG F7-REF"], "F7-T3": ["EEG F7-REF", "EEG T3-REF"],
    "T3-T5": ["EEG T3-REF", "EEG T5-REF"], "T5-O1": ["EEG T5-REF", "EEG O1-REF"],
    "FP2-F8": ["EEG FP2-REF", "EEG F8-REF"], "F8-T4": ["EEG F8-REF", "EEG T4-REF"],
    "T4-T6": ["EEG T4-REF", "EEG T6-REF"], "T6-O2": ["EEG T6-REF", "EEG O2-REF"],
    "A1-T3": ["EEG A1-REF", "EEG T3-REF"], "T3-C3": ["EEG T3-REF", "EEG C3-REF"],
    "C3-CZ": ["EEG C3-REF", "EEG CZ-REF"], "CZ-C4": ["EEG CZ-REF", "EEG C4-REF"],
    "C4-T4": ["EEG C4-REF", "EEG T4-REF"], "T4-A2": ["EEG T4-REF", "EEG A2-REF"],
    "FP1-F3": ["EEG FP1-REF", "EEG F3-REF"], "F3-C3": ["EEG F3-REF", "EEG C3-REF"],
    "C3-P3": ["EEG C3-REF", "EEG P3-REF"], "P3-O1": ["EEG P3-REF", "EEG O1-REF"],
    "FP2-F4": ["EEG FP2-REF", "EEG F4-REF"], "F4-C4": ["EEG F4-REF", "EEG C4-REF"],
    "C4-P4": ["EEG C4-REF", "EEG P4-REF"], "P4-O2": ["EEG P4-REF", "EEG O2-REF"]
}

def make_bipolar(raw):
    """
    Converts a raw MNE object from a referential montage to the TCP bipolar montage.
    """
    channels = raw.ch_names
    data = raw.get_data(units='uV')
    new_data = []
    new_channels = []

    for new_ch_name, (ch1, ch2) in TCP_BIPOLAR_MONTAGE.items():
        if ch1 in channels and ch2 in channels:
            new_data.append(data[channels.index(ch1)] - data[channels.index(ch2)])
            new_channels.append(new_ch_name)

    return np.array(new_data), new_channels

def process_and_dump_file(params):
    """
    Worker function for multiprocessing. Reads an EDF file, processes it,
    and saves the segments as pickle files.
    """
    file_path, dump_folder, label, dataset_name = params
    file_name = os.path.basename(file_path)

    try:
        # --- File Loading ---
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Handle different channel naming conventions (e.g., 'LE' vs 'REF')
        if any("LE" in ch for ch in raw.ch_names):
            raw.rename_channels(lambda x: x.replace("LE", "REF"))
        
        # Ensure the raw object only contains the channels we need
        raw.pick_channels(CH_ORDER_STANDARD, ordered=True)

        if raw.ch_names != CH_ORDER_STANDARD:
            raise ValueError("Channel order mismatch after picking.")

        # --- Preprocessing ---
        raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
        raw.notch_filter(60, verbose=False)
        if int(raw.info['sfreq']) != 256:
            raw.resample(256, npad="auto", n_jobs=1, verbose=False)

        # --- Re-referencing ---
        channeled_data, channeled_channels = make_bipolar(raw)
        if len(channeled_channels) < 20:
            raise ValueError(f"Found only {len(channeled_channels)} channels after making bipolar.")

        # --- Windowing and Saving ---
        window_size_samples = 5 * 256
        for i in range(channeled_data.shape[1] // window_size_samples):
            dump_path = os.path.join(dump_folder, f"{file_name.split('.')[0]}_{i}.pkl")
            segment = channeled_data[:, i * window_size_samples: (i + 1) * window_size_samples]
            
            data_to_dump = {"X": segment}
            if label is not None:
                data_to_dump["y"] = label
            
            with open(dump_path, "wb") as f:
                pickle.dump(data_to_dump, f)

    except Exception as e:
        error_log_file = f"{dataset_name}-process-errors.txt"
        with open(error_log_file, "a") as f:
            f.write(f"Error processing {file_name}: {e}\n")

def get_tuab_parameters(root_dir, output_dir):
    """Generates the parameter list for processing the TUAB dataset."""
    print("Setting up TUAB dataset processing...")
    parameters = []
    channel_std = "01_tcp_ar"

    # Define paths
    train_abnormal_path = os.path.join(root_dir, "train", "abnormal", channel_std)
    train_normal_path = os.path.join(root_dir, "train", "normal", channel_std)
    eval_abnormal_path = os.path.join(root_dir, "eval", "abnormal", channel_std)
    eval_normal_path = os.path.join(root_dir, "eval", "normal", channel_std)

    # Get subject lists and split train/val
    train_val_a_sub = list(set([item.split("_")[0] for item in os.listdir(train_abnormal_path)]))
    np.random.shuffle(train_val_a_sub)
    train_a_sub, val_a_sub = np.split(np.array(train_val_a_sub), [int(len(train_val_a_sub) * 0.8)])

    train_val_n_sub = list(set([item.split("_")[0] for item in os.listdir(train_normal_path)]))
    np.random.shuffle(train_val_n_sub)
    train_n_sub, val_n_sub = np.split(np.array(train_val_n_sub), [int(len(train_val_n_sub) * 0.8)])

    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(eval_abnormal_path)]))
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(eval_normal_path)]))

    # Create parameter lists
    path_map = {
        "train": (os.path.join(output_dir, "processed", "train"), {train_abnormal_path: (train_a_sub, 1), train_normal_path: (train_n_sub, 0)}),
        "val": (os.path.join(output_dir, "processed", "val"), {train_abnormal_path: (val_a_sub, 1), train_normal_path: (val_n_sub, 0)}),
        "test": (os.path.join(output_dir, "processed", "test"), {eval_abnormal_path: (test_a_sub, 1), eval_normal_path: (test_n_sub, 0)}),
    }

    for split_name, (dump_folder, sources) in path_map.items():
        os.makedirs(dump_folder, exist_ok=True)
        for source_path, (subjects, label) in sources.items():
            for subject in subjects:
                for file in os.listdir(source_path):
                    if file.startswith(subject):
                        parameters.append((os.path.join(source_path, file), dump_folder, label, "tuab"))
    return parameters

def get_tusl_parameters(root_dir, output_dir):
    """Generates the parameter list for processing the TUSL dataset."""
    print("Setting up TUSL dataset processing...")
    subjects = os.listdir(root_dir)
    np.random.shuffle(subjects)
    train_subs, val_subs = np.split(np.array(subjects), [int(len(subjects) * 0.8)])

    parameters = []
    path_map = {
        "train": (os.path.join(output_dir, "processed", "train"), train_subs),
        "val": (os.path.join(output_dir, "processed", "val"), val_subs)
    }

    for split_name, (dump_folder, subs) in path_map.items():
        os.makedirs(dump_folder, exist_ok=True)
        for subject in subs:
            subject_dir = os.path.join(root_dir, subject)
            for session in os.listdir(subject_dir):
                montage_dir = os.path.join(subject_dir, session, os.listdir(os.path.join(subject_dir, session))[0])
                for file in os.listdir(montage_dir):
                    if file.endswith(".edf"):
                        parameters.append((os.path.join(montage_dir, file), dump_folder, None, "tusl"))
    return parameters

def get_tuar_parameters(root_dir, output_dir):
    """Generates the parameter list for processing the TUAR dataset."""
    print("Setting up TUAR dataset processing...")
    parameters = []
    montages = ["01_tcp_ar", "02_tcp_le"]

    all_files = []
    for montage in montages:
        montage_dir = os.path.join(root_dir, montage)
        files = [os.path.join(montage_dir, f) for f in os.listdir(montage_dir) if f.endswith(".edf")]
        all_files.extend(files)

    np.random.shuffle(all_files)
    train_files, val_files = np.split(np.array(all_files), [int(len(all_files) * 0.8)])
    
    path_map = {
        "train": (os.path.join(output_dir, "processed", "train"), train_files),
        "val": (os.path.join(output_dir, "processed", "val"), val_files)
    }

    for split_name, (dump_folder, files) in path_map.items():
        os.makedirs(dump_folder, exist_ok=True)
        for file_path in files:
            parameters.append((file_path, dump_folder, None, "tuar"))

    return parameters

def main():
    parser = argparse.ArgumentParser(description="Process raw TUH EEG datasets into pickle files.")
    parser.add_argument("dataset", choices=["tuab", "tusl", "tuar"], help="The dataset to process.")
    parser.add_argument("--root_dir", required=True, help="Path to the raw EDF dataset directory.")
    parser.add_argument("--output_dir", required=True, help="Path to save the processed pickle files.")
    parser.add_argument("--processes", type=int, default=24, help="Number of parallel processes to use.")
    args = parser.parse_args()

    # --- Setup output directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "processed"), exist_ok=True)
    
    # --- Get parameters based on dataset ---
    if args.dataset == "tuab":
        parameters = get_tuab_parameters(args.root_dir, args.output_dir)
    elif args.dataset == "tusl":
        parameters = get_tusl_parameters(args.root_dir, args.output_dir)
    elif args.dataset == "tuar":
        parameters = get_tuar_parameters(args.root_dir, args.output_dir)
    else:
        raise ValueError("Invalid dataset choice.")

    print(f"Found {len(parameters)} files to process using {args.processes} processes.")

    # --- Run multiprocessing ---
    with Pool(processes=args.processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_and_dump_file, parameters), total=len(parameters)))

    print("Processing complete.")

if __name__ == "__main__":
    main()