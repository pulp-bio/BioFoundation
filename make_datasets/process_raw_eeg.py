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
import pandas as pd

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

TUSL_LABEL_MAP = {
    'bckg': 1,
    'seiz': 2,
    'slow': 3
}

MULTILABEL_MAP = {
    # chew family => 1
    'chew': 1, 'chew_elec': 1, 'chew_musc': 1,
    # elec => 2
    'elec': 2,
    # eyem family => 3
    'eyem': 3, 'eyem_chew': 3, 'eyem_elec': 3, 'eyem_musc': 3, 'eyem_shiv': 3,
    # musc family => 4
    'musc': 4, 'musc_elec': 4,
    # shiv family => 5
    'shiv': 5, 'shiv_elec': 5
}

TUAR_ARTIFACT_LABELS_BINARY = {
    'chew', 'chew_elec', 'chew_musc', 'cpsz', 'elec', 'elpp',
    'eyem', 'eyem_chew', 'eyem_elec', 'eyem_musc', 'eyem_shiv',
    'fnsz', 'gnsz', 'musc', 'musc_elec', 'shiv', 'shiv_elec',
    'tcsz'
}

def init_label_array(mode, n_channels, n_times):
    """
    Initialize an empty label array depending on mode.
    """
    if mode == "Binary":
        # Single dimension: n_times
        return np.zeros((n_times,), dtype=np.int8)
    elif mode == "MultiBinary":
        # Two dimensions: (n_channels, n_times)
        return np.zeros((n_channels, n_times), dtype=np.int8)
    elif mode == "MultiLabel":
        # Two dimensions: (n_channels, n_times)
        return np.zeros((n_channels, n_times), dtype=np.int8)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
def apply_csv_annotations(mode, df_art, label_array, channel_map, sfreq, n_times):
    """
    For each row in df_art, update label_array with the given artifact info.
    """
    for row in df_art.itertuples():
        start_sec = getattr(row, 'start_time', None)
        stop_sec = getattr(row, 'stop_time', None)
        label_str = getattr(row, 'label', "")
        channel_name = getattr(row, 'channel', None)
        if start_sec is None or stop_sec is None or channel_name is None:
            continue

        start_ix = int(round(start_sec * sfreq))
        stop_ix = int(round(stop_sec * sfreq))
        start_ix = max(start_ix, 0)
        stop_ix = min(stop_ix, n_times)

        # If the channel is not in our final set, skip
        if channel_name not in channel_map:
            continue
        ch_idx = channel_map[channel_name]
        
        if mode == "Binary":
            # If label_str is in ARTIFACT_LABELS_BINARY => set 1
            if label_str in TUAR_ARTIFACT_LABELS_BINARY:
                label_array[start_ix:stop_ix] = 1  # shape (n_times,)
        elif mode == "MultiBinary":
            # If label_str is in ARTIFACT_LABELS_BINARY => set 1 for that channel
            if label_str in TUAR_ARTIFACT_LABELS_BINARY:
                label_array[ch_idx, start_ix:stop_ix] = 1
        elif mode == "MultiLabel":
            # Find which group label this belongs to; if not found => 0
            label_code = MULTILABEL_MAP.get(label_str, 0)
            label_array[ch_idx, start_ix:stop_ix] = label_code

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
    
    This function now routes to different logic based on the dataset.
    """
    # <-- UPDATED PARAM UNPACKING -->
    file_path, dump_folder, label_or_mode, dataset_name = params
    file_name = os.path.basename(file_path)
    window_size_samples = 5 * 256 # 5 seconds at 256 Hz
    chunk_size = window_size_samples # Alias for clarity in TUAR logic

    try:
        # --- 1. File Loading and Common Preprocessing ---
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        if any("LE" in ch for ch in raw.ch_names):
            raw.rename_channels(lambda x: x.replace("LE", "REF"))
        
        raw.pick_channels(CH_ORDER_STANDARD, ordered=True)

        if raw.ch_names != CH_ORDER_STANDARD:
            raise ValueError("Channel order mismatch after picking.")

        raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
        raw.notch_filter(60, verbose=False)
        if int(raw.info['sfreq']) != 256:
            raw.resample(256, npad="auto", n_jobs=1, verbose=False)
        
        final_sfreq = raw.info['sfreq']

        # --- 2. Re-referencing (Bipolar) ---
        channeled_data, channeled_channels = make_bipolar(raw)
        if len(channeled_channels) < 20:
            raise ValueError(f"Found only {len(channeled_channels)} channels after making bipolar.")
        
        n_channels, n_times = channeled_data.shape

        # --- 3. Dataset-Specific Labeling and Windowing ---

        if dataset_name == "tuab":
            # --- TUAB Logic (File-level label) ---
            label = label_or_mode # <-- Get label
            for i in range(n_times // window_size_samples):
                dump_path = os.path.join(dump_folder, f"{file_name.split('.')[0]}_{i}.pkl")
                segment = channeled_data[:, i * window_size_samples: (i + 1) * window_size_samples]
                
                data_to_dump = {"X": segment}
                if label is not None:
                    data_to_dump["y"] = label
                
                with open(dump_path, "wb") as f:
                    pickle.dump(data_to_dump, f)

        elif dataset_name == "tusl":
            # --- TUSL Logic (Segment-level label from CSV) ---
            csv_path = os.path.splitext(file_path)[0] + ".csv"
            label_array = np.zeros((n_times,), dtype=np.int8)

            if os.path.exists(csv_path):
                df_art = pd.read_csv(csv_path, comment='#')
                for row in df_art.itertuples():
                    start_sec = getattr(row, 'start_time', None)
                    stop_sec = getattr(row, 'stop_time', None)
                    label_str = getattr(row, 'label', "")
                    if start_sec is None or stop_sec is None:
                        continue

                    start_ix = int(round(start_sec * final_sfreq))
                    stop_ix = int(round(stop_sec * final_sfreq))
                    start_ix = max(start_ix, 0)
                    stop_ix = min(stop_ix, n_times)

                    code = TUSL_LABEL_MAP.get(label_str, 0)
                    if code != 0:
                        label_array[start_ix:stop_ix] = code
            
            for i in range(n_times // window_size_samples):
                dump_path = os.path.join(dump_folder, f"{file_name.split('.')[0]}_{i}.pkl")
                X_chunk = channeled_data[:, i * window_size_samples: (i + 1) * window_size_samples]
                y_segment = label_array[i * window_size_samples: (i + 1) * window_size_samples]

                vals, counts = np.unique(y_segment, return_counts=True)
                chunk_label = vals[np.argmax(counts)]

                data_to_dump = {"X": X_chunk, "y": chunk_label}
                with open(dump_path, "wb") as f:
                    pickle.dump(data_to_dump, f)

        elif dataset_name == "tuar":
            # --- TUAR Logic (Multi-mode artifact labeling) ---
            MODE = label_or_mode # <-- Get mode
            csv_path = os.path.splitext(file_path)[0] + ".csv"
            
            label_array = init_label_array(MODE, n_channels, n_times)
            channel_map = {name: i for i, name in enumerate(channeled_channels)}

            if os.path.exists(csv_path):
                df_art = pd.read_csv(csv_path, comment='#')
                apply_csv_annotations(MODE, df_art, label_array, channel_map, final_sfreq, n_times)
            
            for i in range(n_times // window_size_samples):
                start = i * window_size_samples
                end = (i + 1) * window_size_samples
                X_chunk = channeled_data[:, start:end]
                
                chunk_label = None

                if MODE == "Binary":
                    y_chunk = label_array[start:end] # 1D array
                    fraction_artifact = np.mean(y_chunk)
                    chunk_label = 1 if fraction_artifact >= 0.3 else 0 # Scalar

                elif MODE == "MultiBinary":
                    y_chunk = label_array[:, start:end] # 2D array
                    chunk_label = np.zeros((n_channels,), dtype=np.int8) # 1D array
                    for ch in range(n_channels):
                        fraction_artifact = np.mean(y_chunk[ch])
                        chunk_label[ch] = 1 if fraction_artifact >= 0.3 else 0

                elif MODE == "MultiLabel":
                    y_chunk = label_array[:, start:end] # 2D array
                    chunk_label = np.zeros((n_channels,), dtype=np.int8) # 1D array
                    for ch in range(n_channels):
                        values, counts = np.unique(y_chunk[ch], return_counts=True)
                        found_label = 0
                        max_count = 0
                        for val, cnt in zip(values, counts):
                            fraction = cnt / float(chunk_size)
                            if fraction >= 0.3 and cnt > max_count:
                                max_count = cnt
                                found_label = val
                        chunk_label[ch] = found_label
                
                else:
                    raise ValueError(f"Invalid mode {MODE} specified for TUAR processing.")

                # Dump the pickle
                dump_path = os.path.join(dump_folder, f"{file_name.split('.')[0]}_{i}.pkl")
                data_to_dump = {"X": X_chunk, "y": chunk_label}
                with open(dump_path, "wb") as f:
                    pickle.dump(data_to_dump, f)

    except Exception as e:
        error_log_file = f"{dataset_name}-process-errors.txt"
        with open(error_log_file, "a") as f:
            f.write(f"Error processing {file_name}: {e}\n")

# --- Parameter Generation Functions ---

def get_tuab_parameters(root_dir, output_dir):
    """Generates the parameter list for processing the TUAB dataset."""
    print("Setting up TUAB dataset processing...")
    parameters = []
    channel_std = "01_tcp_ar"

    train_abnormal_path = os.path.join(root_dir, "train", "abnormal", channel_std)
    train_normal_path = os.path.join(root_dir, "train", "normal", channel_std)
    eval_abnormal_path = os.path.join(root_dir, "eval", "abnormal", channel_std)
    eval_normal_path = os.path.join(root_dir, "eval", "normal", channel_std)

    train_val_a_sub = list(set([item.split("_")[0] for item in os.listdir(train_abnormal_path)]))
    np.random.shuffle(train_val_a_sub)
    train_a_sub, val_a_sub = np.split(np.array(train_val_a_sub), [int(len(train_val_a_sub) * 0.8)])

    train_val_n_sub = list(set([item.split("_")[0] for item in os.listdir(train_normal_path)]))
    np.random.shuffle(train_val_n_sub)
    train_n_sub, val_n_sub = np.split(np.array(train_val_n_sub), [int(len(train_val_n_sub) * 0.8)])

    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(eval_abnormal_path)]))
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(eval_normal_path)]))

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
                        # Pass 'label' (0 or 1) in the 'label_or_mode' slot
                        parameters.append((os.path.join(source_path, file), dump_folder, label, "tuab"))
    return parameters

def get_tusl_parameters(root_dir, output_dir):
    """Generates the parameter list for processing the TUSL dataset."""
    print("Setting up TUSL dataset processing...")
    subjects = os.listdir(root_dir)
    np.random.shuffle(subjects)
    train_subs, val_subs = np.split(np.array(subjects), [int(len(subjects) * 0.8)])
    val_subs, test_subs = np.split(val_subs, [int(len(val_subs) * 0.5)])

    parameters = []
    path_map = {
        "train": (os.path.join(output_dir, "processed", "train"), train_subs),
        "val": (os.path.join(output_dir, "processed", "val"), val_subs),
        "test": (os.path.join(output_dir, "processed", "test"), test_subs)
    }

    for split_name, (dump_folder, subs) in path_map.items():
        os.makedirs(dump_folder, exist_ok=True)
        for subject in subs:
            subject_dir = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_dir): continue
            for session in os.listdir(subject_dir):
                session_path = os.path.join(subject_dir, session)
                if not os.path.isdir(session_path): continue
                montage_dirs = [d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))]
                if not montage_dirs: continue
                montage_dir = os.path.join(session_path, montage_dirs[0])
                
                for file in os.listdir(montage_dir):
                    if file.endswith(".edf"):
                        # Pass 'None' in the 'label_or_mode' slot
                        parameters.append((os.path.join(montage_dir, file), dump_folder, None, "tusl"))
    return parameters

def get_tuar_parameters(root_dir, output_dir, mode):
    """Generates the parameter list for processing the TUAR dataset."""
    print(f"Setting up TUAR dataset processing in '{mode}' mode...") # <-- Added mode
    parameters = []
    montages = ["01_tcp_ar", "02_tcp_le"]

    all_files = []
    for montage in montages:
        montage_dir = os.path.join(root_dir, montage)
        if not os.path.isdir(montage_dir): continue
        files = [os.path.join(montage_dir, f) for f in os.listdir(montage_dir) if f.endswith(".edf")]
        all_files.extend(files)

    np.random.shuffle(all_files)
    train_files, val_files = np.split(np.array(all_files), [int(len(all_files) * 0.8)])
    val_files, test_files = np.split(val_files, [int(len(val_files) * 0.5)])
    
    path_map = {
        "train": (os.path.join(output_dir, "processed", "train"), train_files),
        "val": (os.path.join(output_dir, "processed", "val"), val_files),
        "test": (os.path.join(output_dir, "processed", "test"), test_files)
    }

    for split_name, (dump_folder, files) in path_map.items():
        os.makedirs(dump_folder, exist_ok=True)
        for file_path in files:
            # Pass 'mode' (e.g., "Binary") in the 'label_or_mode' slot
            parameters.append((file_path, dump_folder, mode, "tuar"))

    return parameters

def main():
    parser = argparse.ArgumentParser(description="Process raw TUH EEG datasets into pickle files.")
    parser.add_argument("dataset", choices=["tuab", "tusl", "tuar"], help="The dataset to process.")
    parser.add_argument("--root_dir", required=True, help="Path to the raw EDF dataset directory.")
    parser.add_argument("--output_dir", required=True, help="Path to save the processed pickle files.")
    parser.add_argument("--processes", type=int, default=24, help="Number of parallel processes to use.")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="Binary", 
        choices=["Binary", "MultiBinary", "MultiLabel"],
        help="Labeling mode for TUAR dataset (default: Binary)."
    )
    args = parser.parse_args()

    if args.dataset == "tuab":
        dataset_name = "TUAB_data"
    elif args.dataset == "tusl":
        dataset_name = "TUSL_data"
    elif args.dataset == "tuar":
        dataset_name = "TUAR_data"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, dataset_name, "processed"), exist_ok=True)
    output_dir_dataset = os.path.join(args.output_dir, dataset_name)
    
    if args.dataset == "tuab":
        parameters = get_tuab_parameters(args.root_dir, output_dir_dataset)
    elif args.dataset == "tusl":
        parameters = get_tusl_parameters(args.root_dir, output_dir_dataset)
    elif args.dataset == "tuar":
        parameters = get_tuar_parameters(args.root_dir, output_dir_dataset, args.mode)
    else:
        raise ValueError("Invalid dataset choice.")

    print(f"Found {len(parameters)} files to process using {args.processes} processes.")

    with Pool(processes=args.processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_and_dump_file, parameters), total=len(parameters)))

    print("Processing complete.")

if __name__ == "__main__":
    main()