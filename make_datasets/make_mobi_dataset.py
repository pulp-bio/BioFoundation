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
import h5py
import argparse
import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm

## MoBI-specific constants

MoBI_CHANNEL_ORDER_with_EOG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1",
            "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5",
            "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8","PO9",
            "O1", "Oz", "O2", "PO10", "AF7", "AF3", "AF4", "AF8",
            "F5", "F1", "F2", "F6", "FT9", "FT7", "FC3",
            "FC4", "FT8", "FT10", "C5", "C1", "C2", "C6", "TP7",
            "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6",
            "PO7", "PO3", "POz", "PO4", "PO8"
            ]

JOINTS = ["GHR", "GKR", "GAR", "GHL", "GKL", "GAL", "PHR", "PKR", "PAR", "PHL", "PKL", "PAL"]

MoBI_EOG_CHANNELS = ['TP9', 'TP10', 'FT9', 'FT10']

MoBI_CHANNEL_ORDER = [ch for ch in MoBI_CHANNEL_ORDER_with_EOG if ch not in MoBI_EOG_CHANNELS]

ORIGINAL_SFREQ = 100 # Hz

RESAMPLE_SFREQ = 100 # Hz ; no upsampling, just keep original sampling rate

WINDOW_SEC = 2.0 # seconds

STRIDE_SEC = 0.05 # seconds ; stride to generate segments, e.g. 0.05s = 50ms stride means 20 segments per second of data

## Dataset-specific processing functions

def raw_from_txt(txt_path_joints, txt_path_eeg, sfreq=100):
    ## 1. read txt
    # skip two header lines to read the data, open as df and print the columns
    joints_data = np.loadtxt(txt_path_joints, skiprows=2)
    # skip one header line to read the data, open as df and print the columns
    eeg_data = np.loadtxt(txt_path_eeg, skiprows=1)

    ## 2. convert to df
    joints_df = pd.DataFrame(joints_data)
    eeg_df = pd.DataFrame(eeg_data)

    ## 3. set columns
    joints_df.columns = ["Time"] + JOINTS
    eeg_df.columns = ["Time"] + MoBI_CHANNEL_ORDER_with_EOG

    # remove "Time" column from eeg_df and use it as index
    eeg_df.set_index("Time", inplace=True)
    joints_df.set_index("Time", inplace=True)

    info = mne.create_info(
        ch_names=MoBI_CHANNEL_ORDER_with_EOG,
        sfreq=sfreq,
        ch_types=["eeg"] * len(MoBI_CHANNEL_ORDER_with_EOG)
    )

    # convert to mne Raw object
    raw = mne.io.RawArray(eeg_df[MoBI_CHANNEL_ORDER_with_EOG].values.T, info)

    # drop EOG channels from raw
    raw.drop_channels(MoBI_EOG_CHANNELS)

    # assert that joints and eeg have the same number of samples
    assert raw.n_times == joints_df.shape[0], "Number of samples in raw and joints_df do not match"

    return raw, joints_df

def preprocess_raw(raw, joints_df):
    # Set montage
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        match_alias={'cb1': 'POO7', 'cb2': 'POO8'}
    )

    # Reorder channels
    raw.pick_channels(MoBI_CHANNEL_ORDER, ordered=True)

    # divide angles by 90 for normalization
    joints_df = joints_df / 90.0

    return raw, joints_df

def split_train_val_test(raw, joints_df):
    sfreq = int(raw.info['sfreq'])

    # Minutes → samples
    two_min = int(2 * 60 * sfreq)
    ten_min = int(10 * 60 * sfreq)
    five_min = int(5 * 60 * sfreq)

    n_samples = raw.n_times

    # --- Remove standing still (start & end) ---
    usable_start = two_min
    usable_end = n_samples 

    raw = raw.copy().crop(
        tmin=usable_start / sfreq,
        tmax=(usable_end - 1) / sfreq
    )
    joints_df = joints_df.iloc[usable_start:usable_end]

    # --- Explicit split boundaries ---
    train_start = 0
    train_end = train_start + ten_min

    val_start = train_end
    val_end = val_start + five_min

    test_start = val_end
    test_end = test_start + five_min

    # Safety check to ensure we don't exceed available data
    if test_end > raw.n_times:
        print("Warning: Not enough data for the specified splits. Adjusting test_end to n_times.")
        test_end = raw.n_times

    splits = {
        "train": (
            raw.copy().crop(
                tmin=train_start / sfreq,
                tmax=(train_end - 1) / sfreq
            ),
            joints_df.iloc[train_start:train_end]
        ),
        "val": (
            raw.copy().crop(
                tmin=val_start / sfreq,
                tmax=(val_end - 1) / sfreq
            ),
            joints_df.iloc[val_start:val_end]
        ),
        "test": (
            raw.copy().crop(
                tmin=test_start / sfreq,
                tmax=(test_end - 1) / sfreq
            ),
            joints_df.iloc[test_start:test_end]
        )
    }

    return splits

def segment_raw_with_stride(raw, joints_df, window_sec=2.0, stride_sec=0.05):
    sfreq = int(raw.info['sfreq'])
    window_samples = int(window_sec * sfreq)
    stride_samples = int(stride_sec * sfreq)

    data = raw.get_data()          # (n_channels, n_times)
    n_channels, n_times = data.shape

    segments = []
    segments_joints = []

    max_start = n_times - window_samples

    for start in range(0, max_start + 1, stride_samples):
        end = start + window_samples

        # EEG segment
        eeg_seg = data[:, start:end]

        # Label: average over last stride window
        label_start = end - stride_samples
        label_end = end

        joint_seg = joints_df.iloc[label_start:label_end].values
        joint_avg = joint_seg.mean(axis=0)  # (12,)

        segments.append(eeg_seg)
        segments_joints.append(joint_avg)

    segments = np.stack(segments)              # (N, C, T)
    segments_joints = np.stack(segments_joints)  # (N, 12)

    return segments, segments_joints

def write_segments_to_h5_append_regression(h5_path, segments, y, group_size=1000):
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "a") as h5f:

        existing_groups = [
            int(k.split("_")[-1])
            for k in h5f.keys()
            if k.startswith("data_group_")
        ]
        start_idx = max(existing_groups) + 1 if existing_groups else 0

        n_segments = segments.shape[0]

        for i in tqdm(range(0, n_segments, group_size),
                      desc=f"Appending to {os.path.basename(h5_path)}"):

            grp_idx = start_idx + (i // group_size)
            grp = h5f.create_group(f"data_group_{grp_idx}")

            X = segments[i:i + group_size]
            Y = y[i:i + group_size]

            grp.create_dataset("X", data=X, compression="gzip")
            grp.create_dataset("y", data=Y, compression="gzip")

## Main processing functions

def process_and_convert_to_h5(prepath, output_path):
    out_dir =  os.path.join(output_path, "MoBI_regression_2s_sample")
    data_dir = prepath

    sessions = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for ses in sessions[:]:
        subject_id, trial_id = ses.split("-")
        print(f"Processing subject {subject_id}, trial {trial_id}")

        joints_path = os.path.join(data_dir, ses, "joints.txt")
        eeg_path = os.path.join(data_dir, ses, "eeg.txt")

        # load data and convert to raw and joints_df (angle values in degrees)
        raw, joints_df = raw_from_txt(joints_path, eeg_path,sfreq=ORIGINAL_SFREQ)

        # preprocess raw and joints_df (set montage, reorder channels, normalize angles)
        raw, joints_df = preprocess_raw(raw, joints_df)

        # split into train/val/test (10min/5min/5min)
        splits = split_train_val_test(raw, joints_df)

        for split_name, (split_raw, split_joints) in splits.items():
            # segment into windows with stride and calculate average joint angles for each segment (last stride window)
            segments, segments_joints = segment_raw_with_stride(split_raw, split_joints,
                                                    window_sec=WINDOW_SEC,
                                                    stride_sec=STRIDE_SEC)

            h5_path = os.path.join(
                out_dir,
                f"{split_name}.h5"
            )

            write_segments_to_h5_append_regression(h5_path, segments, segments_joints)
    print("Preprocessing complete. HDF5 files saved to:", output_path)

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