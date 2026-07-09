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
import os 
import mne
import argparse
import pickle
import h5py

import numpy as np

from pathlib import Path
from tqdm import tqdm

standard = {'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2', 'ECG'}

mapping = {
    'EEG F4-M1': 'F4',
    'EEG C4-M1': 'C4',
    'EEG O2-M1': 'O2', 
    'EEG C3-M2': 'C3',
    'ECG': 'I'
}
all_channels = list(standard)

def match_annotations_and_files_and_split(input_dir):
    """
    Lists all .edf files in the directory and matches signal files with annotations. 
    Additionally, splits into train, val and test according to https://arxiv.org/pdf/2504.19596.
    First 100 subjects goes to train, next 25 for validation and next 26 for test.
    
    Args:
        input_dir (pathlib.Path): Path to the directory with recordings.
    """

    files = input_dir.rglob('*.edf')

    # .edf files are signal and _sleepscoring.edf are annotations
    signal_files = []
    annotation_files = []
    for file in files:
        
        file_name = file.parts[-1][:-4]

        if len(file_name.split('_')) == 1: # if there's no underscore, it's signal file
            signal_files.append(file)
        else: # it's annotation file
            annotation_files.append(file)

    # Files are already sorted
    train_files, train_ann = signal_files[:100], annotation_files[:100]
    val_files, val_ann = signal_files[100:125], annotation_files[100:125]
    test_files, test_ann = signal_files[125:], annotation_files[125:]

    return train_files, train_ann, val_files, val_ann, test_files, test_ann

def dump_pickle(X, y, file_name, split):
    """
    Writes data to a pickle file. 

    Args:
        X (np.array): Data to write to pickle. Shape: (num_samples, num_channels, sample_length)
        y (np.array): Label corresponding to each sample. 5-class single label classification. Shape: (num_samples,)
        file_name (str): Name to give to a pickle file.
        split (str): Either "train", "val" or "test".
    """

    data_dict = {"X": X, "y": y}
    dump_path = os.path.join(output_dir, split, f"{file_name}.pkl")

    # Write data to the path 
    with open(dump_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"Dumped into {file_name}.pkl pickle!")

def create_hdf5(output_dir, split):

    folder_path = os.path.join(output_dir, split)
    target_file = os.path.join(output_dir, f"{split}.h5")
    files = sorted(os.listdir(folder_path))

    with h5py.File(target_file, 'w') as h5f:
        for i, file in enumerate(tqdm(files)):
            with open(os.path.join(folder_path, file), 'rb') as f:

                sample = pickle.load(f)
                
                group = h5f.create_group(f"data_group_{i}")
                X_data = np.array(sample["X"])
                group.create_dataset("X", data=X_data)

                y_data = np.array(sample['y'])
                group.create_dataset("y", data=y_data)


def process_split(signal_files, annotation_files, output_dir, split):
    """
    Process one split (train/val/test).
    Adds annotations, extracts relevant channels, filters, and epochs based on sleep stage annotations. 

    Args:
        signal_files (list[Path]): List of paths to the signal .edf files.
        annotation_files (list[Path]): List of paths to the annotation .edf files.
        output_dir (str): Output directory to write files.
        split (str): Either "train", "val" or "test".
    """
    # Make directory for the split if it doesn't exist
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for signal, annot in zip(signal_files, annotation_files):

        # Get filename for saving
        file_name = signal.parts[-1][:-4]

        # Load signal and annotations
        raw = mne.io.read_raw_edf(signal, preload=True, verbose=False, include=all_channels)
        ann = mne.read_annotations(annot)
        raw.set_annotations(ann)

        # Rename channel type to ecg for easier filtering // rename for positions
        raw.set_channel_types({'ECG': 'ecg'})
        raw.rename_channels(mapping)
        raw.filter(l_freq=0.1, h_freq=75.0, picks='eeg', verbose='ERROR')
        raw.filter(l_freq=0.5, h_freq=120.0, picks='ecg', verbose='ERROR')
        raw.notch_filter(50.0, verbose="ERROR")

        # Get annotations
        events, event_id = mne.events_from_annotations(raw)
        event_id = {k: v for k, v in event_id.items() if v not in [1,2]} # event_id markers 1,2 correspond to the light on/off moments
        epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=30, baseline=None, verbose=True)
        data = epochs.get_data()[:, :, :-1] # epochs returns 30*sampling_freq + 1 samples so we discard the last one

        # We need to create labels - it's a third column from events
        # Just need to discard the 1 and 2 corresponding to lights on/off
        # Subtracting 3 to shift to 0-4 values
        labels = events[:, 2]
        remove_on_off = labels[(labels != 1) & (labels != 2)]
        labels_corrected = remove_on_off - 3

        # Write to pickle
        dump_pickle(X=data, y=labels_corrected, file_name=file_name, split=split)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prepath', type=str, default='#CHANGEME')
    parser.add_argument('--output_path', type=str, default='#CHANGEME')

    args = parser.parse_args()
    input_dir = Path(args.prepath)
    output_dir = args.output_path

    # List all the files in the directory
    files = [f for f in input_dir.rglob('*.edf')]
    train_files, train_ann, val_files, val_ann, test_files, test_ann = match_annotations_and_files_and_split(input_dir)

    # Make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each split
    process_split(train_files, train_ann, output_dir, split="train")
    process_split(val_files, val_ann, output_dir, split="val")
    process_split(test_files, test_ann, output_dir, split="test")

    # Write to HDF5
    create_hdf5(output_dir, split="train")
    create_hdf5(output_dir, split="val")
    create_hdf5(output_dir, split="test")


