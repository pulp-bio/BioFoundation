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
import mne 
import argparse
import os
import numpy as np
import csv
import datetime
import pickle
import h5py
from tqdm import tqdm
# Code taken from SEED-VII instructions
sampling_freq=256

# We decide to label 
# neutral = 0
# happy = 1
# sad = 2
# disgust = 3
# anger = 4
# fear = 5
# surprise = 6

per_session_labels = {
    '1': [1, 0, 3, 2, 4, 4, 2, 3, 0, 1, 1, 0, 3, 2, 4, 4, 2, 3, 0, 1],
    '2': [4, 2, 5, 0, 6, 6, 0, 5, 2, 4, 4, 2, 5, 0, 6, 6, 0, 5, 2, 4],
    '3': [1, 6, 3, 5, 4, 4, 5, 3, 6, 1, 1, 6, 3, 5, 4, 4, 5, 3, 6, 1],
    '4': [3, 2, 5, 6, 1, 1, 6, 5, 2, 3, 3, 2, 5, 6, 1, 1, 6, 5, 2, 3]
}

channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2', 'ECG']

def process_and_split(input_dir, output_dir, file_dict):
    """
    Process sessions and place it in correct split (train/val/test). Following splits from PhysioOmni.
    20 subjects - 4 sessions per each - 20 trials in each
    First 10 trials from each session go to the train split.
    Next 5 trials from each session go to the val split and last 5 to the test split.
    """
    collect_num = 0
    for key, value in file_dict.items():
        for raw_file in value:
            print("Starting with path:", raw_file)
            # Prepare the path to the file
            raw_path = os.path.join(input_dir, raw_file)
            raw = mne.io.read_raw_cnt(raw_path, ecg=['ECG'])
            
            # Drop useless channels 
            raw.drop_channels(['M1', 'M2', 'HEO', 'VEO'])
            raw.load_data()
            
            # Preprocessing 
            raw.filter(l_freq=0.1, h_freq=75.0, picks='eeg', verbose='ERROR')
            raw.filter(l_freq=0.5, h_freq=120.0, picks='ecg', verbose='ERROR')
            raw.notch_filter(50.0, verbose='ERROR')
            
            if raw.info['sfreq'] != sampling_freq:
                raw.resample(sampling_freq)
            
            # Get triggers for each trial
            trigger, _ = mne.events_from_annotations(raw)
            data, times = raw.get_data(return_times=True)
            
            t = trigger[:, 0]
            
            # We need to separatrly handle files authors specify are not accurately handled with triggers 
            if raw_file == "14_20221015_1.cnt":
                t = []
                start = datetime.datetime.strptime('14:25:34', '%H:%M:%S')
                with open('#CHANGEME') as f:
                    trigger = csv.reader(f)
                    for row in trigger:
                        end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                        time_diff = end.timestamp() - start.timestamp()
                        t.append(int(round(time_diff * sampling_freq)))
            elif raw_file == "9_20221111_3.cnt":
                t = []
                start = datetime.datetime.strptime('14:01:27', '%H:%M:%S')
                with open(os.path.join('#CHANGEME')) as f:
                    trigger = csv.reader(f)
                    for row in trigger:
                        end = datetime.datetime.strptime(row[1].split(' ')[-1], '%H:%M:%S.%f')
                        time_diff = end.timestamp() - start.timestamp()
                        t.append(int(round(time_diff * sampling_freq)))
            
            session_idx = int(raw_file[-5])  
            for i in range(20):
                # Get the data between two triggers
                preprocessed_clip = data[:, t[2 * i]:t[2 * i + 1]]
                num = preprocessed_clip.shape[1] // sampling_freq
                collect_num += num
                
                # We want to fetch int number of seconds and reshape such that each second is segment
                signal = preprocessed_clip[:, :num * sampling_freq]
                split_signal = signal.reshape(signal.shape[0], num, sampling_freq).transpose(1, 0, 2)
                
                for idx in range(num):
                    data_dict = {"X": split_signal[idx, :, :], 'y': per_session_labels[str(session_idx)][i]}
                    dump_name = f"{key}_seed_{session_idx}_{i}_{idx}.pkl"
                    
                    # first 10 trials from session go to train split
                    if i < 10:
                        dump_path = os.path.join(output_dir, "train", dump_name)
                        
                    elif i < 15 and i >= 10:
                        dump_path = os.path.join(output_dir, "val", dump_name)
                    
                    else:
                        dump_path = os.path.join(output_dir, "test", dump_name)
                        
                    with open(dump_path, "wb") as f: 
                        pickle.dump(data_dict, f)
            print("Current sample count:", collect_num)   

def create_hdf5(source_dir, target_file, finetune=True, group_size=1000):
    """
    Lists all the pickle files in the source directory and writes them in the .h5 file.

    Args:
        source_dir (str): Source directory where pickle files are found. 
        target_file (str): Path to the target .h5 file. 
        finetune (bool): Whether data has label or not. 
        group_size (int): Group size in HDF5 saving. 
    """
    # List all the files in the folder
    files = sorted(os.listdir(source_dir))
    data_group = []

    
    with h5py.File(target_file, 'w') as h5f:
        for i, file in enumerate(tqdm(files)):
            with open(os.path.join(source_dir, file), 'rb') as f:
                
                sample = pickle.load(f)
                data_group.append(sample)
                
                if (i + 1) % group_size == 0 or i == len(files) - 1:
                    
                    grp = h5f.create_group(f"data_group_{i // group_size}")
                    X_data = np.array([s['X'] for s in data_group])
                    grp.create_dataset("X", data=X_data)
                    
                    if(finetune):
                        y_data = np.array([s['y'] for s in data_group])
                        print(y_data)
                        grp.create_dataset("y", data=y_data)

                    data_group = []
                    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepath', type=str, default='#CHANGEME')
    parser.add_argument('--output_path', type=str, default='#CHANGEME')
    
    # Parse arguments
    args = parser.parse_args()
    input_dir = args.prepath
    output_dir = args.output_path
    
    # List all files in the directory and prepare per subjects
    raw_files = os.listdir(input_dir)
    file_dict = {}
    
    for file in raw_files:
        if file[:-15] not in file_dict.keys():
            file_dict[file[:-15]] = [file]
        else:
            file_dict[file[:-15]].append(file)
            
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    process_and_split(input_dir, output_dir, file_dict)
    
    # Finally, write to HDF5
    to_do = ['train', 'val', 'test']
    for td in to_do:
        if os.path.exists(output_dir + '/' + td + '.h5'):
            print(f"File {td}.h5 already exists!")
        else:
            print(f"Creating file {td}.h5.")
            create_hdf5(output_dir + "/" + td, output_dir + "/" + td + ".h5")        
    
