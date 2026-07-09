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

import pickle
import argparse
import h5py
import os

import numpy as np

from process_raw_ecg import preprocess_signal
from tqdm import tqdm

def binary_classification(ecg_process, ppg_process, labels, subject, split_type):
    """
    Label IDs: 
        - 0 = not defined/transient
        - 1 = baseline
        - 2 = stress
        - 3 = amusement
        - 4 = meditation
        - 5/6/7 = should be ignored in this dataset
        
    Binary classification: Combined baseline and amusement sessions form non-stress class, while the stress class is original stress session.
    """
    # Each segment is 1min long
    subseq_size_label = 700 * 60
    subseq_size_data = 256 * 60
    
    # Extact unique labels and their occurence points 
    uniques, uniques_index = np.unique(labels, return_index=True)
    counter_subject = 0
    for unique, unique_startidx in zip(uniques, uniques_index):
        if unique not in [1, 2, 3]: 
            continue
        
        while True:
            if unique_startidx // subseq_size_label * subseq_size_data > ppg_process.shape[1]:
                break
            for next_idx in range(unique_startidx, len(labels)):
                if unique != labels[next_idx]:
                    break
            
            totalsubseqs = (next_idx - unique_startidx) // subseq_size_label
            startidx = unique_startidx // subseq_size_label * subseq_size_data
            data_temp_60sec_ppg = ppg_process[:, startidx:startidx + totalsubseqs * subseq_size_data]
            data_temp_60sec_ecg = ecg_process[:, startidx:startidx + totalsubseqs * subseq_size_data]
        
            if unique_startidx // subseq_size_label * subseq_size_data + totalsubseqs * subseq_size_data > ppg_process.shape[1]:
                totalsubseqs = data_temp_60sec_ppg.shape[0] // subseq_size_data
                data_temp_60sec_ppg = data_temp_60sec_ppg[:totalsubseqs * subseq_size_data, :]
                data_temp_60sec_ecg = data_temp_60sec_ecg[:totalsubseqs * subseq_size_data, :]

            if totalsubseqs == 0:
                break
        
            data_temp_60sec_ppg = np.stack(np.split(data_temp_60sec_ppg, totalsubseqs, 1), 0)
            data_temp_60sec_ecg = np.stack(np.split(data_temp_60sec_ecg, totalsubseqs, 1), 0)
        
            # Concat both signals 
            total_concat = np.concatenate([data_temp_60sec_ecg, data_temp_60sec_ppg], axis=1)
            unique_temp = 1 if unique == 3 else unique
            label_temp_60sec = np.repeat(unique_temp, totalsubseqs)
        
            print(f"{subject}:", total_concat.shape, label_temp_60sec)
        
            # Write each sample piece to pickle 
            for idx in range(len(label_temp_60sec)):
            
                data_dict = {"X": total_concat[idx], "y": label_temp_60sec[idx]}
                dump_path = os.path.join(output_dir, split_type, f"wesad_{subject}_{counter_subject}.pkl")
                counter_subject += 1
            
                with open(dump_path, 'wb') as f:
                    pickle.dump(data_dict, f)
            
            if unique != 4:
                break
        
def multiclass_classification(ecg_process, ppg_process, labels, subject, split_type):
    """
    Multiclass classification: Stress, Baseline, Amusement and Meditation.
    """
    # Each segment is 1min long
    subseq_size_label = 700 * 60
    subseq_size_data = 256 * 60
    
    # Extact unique labels and their occurence points 
    uniques, uniques_index = np.unique(labels, return_index=True)
    counter_subject = 0
    for unique, unique_startidx in zip(uniques, uniques_index):
        
        flag = False
        if unique not in [1, 2, 3, 4]:
            continue
        
        while True:
            if unique_startidx // subseq_size_label * subseq_size_data > ppg_process.shape[1]:
                break
            for next_idx in range(unique_startidx, len(labels)):
                if unique != labels[next_idx]:
                    break
         
            totalsubseqs = (next_idx - unique_startidx) // subseq_size_label
            startidx = unique_startidx // subseq_size_label * subseq_size_data
            data_temp_60sec_ppg = ppg_process[:, startidx:startidx + totalsubseqs * subseq_size_data]
            data_temp_60sec_ecg = ecg_process[:, startidx:startidx + totalsubseqs * subseq_size_data]  
        
            if unique_startidx // subseq_size_label * subseq_size_data + totalsubseqs * subseq_size_data > ppg_process.shape[1]:
                totalsubseqs = data_temp_60sec_ppg.shape[0] // subseq_size_data
                data_temp_60sec_ppg = data_temp_60sec_ppg[:totalsubseqs * subseq_size_data, :]
                data_temp_60sec_ecg = data_temp_60sec_ecg[:totalsubseqs * subseq_size_data, :]

            if totalsubseqs == 0:
                break
            
            data_temp_60sec_ppg = np.stack(np.split(data_temp_60sec_ppg, totalsubseqs, 1), 0)
            data_temp_60sec_ecg = np.stack(np.split(data_temp_60sec_ecg, totalsubseqs, 1), 0)
            
            # Concat both signals 
            total_concat = np.concatenate([data_temp_60sec_ecg, data_temp_60sec_ppg], axis=1)
            label_temp_60sec = np.repeat(unique, totalsubseqs)
            
            print(f"{subject}:", total_concat.shape, label_temp_60sec)
            # Write each sample piece to pickle 
            for idx in range(len(label_temp_60sec)):
            
                data_dict = {"X": total_concat[idx], "y": label_temp_60sec[idx]}
                dump_path = os.path.join(output_dir, split_type, f"wesad_{subject}_{counter_subject}.pkl")
                counter_subject += 1
                
                with open(dump_path, 'wb') as f:
                    pickle.dump(data_dict, f)
                
            if unique != 4:
                break
            else: 
                if flag:
                    break
                flag = True
                new_label = labels[next_idx:]
                uniques_temp, uniques_indedata_temp = np.unique(new_label, return_index=True)
                try:
                    unique_startidx = uniques_indedata_temp[np.where(uniques_temp == 4)][0] + next_idx
                except IndexError:
                    break          

def process_split(input_dir, output_dir, subjects, split_type, classification_type):
    """
    Process PPG and ECG data, depending on the classification type of the task.
    """
    # Create directory for the split
    os.makedirs(os.path.join(output_dir, split_type), exist_ok=True)
    
    # Iterate over all subjects
    for subject in subjects:
        file_path = os.path.join(input_dir, subject, f"{subject}.pkl")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        # Extract the channels and labels
        ecg = data['signal']['chest']['ECG'].T
        ppg = data['signal']['wrist']['BVP'].T 
        labels = data['label']
        
        # Preprocess the signal
        ecg_process = preprocess_signal(ecg, fs=700, low=0.5, high=120.0, downsample_fs=256, upsample_fs=None)
        ppg_process = preprocess_signal(ppg, fs=64, low=0.5, high=8.0, downsample_fs=None, upsample_fs=256)
        
        # Depending on the classification_type prepare the data
        if classification_type == "binary":
            binary_classification(ecg_process, ppg_process, labels, subject, split_type)
        else:
            multiclass_classification(ecg_process, ppg_process, labels, subject, split_type)
            
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
    parser.add_argument('--prepath', type=str, default='#CHANGEME', help='Input directory containing raw files.')
    parser.add_argument('--output_path', type=str, default='#CHANGEME')
    
    # Parse arguments
    args = parser.parse_args()
    input_dir = args.prepath
    output_dir = args.output_path
    
    # Extract all the folders related to the subjects
    folders = os.listdir(input_dir)
    folders.sort()
    
    # We need only ones that have S in the name
    subjects = []
    for name in folders:
        if name[0] != "S":
            continue
        subjects.append(name)
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Taking pre-determined train/val/test splits from Pulse-PPG (binary classification)
    process_split(input_dir, output_dir, subjects[:11], split_type='train_binary', classification_type='binary')
    process_split(input_dir, output_dir, subjects[11:13], split_type='val_binary', classification_type='binary')
    process_split(input_dir, output_dir, subjects[13:15], split_type='test_binary', classification_type='binary')
    
    process_split(input_dir, output_dir, subjects[:11], split_type='train_multiclass', classification_type='multiclass')
    process_split(input_dir, output_dir, subjects[11:13], split_type='val_multiclass', classification_type='multiclass')
    process_split(input_dir, output_dir, subjects[13:15], split_type='test_multiclass', classification_type='multiclass')
    
    # Finally, write to HDF5
    to_do = ['train_binary', 'val_binary', 'test_binary', 'train_multiclass', 'val_multiclass', 'test_multiclass']
    for td in to_do:
        if os.path.exists(output_dir + '/' + td + '.h5'):
            print(f"File {td}.h5 already exists!")
        else:
            print(f"Creating file {td}.h5.")
            create_hdf5(output_dir + "/" + td, output_dir + "/" + td + ".h5")
            
    
    
    
