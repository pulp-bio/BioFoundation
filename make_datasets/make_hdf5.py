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
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import shutil

def create_hdf5(source_dir, target_file, finetune=True,group_size=1000):
    """
    Creates an HDF5 file from a directory of pickle files.
    Includes error handling for corrupt files.
    """
    files = os.listdir(source_dir)
    data_group = []
    # Filter out non-pickle files
    files = [f for f in files if f.endswith('.pkl')]
    
    with h5py.File(target_file, 'w') as h5f:
        for i, file in enumerate(tqdm(files, desc=f"Creating {target_file}")):
            with open(os.path.join(source_dir, file), 'rb') as f:
                try:
                    sample = pickle.load(f)
                    data_group.append(sample)
                except (pickle.UnpicklingError, EOFError) as e:
                    print(f"Warning: Skipping corrupt pickle file {file}: {e}")
                    continue
                
                if (i + 1) % group_size == 0 or i == len(files) - 1:
                    if not data_group:
                        continue
                    
                    grp = h5f.create_group(f"data_group_{i // group_size}")
                    
                    try:
                        X_data = np.array([s['X'] for s in data_group])
                        grp.create_dataset("X", data=X_data)
                    except KeyError:
                        print(f"Error: 'X' key missing in data group {i // group_size}. Skipping group.")
                        del h5f[f"data_group_{i // group_size}"]
                        data_group = []
                        continue
                    except Exception as e:
                        print(f"Error packing X data for group {i // group_size}: {e}")
                        del h5f[f"data_group_{i // group_size}"]
                        data_group = []
                        continue

                    if(finetune):
                        try:
                            y_data = np.array([s['y'] for s in data_group])
                            grp.create_dataset("y", data=y_data)
                        except KeyError:
                            print(f"Error: 'y' key missing in finetune mode for group {i // group_size}. Skipping group.")
                            del h5f[f"data_group_{i // group_size}"]
                        except Exception as e:
                             print(f"Error packing y data for group {i // group_size}: {e}")
                             del h5f[f"data_group_{i // group_size}"]
                    
                    data_group = []

def process_dataset(prepath, dataset_name, splits, finetune, remove_pkl):
    """
    Helper function to process a single dataset.
    prepath: The root directory for data
    dataset_name: e.g., 'TUAR_data'
    splits: list of splits, e.g., ['train', 'val']
    finetune: boolean, whether to save 'y' labels
    remove_pkl: boolean, whether to delete the processed pkl directory
    """
    print(f"--- Processing {dataset_name} ---")
    
    # Path to the directory containing all split folders (train/, val/, test/)
    processed_dir_path = os.path.join(prepath, dataset_name, "processed")

    for td in splits:
        target_h5_file = os.path.join(prepath, dataset_name, f"{td}.h5")
        source_pickle_dir = os.path.join(processed_dir_path, td) # Use processed_dir_path

        if os.path.exists(target_h5_file):
            print(f"{dataset_name} {td}.h5 already exists. Skipping...")
        elif not os.path.isdir(source_pickle_dir):
            print(f"Source directory not found: {source_pickle_dir}")
            print(f"Skipping {dataset_name} {td}.h5")
        else:
            # Ensure target directory exists
            os.makedirs(os.path.dirname(target_h5_file), exist_ok=True)
            create_hdf5(source_pickle_dir, target_h5_file, finetune=finetune)

    # After processing all splits for this dataset, remove the pkl directory if requested
    if remove_pkl:
        if os.path.isdir(processed_dir_path):
            print(f"Removing processed .pkl directory: {processed_dir_path}")
            try:
                shutil.rmtree(processed_dir_path)
                print(f"Successfully removed {processed_dir_path}")
            except Exception as e:
                print(f"Error removing {processed_dir_path}: {e}")
        else:
            print(f"Processed directory not found, cannot remove: {processed_dir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HDF5 files from processed .pkl files.")
    parser.add_argument(
        "--prepath",
        type=str,
        required=True,
        help="The root directory containing the processed dataset folders (e.g., TUAR_data, TUSL_data)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="All",
        choices=["TUAR", "TUSL", "TUAB", "All"],
        help="Which dataset to process. 'All' processes all three."
    )
    parser.add_argument(
        "--remove_pkl",
        action="store_true",
        help="If set, removes the 'processed' directory (containing .pkl files) after HDF5 creation."
    )
    args = parser.parse_args()

    # Define the splits we expect for ALL datasets
    all_splits = ['train', 'val', 'test']
    
    datasets_to_process = []
    if args.dataset == "All":
        datasets_to_process = ["TUAR_data", "TUSL_data", "TUAB_data"]
    elif args.dataset == "TUAR":
        datasets_to_process = ["TUAR_data"]
    elif args.dataset == "TUSL":
        datasets_to_process = ["TUSL_data"]
    elif args.dataset == "TUAB":
        datasets_to_process = ["TUAB_data"]

    # Loop through the selected datasets and process them
    for data_folder_name in datasets_to_process:
        process_dataset(
            args.prepath,
            dataset_name=data_folder_name,
            splits=all_splits,
            finetune=True,
            remove_pkl=args.remove_pkl
        )

    print("HDF5 creation complete.")