Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Dataset Preparation Scripts

This directory contains scripts for processing raw datasets into formats suitable for efficient training and evaluation.

## TUEG preparation (`make_tueg_bipolar.py`)

We provide an example script on how to format TUEG into bipolar. This is then used by our pre-training scripts. Make sure to change all of the occurences of `#CHANGEME`. The main processing script follow a very similar pattern to the one explained for `process_raw_eeg.py` here below.

**Prerequisites**:
-   You must have the raw TUEG dataset downloaded from their official sources.

**Command-Line Usage**:
The script takes the dataset name, the path to the raw data, and an output directory as arguments.

```bash
python make_datasets/make_tueg_bipolar.py --in_path /path/to/raw_data --hdf5_file_path /path/to/processed_data
```

### Arguments:
-   `--in_path`: The absolute path to the root directory of the downloaded raw EDF files for the TUEG dataset.
-   `--hdf5_file_path`: The directory where the script will save the procsessed `.h5` files.

---

## Raw EEG Preprocessing (`process_raw_eeg.py`)

Before the datasets can be converted to HDF5, the raw EDF files must first be preprocessed, windowed, and saved as intermediate pickle files. The `process_raw_eeg.py` script handles this entire pipeline.

### Key Processing Steps
1.  **Loads EDF files**: Reads the raw EEG data.
2.  **Selects and Orders Channels**: Standardizes all recordings to a 21-channel layout.
3.  **Applies Filters**: A band-pass filter (0.1-75.0 Hz) and a 60 Hz notch filter are applied.
4.  **Resamples Data**: All data is resampled to 256 Hz.
5.  **Creates Bipolar Montage**: Re-references the signals to the standard TCP bipolar montage.
6.  **Windows Data**: Segments the continuous signal into 5-second windows.
7.  **Saves Segments**: Each 5-second window is saved as a separate `.pkl` file.

### How to Use the Script

The script is designed to be run from the command line and requires you to specify which dataset you are processing.

**Prerequisites**:
-   You must have the raw TUH datasets (TUAB, TUSL, or TUAR) downloaded from their official sources.
-   The required Python packages, including `mne`, `numpy`, and `tqdm`, must be installed.

**Command-Line Usage**:
The script takes the dataset name, the path to the raw data, and an output directory as arguments.

```bash
python make_datasets/process_raw_eeg.py <dataset_name> --root_dir /path/to/raw_data --output_dir /path/to/processed_data
```
### Arguments:
-   `<dataset_name>`: The dataset to process. Must be one of `tuab`, `tusl`, or `tuar`.
-   `--root_dir`: The absolute path to the root directory of the downloaded raw EDF files for the specified dataset.
-   `--output_dir`: The directory where the script will save the processed `.pkl` files. The script will automatically create `train`, `val`, and `test` subdirectories inside a `processed` folder here.
-   `--processes`: (optional): The number of CPU cores to use for parallel processing (default is 24).
### Example
To process the TUAB dataset:
```bash
python make_datasets/process_raw_eeg.py tuab --root_dir /eeg_data/TUAB/edf --output_dir /processed_eeg/TUAB_data
```

## HDF5 Conversion (`make_hdf5.py`)
After you have generated the processed `.pkl` files using the script above, the `make_hdf5.py` script is used to bundle these thousands of small pickle files into large, efficient HDF5 (`.h5`) files. This is the final step before training.
### How to use the Script
1.  **Set the Data Path**: Open `make_hdf5.py` and modify `prepath` variable to point to the directory containing the processed data folders (e.g., `/processed_eeg/`).
2.  **Run the Script**: 
```bash
python make_datasets/make_hdf5.py
```
The script will then create the final `.h5` files needed for the `HDF5Loader` in your training pipeline.