Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Dataset Preparation Scripts

This directory contains scripts for processing raw datasets into formats suitable for efficient training and evaluation.

Two kinds of pipeline live here. The HDF5 pipelines described first are used by LUNA, FEMBA, LuMamba, PanLUNA and TinyMyo. S-CEReBrO uses LMDB pipelines whose samples also carry each channel's electrode coordinates; they are described in [S-CEReBrO Dataset Preprocessing (LMDB)](#s-cerebro-dataset-preprocessing-lmdb). Neither reads or writes the other's files, so both can be prepared side by side.

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

Before the datasets can be converted to HDF5, the raw EDF files must first be preprocessed, windowed, and saved as intermediate
pickle files. The `process_raw_eeg.py` script handles this entire pipeline.

**Key Processing Steps**
1. **Loads EDF files**: Reads the raw EEG data.
2. **Selects and orders channels**: Standardizes all recordings to a 21‑channel layout.
3. **Applies filters**: A band‑pass filter (0.1–75.0 Hz) and a 60 Hz notch filter are applied.
4. **Resamples data**: All data is resampled to 256 Hz.
5. **Creates bipolar montage**: Re‑references the signals to the standard TCP bipolar montage.
6. **Windows data**: Segments the continuous signal into 5‑second windows.
7. **Generates labels**: Applies labels based on the dataset and specified mode:
   - **tuab**: Assigns a single file‑level label (normal/abnormal) to all windows from that file.
    - **tusl**: Assigns a segment‑level label (`bckg`, `seiz`, `slow`) based on the majority label found in the 5‑second window.
    - **tuar**: Assigns segment‑level artifact labels based on the chosen `--mode` (e.g., a single 0/1 for **Binary** or a per‑channel array for **MultiLabel**).
8. **Saves segments**: Each 5‑second window (data + label) is saved as a separate `.pkl` file.

**How to Use the Script**
The script is designed to be run from the command line and requires you to specify which dataset you are processing.

**Prerequisites**
- You must have the raw **TUH** datasets (**TUAB**, **TUSL**, or **TUAR**) downloaded from their official sources.
- The required Python packages, including `mne`, `numpy`, `pandas`, and `tqdm`, must be installed.

**Command‑Line Usage**
```bash
python make_datasets/process_raw_eeg.py <dataset_name>       --root_dir /path/to/raw_data       --output_dir /processed_eeg [options]
```

**Arguments**
- `<dataset_name>`: The dataset to process. Must be one of `tuab`, `tusl`, or `tuar`.
- `--root_dir`: The absolute path to the root directory of the downloaded raw EDF files for the specified dataset.
- `--output_dir`: The directory where the script will save the processed `.pkl` files. The script will automatically create `train`, `val`, and `test` subdirectories inside a `processed` folder here.
- `--processes`: (Optional) The number of CPU cores to use for parallel processing (default: `24`).
- `--mode`: (Optional) Specifies the labeling mode for the **TUAR** dataset only. Must be one of `Binary`, `MultiBinary`, or `MultiLabel` (default: `Binary`).

**Examples**
- To process the **TUAB** dataset:
  ```bash
  python make_datasets/process_raw_eeg.py tuab         --root_dir /eeg_data/TUAB/edf         --output_dir /processed_eeg
  ```

- To process the **TUAR** dataset using the **MultiLabel** mode:
  ```bash
  python make_datasets/process_raw_eeg.py tuar         --root_dir /eeg_data/TUAR/edf         --output_dir /processed_eeg       --mode MultiLabel
  ```

---

## HDF5 Conversion (`make_hdf5.py`)

After you have generated the processed `.pkl` files using the script above, the `make_hdf5.py` script is used to bundle these
thousands of small pickle files into large, efficient HDF5 (`.h5`) files. This is the final step before training.

**How to Use the Script**
The script is run from the command line and takes the path to your processed data as an argument.

**Command‑Line Usage**
```bash
python make_datasets/make_hdf5.py --prepath /processed_eeg [options]
```

**Arguments**
- `--prepath` (Required): The absolute path to the directory containing your dataset folders (e.g., `/processed_eeg/`). This directory
  should contain the `TUAB_data`, `TUSL_data`, etc., folders created by `process_raw_eeg.py`, so this should mirror the `--output_dir` used previously.
- `--dataset` (Optional): Which dataset to process. Choices: `TUAR`, `TUSL`, `TUAB`, `All` (default: `All`).
- `--remove_pkl` (Optional): If included, this flag will delete the `processed` directory (containing all the intermediate `.pkl` files)
  for a dataset after its `.h5` files are successfully created.

**Examples**
- To convert all datasets found in `/processed_eeg` and remove the `.pkl` files after conversion:
  ```bash
  python make_datasets/make_hdf5.py --prepath /processed_eeg --dataset All --remove_pkl
  ```

- To convert only the **TUAB** dataset and keep the original `.pkl` files:
  ```bash
  python make_datasets/make_hdf5.py --prepath /processed_eeg --dataset TUAB
  ```

  ## Non-TUH Dataset Preprocessing 

  For **non-TUH datasets for LuMamba**, preprocessing and export scripts are provided for:
  - [**MoBI**](https://springernature.figshare.com/articles/dataset/A_mobile_brain-body_imaging_dataset_recorded_during_treadmill_walking_with_a_brain-computer_interface/5807511?backTo=%2Fcollections%2FA_mobile_brain-body_imaging_dataset_recorded_during_treadmill_walking_with_a_brain-computer_interface%2F3894013&file=10263603) (gait prediction)
  - [**MODMA**](https://modma.lzu.edu.cn/data/index/) (depression detection)
  - [**Mumtaz2016**](https://figshare.com/articles/dataset/EEG_Data_New/4244171) (depression detection)
  - [**TDBrain**](https://brainclinics.com/resources/) (Parkinson's detection)
  - [**APAVA**](https://osf.io/jbysn/overview) (Alzheimer's detection)

  For **non-TUH datasets for PanLUNA**, preprocessing and export scripts are provided for:

  - Pretraining Datasets:
     - [**Siena Scalp EEG Database**](https://physionet.org/content/siena-scalp-eeg/1.0.0/) - 29 channel EEG dataset.
     - [**MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset**](https://physionet.org/content/mimic-iv-ecg/1.0/) - 12 channel ECG dataset.
     - [**CODE-15%**](https://zenodo.org/records/4916206) - 12 channel ECG dataset.
     - [**PulseDB**](https://github.com/pulselabteam/PulseDB) - 1 ECG channel and 1 PPG channel.
  - Fine-tuning Dataset:
     - [**HMC**](https://physionet.org/content/hmc-sleep-staging/1.1/) - 4 EEG channels and 1 ECG channel.
     - [**SEED-VII**](https://bcmi.sjtu.edu.cn/home/seed/seed-vii.html) - 62 EEG channels and 1 ECG channel.
     - [**PTB-XL**](https://physionet.org/content/ptb-xl/1.0.3/) - 12 ECG channels.
     - [**CSN**](https://physionet.org/content/ecg-arrhythmia/1.0.0/) - 12 ECG channels.
     - [**CPSC 2018**](https://www.kaggle.com/datasets/physionet/china-physiological-signal-challenge-in-2018) - 12 ECG channels.
     - [**WESAD**](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset) - 1 ECG and 1 PPG channel.
  
  Scripts can be found in `BioFoundation/make_datasets/` under script names corresponding to the datasets.

  Scripts handle both **preprocessing** and **HDF5 conversion**.

  **How to Use the Script**
  - Download the dataset of your choice (see link to dataset and user agreement forms).
  - Select the script corresponding to the dataset of your choice, e.g. use `make_tdbrain_dataset.py` if you want to preprocess and export the TDBRain dataset to HDF5.
  - The script is run from the command line and takes the path to the raw data directory `prepath` (the original download folder) and `output_path`, the export directory containing your dataset folders (e.g. `./processed_eeg/`).
  - The script will automatically handle train/test/val splits and create a dataset subfolder in `output_path` containing preprocessed train/test/val HDF5 files.

  **Command-Line Usage**
  To preprocess and convert the APAVA dataset, where raw .mat files are in `/raw_data/APAVA`:
    ```bash
    python make_datasets/make_apava_dataset.py --prepath "/raw_data/APAVA" --output_path "/processed_eeg"
    ```

  **Notes on file naming**
  - The `make_<dataset_name>_dataset.py` scripts were tailored to most recent dataset versions on official platforms. File naming and format assumptions in the scripts may have to be adjusted in future versions.
  - For **Mumtaz2016**, file naming errors were corrected manually (see `(!!!) Important` tags in `make_mumtaz_dataset.py` to perform the necessary checks).
  - For **PTB-XL**, **CSN** and **CPSC 2018** datasets, go to [MERL ICML 2024 repository](https://github.com/cheliu-computation/MERL-ICML2024/tree/main/finetune/data_split) and download `csv` files providing `train/val/test` splits and provide the path to the folder storing these `csv` files under `--csv_files_dir` argument in corresponding scripts.
  

---

## S-CEReBrO Dataset Preprocessing (LMDB)

S-CEReBrO reads LMDB files in which every sample carries, besides the signal, the 3D coordinates of both electrodes of each channel. It therefore has its own preprocessing script for every corpus it uses, including TUAB and TUEG, which exist twice in this directory: once as HDF5 above, for the other model families, and once as LMDB here. The two never read or write each other's files.

**How to Use the Scripts**
- Download the dataset from the source listed below, accepting its licence or data-use agreement where one is required. Access terms change; check each source before downloading.
- Run the matching script as a module from the repository root, so the shared electrode table in `common.py` resolves:
  ```bash
  python -m make_datasets.<script> --input_dir /path/to/raw --output_dir /path/to/output
  ```
- Fine-tuning scripts write `train.lmdb`, `val.lmdb` and `test.lmdb`, each with an index file (`train.meta.json` and so on). The split is fixed inside each script, at subject or recording level, so it is identical for every run and every model. It is the split of the reference implementation's CBraMod mode, except where a dataset section says otherwise.
- Pre-training scripts write one pooled `all.lmdb` with `all.meta.json`; the pre-training data module holds out its own validation fraction. TUEG writes `TUEG.lmdb` and `TUEG_keys.txt` instead.
- The S-CEReBrO configs read each output from the default location in the tables below. Pass `dataset_root=/path` for one fine-tuning corpus, or `pretraining_root=/path` for all pre-training corpora, to read them from elsewhere.

**Arguments shared by every script except `unrar_isruc.py`**

| Argument | Meaning |
| --- | --- |
| `--input_dir` | Required. The raw dataset folder described in each dataset's section. |
| `--output_dir` | Required. Where the LMDB files are written; created if missing. |
| `--num_workers` | Parallel worker processes (default `8`). `1` runs in the calling process, which keeps tracebacks readable when checking a new download; `make_tueg.py` always uses separate worker processes. |
| `--dry_run` | Process every recording and report what would be written, without writing anything. |

Script-specific arguments are listed in each dataset's section.

**Dry run**

`--dry_run` reads and processes every recording exactly as a real run would, but writes nothing and does not create the output folder. At the end, the scripts that write `train`/`val`/`test` or `all` LMDBs print the sample count per split, the number of subjects, the label distribution and the window shapes, and warn about any subject that appears in two splits. `make_tueg.py` prints the number of windows, the storage a real run needs and the `--num_windows` value to give it, and lists every recording it would skip; `make_chb_mit_pickles.py` prints the number of recordings it would convert. A dry run takes as long as a real run minus the writing. It is the way to check a new download before committing disk space.

**Pre-training datasets**

| Corpus | Script | Default location | Access |
| --- | --- | --- | --- |
| TUEG | `make_tueg.py` | `$DATA_PATH/pretraining/TUEG` | Registration and data-use agreement |
| SEED, SEED-IV | `make_seed.py`, once per corpus | `$DATA_PATH/pretraining/SEED`, `$DATA_PATH/pretraining/SEED-IV` | Licence agreement with BCMI |
| SEED-FRA, SEED-GER | `make_seed_fra_ger.py`, once per corpus | `$DATA_PATH/pretraining/SEED-FRA`, `$DATA_PATH/pretraining/SEED-GER` | Licence agreement with BCMI |
| BOAS | `make_boas.py` | `$DATA_PATH/pretraining/BOAS` | Open access |
| GWD | `make_gwd.py` | `$DATA_PATH/pretraining/GWD` | See the download page |
| Sleep-EDFx | `make_sleepedfx.py` | `$DATA_PATH/pretraining/SleepEDFx` | Open access |
| BCI Challenge @ NER 2015 | `make_bci_ner.py` | `$DATA_PATH/pretraining/BCI-NER` | Kaggle account, accepting the competition rules |

**Fine-tuning datasets**

| Dataset | Script | Default location | Access |
| --- | --- | --- | --- |
| TUAB | `make_tuab.py` | `$DATA_PATH/finetuning/TUAB` | Registration and data-use agreement |
| CHB-MIT | `make_chb_mit_pickles.py`, then `make_chb_mit.py` | `$DATA_PATH/finetuning/CHB-MIT` | Open access |
| Neonate | `make_neonate.py` | `$DATA_PATH/finetuning/Neonate` | Open access |
| PhysioNet-MI | `make_physionet_mi.py` | `$DATA_PATH/finetuning/PhysioNet-MI` | Open access |
| SHU-MI | `make_shu_mi.py` | `$DATA_PATH/finetuning/SHU-MI` | Open access |
| STEW | `make_stew.py` | `$DATA_PATH/finetuning/STEW` | IEEE DataPort account |
| Mumtaz2016 | `make_mumtaz.py` | `$DATA_PATH/finetuning/Mumtaz` | Open access |
| MentalArithmetic | `make_mental_arithmetic.py` | `$DATA_PATH/finetuning/MentalArithmetic` | Open access |
| SEED-V | `make_seed_v.py` | `$DATA_PATH/finetuning/SEED-V` | Licence agreement with BCMI |
| ISRUC | `unrar_isruc.py`, then `make_isruc.py` | `$DATA_PATH/finetuning/ISRUC` | Open access |
| SEED-VIG | `make_seed_vig.py` | `$DATA_PATH/finetuning/SEED-VIG` | Licence agreement with BCMI |

### Pre-training corpora

**TUEG** (`make_tueg.py`)
- **Download**: [TUH EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/).
- **Input**: the corpus root. Every `.edf` below it is read. Recordings under a `02_tcp_le` or `04_tcp_le` folder are treated as referenced to the linked ears, all others to the average.
- **Writes**: 30 s windows at 200 Hz with up to 64 channels, each padded to 64, into `TUEG.lmdb`; the window keys, sorted, into `TUEG_keys.txt`; and every recording that yielded no windows, with the reason, into `TUEG_skipped.tsv`.
- **Arguments**:
  - `--tuab_dir`: the raw TUAB release. Every subject found there is excluded, so pre-training cannot see a TUAB subject.
  - `--allow_tuab_overlap`: pre-train on all TUEG subjects instead. Use it only when TUAB is not among your evaluation datasets; one of the two is required.
  - `--num_windows`: size the database for this many windows, such as the count a dry run printed, instead of reading every EDF header first.
  - `--map_size_gb`: reserve exactly this many GB (10^9 bytes) instead.
- **Storage**: each window takes 1,537,536 bytes, so 1,000,000 windows need about 1.54 TB of disk and 2,619,175 windows about 4.03 TB. The database reserves 10% more than that in address space, but disk is used only as data is written. Make sure the output filesystem, and your quota on it, holds the data, and that the shell's virtual-memory limit (`ulimit -v`) is unlimited, because the database is memory-mapped.
- **Files that cannot be used** are skipped with their reason, printed and listed in `TUEG_skipped.tsv`, and the run goes on: a file that cannot be read, filtered or resampled (such as one sampled at 150 Hz or less, too slow for the 75 Hz low-pass), one without any known channel or shorter than 30 s, one whose name is not usable as a key or repeats another file's name, and one whose worker process dies twice, for instance by running out of memory.
- **Memory**: each worker holds one whole recording several times over while filtering and resampling, roughly 1 GB per hour of recording, and at most two recordings per worker wait to be written. Choose `--num_workers` so that this fits.
- **Stopping**: Ctrl-C, the SIGTERM a cluster sends at a job's time limit, or a full database stop the run with `TUEG_keys.txt` listing exactly the windows committed so far. A run cannot be resumed, and the script refuses to write into an existing `TUEG.lmdb`; delete it before running again.
  ```bash
  python -m make_datasets.make_tueg --input_dir /raw_data/TUEG --output_dir $DATA_PATH/pretraining/TUEG \
      --tuab_dir /raw_data/TUAB/edf --num_workers 16 --num_windows 2619175
  ```

**SEED and SEED-IV** (`make_seed.py`)
- **Download**: [SEED](https://bcmi.sjtu.edu.cn/home/seed/) and [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html).
- **Input**: a folder of `.mat` files, searched recursively. Every 62-channel array in each file is used as one recording. The arrays must already be sampled at 200 Hz, as in SEED's preprocessed release; the script does not resample.
- **Writes**: 30 s windows, 62 channels, into `all.lmdb`. Run the script once for SEED and once for SEED-IV, each into its own output folder.
  ```bash
  python -m make_datasets.make_seed --input_dir /raw_data/SEED --output_dir $DATA_PATH/pretraining/SEED
  ```

**SEED-FRA and SEED-GER** (`make_seed_fra_ger.py`)
- **Download**: [SEED-FRA](https://bcmi.sjtu.edu.cn/home/seed/seed-FRA.html) and [SEED-GER](https://bcmi.sjtu.edu.cn/home/seed/seed-GER.html).
- **Input**: a folder of Neuroscan `.cnt` files, searched recursively. Recordings missing any of the 62 channels are skipped.
- **Writes**: 30 s windows at 200 Hz, 62 channels, into `all.lmdb`. Run once per corpus.

**BOAS** (`make_boas.py`)
- **Download**: [BOAS on NEMAR](https://nemar.org/dataset/on005555).
- **Input**: the downloaded folder, searched recursively for `.edf` files. Recordings holding the two headband channels `HB_1` and `HB_2` are used, with those placed at AF7 and AF8; the others, such as the PSG recordings, are skipped.
- **Writes**: 30 s windows at 200 Hz, 2 channels, into `all.lmdb`.

**GWD** (`make_gwd.py`)
- **Download**: [Bitbrain EEG datasets](https://www.bitbrain.com/science/eeg-datasets).
- **Input**: a folder of `.mat` files, searched recursively, each holding `signal` (channels by samples) and a `header` with `sample_rate`, `channels_eeg` and `channels_labels`. Channels whose label is not a known electrode are dropped.
- **Writes**: 30 s windows at 200 Hz into `all.lmdb`.

**Sleep-EDFx** (`make_sleepedfx.py`)
- **Download**: [Sleep-EDF Database Expanded](https://www.physionet.org/content/sleep-edfx/1.0.0/).
- **Input**: the downloaded folder, searched recursively for `.edf` files. Recordings with `EEG Fpz-Cz` and `EEG Pz-Oz` are used; hypnogram files are skipped.
- **Writes**: 30 s windows at 200 Hz, 2 channels, into `all.lmdb`.

**BCI Challenge @ NER 2015** (`make_bci_ner.py`)
- **Download**: [Kaggle competition data](https://www.kaggle.com/competitions/inria-bci-challenge/data).
- **Input**: a folder of session `.csv` files, searched recursively. Each file holds a time column, the electrode columns, `EOG` and `FeedBackEvent`, sampled at 200 Hz.
- **Writes**: 30 s windows of every column except the time, `EOG` and `FeedBackEvent`, into `all.lmdb`.

### Fine-tuning datasets

**TUAB** (`make_tuab.py`)
- **Download**: [TUH Abnormal EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuab).
- **Input**: the release's `edf` folder, containing `train/` and `eval/`, each with `normal/01_tcp_ar/` and `abnormal/01_tcp_ar/`.
- **Writes**: 10 s windows at 200 Hz in the 22-channel TCP bipolar montage, labelled normal (0) or abnormal (1).
- **Split**: the official `eval` set is the test split. Training subjects are divided 80/20 into train and validation by subject (seed 4523), so a subject with both normal and abnormal sessions stays in one split. The reference shuffled normal and abnormal subjects separately.
  ```bash
  python -m make_datasets.make_tuab --input_dir /raw_data/TUAB/edf --output_dir $DATA_PATH/finetuning/TUAB
  ```

**CHB-MIT** (`make_chb_mit_pickles.py`, then `make_chb_mit.py`)
- **Download**: [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/).
- **Input, step 1**: the downloaded folder, containing `chbXX/chbXX_NN.edf` and `chbXX/chbXX-summary.txt`. The step writes one pickle per recording into its `--output_dir`, keeping the folder per patient, and needs `pyedflib`.
- **Input, step 2**: the output folder of step 1, searched recursively for `.pkl` files.
- **Writes**: 10 s windows at 200 Hz in 16 bipolar channels, labelled seizure (1) or not (0).
- **Split**: chb21 and chb22 validate, chb23 and chb24 test, the other patients train.
- **Note**: this preprocessing deliberately replicates BIOT and CBraMod, including its known issues, so that results remain comparable with theirs; see [CHB-MIT note](#chb-mit-note).
  ```bash
  python -m make_datasets.make_chb_mit_pickles --input_dir /raw_data/CHB-MIT --output_dir /raw_data/CHB-MIT_pickles
  python -m make_datasets.make_chb_mit --input_dir /raw_data/CHB-MIT_pickles --output_dir $DATA_PATH/finetuning/CHB-MIT
  ```

**Neonate** (`make_neonate.py`)
- **Download**: [Neonatal EEG recordings with seizure annotations](https://zenodo.org/records/2547147).
- **Input**: the folder holding `eeg1.edf` to `eeg79.edf` and `annotations_2017_A.csv`, `annotations_2017_B.csv` and `annotations_2017_C.csv`, directly in it. These are the annotation file names of the release linked above; a later version of the dataset may rename them, in which case update `ANNOTATION_FILES` in `make_neonate.py` to match.
- **Writes**: 5 s windows at 200 Hz in an 18-channel bipolar montage. A second is labelled by the consensus of the three annotators. A window is a seizure (1) if any of its seconds is, and is skipped if any of its seconds has no consensus. The reference implementation instead labelled window *k* with second *k*, so results are not directly comparable with its preprocessing.
- **Split**: fixed subject lists in the script: 61 train, 8 validation, 10 test.

**PhysioNet-MI** (`make_physionet_mi.py`)
- **Download**: [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).
- **Input**: the folder holding the subject folders `S001` to `S109`, each with its runs as `S001R04.edf` and so on. Runs 04, 06, 08, 10, 12 and 14 are used.
- **Writes**: 4 s epochs at 200 Hz, 64 channels, in 4 classes: left fist, right fist, both fists, both feet.
- **Split**: subjects S001 to S070 train, S071 to S089 validate, S090 to S109 test.

**SHU-MI** (`make_shu_mi.py`)
- **Download**: [SHU dataset on figshare](https://figshare.com/articles/software/shu_dataset/19228725/1).
- **Input**: a folder of `.mat` files, searched recursively, each holding `data` (trials by 32 channels by samples) and `labels` (1 or 2).
- **Writes**: 4 s trials resampled to 200 Hz, 32 channels, 2 classes.
- **Split**: by recording file in sorted order: the first 75 train, the next 25 validate, the rest test.

**STEW** (`make_stew.py`)
- **Download**: [STEW on IEEE DataPort](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset).
- **Input**: the folder holding `sub01_lo.txt`, `sub01_hi.txt` and so on, and `ratings.txt`.
- **Writes**: 4 s windows with 2 s overlap at 200 Hz, 14 channels, cleaned with artifact subspace reconstruction (`asrpy`). The workload rating becomes low (1 to 3), medium (4 to 6) or high (7 to 9).
- **Split**: fixed subject lists in the script: 36 train, 4 validation, 5 test.

**Mumtaz2016** (`make_mumtaz.py`)
- **Download**: [Mumtaz2016 on figshare](https://figshare.com/articles/dataset/EEG_Data_New/4244171).
- **Input**: the folder holding the `.edf` files, named like `H S1 EC.edf` and `MDD S1 EO.edf`, directly in it. `TASK` recordings are ignored.
- **Writes**: 5 s windows at 200 Hz, 19 channels referenced to the linked ears, labelled healthy (0) or depressed (1).
- **Split**: by subject, separately for healthy and depressed subjects, in sorted order of subject name. The first 20 healthy and 21 depressed subjects train, the next 4 and 5 validate, and the rest test. The reference cut the file list instead; that gives the same split when every subject has both recordings.

**MentalArithmetic** (`make_mental_arithmetic.py`)
- **Download**: [EEG During Mental Arithmetic Tasks](https://physionet.org/content/eegmat/1.0.0/).
- **Input**: the folder holding `Subject00_1.edf`, `Subject00_2.edf` and so on, directly in it.
- **Writes**: 5 s windows at 200 Hz, 20 channels, labelled rest (`_1`, 0) or task (`_2`, 1).
- **Split**: by recording file in sorted order: the first 56 train, the next 8 validate, the rest test.

**SEED-V** (`make_seed_v.py`)
- **Download**: [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html).
- **Input**: the folder holding the raw `.cnt` sessions, named `<subject>_<session>_<date>.cnt`, directly in it.
- **Writes**: 4 s segments at 200 Hz from each of the 15 trials of a session, 62 channels, in 5 emotion classes.
- **Split**: subjects 1 to 10 train, 11 to 13 validate, 14 to 16 test.

**ISRUC** (`unrar_isruc.py`, then `make_isruc.py`)
- **Download**: [ISRUC-Sleep](https://sleeptight.isr.uc.pt/), subgroup I: the archives `1.rar` to `100.rar`.
- **Extraction**: `unrar_isruc.py` unpacks the archives into `<n>/<n>.rec` with their hypnograms. It takes `--input_dir`, the folder with the archives; `--output_dir`, which defaults to `--input_dir`; and `--start` and `--end`, the archive numbers, defaulting to 1 and 100. It needs the `rarfile` package and an extraction tool such as `unrar`; unpacking the archives with any such tool directly gives the same layout.
- **Input**: the extracted folder, with `<n>/<n>.rec` and `<n>/<n>_1.txt` (the first scorer's hypnogram). The `.rec` files are EDF and are read as such.
- **Writes**: sequences of 20 consecutive 30 s epochs at 200 Hz, 6 bipolar channels, with one of 5 sleep stages per epoch.
- **Split**: subjects 1 to 80 train, 81 to 90 validate, 91 to 100 test.
  ```bash
  python -m make_datasets.unrar_isruc --input_dir /raw_data/ISRUC
  python -m make_datasets.make_isruc --input_dir /raw_data/ISRUC --output_dir $DATA_PATH/finetuning/ISRUC
  ```

**SEED-VIG** (`make_seed_vig.py`)
- **Download**: [SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html).
- **Input**: the folder holding the EEG `.mat` files directly in it, such as the release's `Raw_Data`.
- **Arguments**: `--labels_dir`, required, the folder holding the PERCLOS `.mat` files with the same file names, such as `perclos_labels`.
- **Writes**: 8 s windows at 200 Hz, 17 channels, each with its PERCLOS value as a continuous target.
- **Split**: by recording file in sorted order: the first 15 train, the next 4 validate, the rest test.
  ```bash
  python -m make_datasets.make_seed_vig --input_dir /raw_data/SEED-VIG/Raw_Data \
      --labels_dir /raw_data/SEED-VIG/perclos_labels --output_dir $DATA_PATH/finetuning/SEED-VIG
  ```

### CHB-MIT note

CHB-MIT is prepared exactly as in [BIOT](https://github.com/ycq091044/BIOT) and [CBraMod](https://github.com/wjq-learning/CBraMod): `make_chb_mit_pickles.py` and `make_chb_mit.py` reproduce their `process1.py` and `process2.py`, so that results are comparable with those works. That preprocessing has known issues (see [CBraMod issue #29](https://github.com/wjq-learning/CBraMod/issues/29)). They are replicated on purpose and should not be corrected here, since doing so would break that comparability:

- chb12, chb13 and chb17 are never converted, because the reference's fixed table of patients and recordings leaves them out.
- Splits follow the recording folders, so chb21, a later session of chb01, puts one patient in both training and validation. The split summary reports this overlap.
- A window is labelled a seizure only if a seizure onset or offset falls inside it, so a window lying wholly within a long seizure is labelled 0.
- For every seizure, extra 10 s windows starting every 5 s, from 1 s before onset until 1 s after offset, are added and all labelled seizure, in every split including test. A window cut short by the end of a recording is resampled to full length.

Read and report CHB-MIT results as following this protocol, and cite BIOT and CBraMod alongside them.

### Notes on file naming

- The scripts follow the file naming of the current releases on the official platforms. A renamed or reorganised release may need the patterns in the script adjusted; run with `--dry_run` and `--num_workers 1` first to check that the expected recordings are found.
