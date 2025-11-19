import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal as signal
from joblib import Parallel, delayed
from scipy.signal import iirnotch
from tqdm import tqdm


# ==== Filter functions (operate at original fs=2000) ====
def notch_filter(data, notch_freq=50.0, Q=30.0, fs=2000.0):
    b, a = iirnotch(notch_freq, Q, fs)
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = signal.filtfilt(b, a, data[:, ch])
    return out


def bandpass_filter_emg(emg, lowcut=20.0, highcut=90.0, fs=2000.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    out = np.zeros_like(emg)
    for c in range(emg.shape[1]):
        out[:, c] = signal.filtfilt(b, a, emg[:, c])
    return out


# ==== Window segmentation ====
def process_emg_features(emg, window_size=1000, stride=500):
    segs, lbls = [], []
    N = len(emg)
    for start in range(0, N, stride):
        end = start + window_size
        if end > N:  # skip the last segment if it is not complete
            continue
        win = emg[start:end]
        segs.append(win)
    return np.array(segs)


def process_one_recording(file_path, fs=2000.0, window_size=1000, stride=500):
    """
    Process a single recording file to extract EMG features and labels
    as to be used in the main pipeline with parallel processing.
    """
    with h5py.File(file_path, "r") as f:
        grp = f["emg2pose"]
        data = grp["timeseries"]
        emg = data["emg"][:].astype(np.float32)

    # ==== Preprocessing EMG data ====
    emg_filt = bandpass_filter_emg(emg, 20, 450, fs=fs)
    emg_filt = notch_filter(emg_filt, 50, 30, fs=fs)

    # z-score
    mu = emg_filt.mean(axis=0)
    sd = emg_filt.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    emg_z = (emg_filt - mu) / sd

    # segment
    segs = process_emg_features(emg_z, window_size, stride)

    return segs


# ==== Main pipeline ====
def main():
    import argparse

    args = argparse.ArgumentParser(description="Process EMG data from DB5.")
    args.add_argument("--data_dir", type=str)
    args.add_argument("--save_dir", type=str)
    args.add_argument(
        "--window_size", type=int, help="Size of the sliding window for segmentation."
    )
    args.add_argument(
        "--stride", type=int, help="Stride for the sliding window segmentation."
    )
    args.add_argument(
        "--subsample", type=float, default=1.0, help="Whether to subsample the data"
    )
    args.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run. -1 means using all available cores.",
    )
    args.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = args.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    fs = 2000.0  # original sampling rate
    window_size, stride = args.window_size, args.stride

    df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    df = df.groupby("split").apply(
        lambda x: (
            x.sample(frac=args.subsample, random_state=args.seed)
            if args.subsample < 1.0
            else x
        )
    )
    df.reset_index(drop=True, inplace=True)

    splits = {}
    for split, df_ in df.groupby("split"):
        sessions = list(df_.filename)
        splits[split] = [
            Path(data_dir).expanduser().joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    all_data = {"train": [], "val": [], "test": []}

    for split, files in splits.items():
        # Here we use joblib to parallelize the file processing, each file is processed independently as the task is embarrassingly parallel. We scale the processing across all available CPU cores since the number of files is around 25k (with training being 17k).
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_one_recording)(file_path, fs, window_size, stride)
            for file_path in tqdm(files, desc=f"Processing {split} files")
        )
        # Collect results
        for segs in tqdm(results, desc=f"Collecting {split} data"):
            all_data[split].append(segs)

        # stack, augment train, transpose, save, and print stats
        X = np.concatenate(all_data[split], axis=0)  # [N, window_size, ch]

        # transpose to [N, ch, window_size]
        X = X.transpose(0, 2, 1)

        # save
        with h5py.File(os.path.join(save_dir, f"{split}.h5"), "w") as hf:
            hf.create_dataset("data", data=X)


if __name__ == "__main__":
    main()
