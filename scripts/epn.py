import glob
import json
import os
import sys

import h5py
import numpy as np
import scipy.signal as signal
from joblib import Parallel, delayed
from scipy.signal import iirnotch
from tqdm.auto import tqdm

# Sampling frequency and EMG channels
tfs, n_ch = 200.0, 8

# Gesture label mapping
gesture_map = {
    "noGesture": 0,
    "waveIn": 1,
    "waveOut": 2,
    "pinch": 3,
    "open": 4,
    "fist": 5,
    "notProvided": 6,
}


# Filtering utilities
def bandpass_filter_emg(emg, low=20.0, high=90.0, fs=tfs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return signal.filtfilt(b, a, emg, axis=1)


def notch_filter_emg(emg, notch=50.0, Q=30.0, fs=tfs):
    w0 = notch / (0.5 * fs)
    b, a = iirnotch(w0, Q)
    return signal.filtfilt(b, a, emg, axis=1)


# Normalization helpers
def zscore_per_channel(emg):
    mean = emg.mean(axis=1, keepdims=True)
    std = emg.std(axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (emg - mean) / std


def adjust_length(x, max_len):
    n_ch, seq_len = x.shape
    if seq_len >= max_len:
        return x[:, :max_len]
    pad = np.zeros((n_ch, max_len - seq_len), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


# Single-sample processing


def extract_emg_signal(sample, seq_len):
    emg = np.stack([v for v in sample["emg"].values()], dtype=np.float32) / 128.0
    emg = bandpass_filter_emg(emg, 20.0, 90.0)
    emg = notch_filter_emg(emg, 50.0, 30.0)
    emg = zscore_per_channel(emg)
    emg = adjust_length(emg, seq_len)
    label = gesture_map.get(sample.get("gestureName", "notProvided"), 6)
    return emg, label


# Process one user JSON for train/validation


def process_user_training(path, seq_len):
    train_X, train_y, val_X, val_y = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data.get("trainingSamples", {}).values():
        emg, lbl = extract_emg_signal(sample, seq_len)
        if lbl != 6:
            train_X.append(emg)
            train_y.append(lbl)
    for sample in data.get("testingSamples", {}).values():
        emg, lbl = extract_emg_signal(sample, seq_len)
        if lbl != 6:
            val_X.append(emg)
            val_y.append(lbl)
    return train_X, train_y, val_X, val_y


# Process one user JSON for testing split


def process_user_testing(path, seq_len):
    train_X, train_y, test_X, test_y = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    buckets = {g: [] for g in gesture_map}
    for sample in data.get("trainingSamples", {}).values():
        buckets.setdefault(sample.get("gestureName", "notProvided"), []).append(sample)
    for samples in buckets.values():
        for i, sample in enumerate(samples):
            emg, lbl = extract_emg_signal(sample, seq_len)
            if lbl == 6:
                continue
            if i < 10:
                train_X.append(emg)
                train_y.append(lbl)
            else:
                test_X.append(emg)
                test_y.append(lbl)
    return train_X, train_y, test_X, test_y


# Save to HDF5
def save_h5(path, data, labels):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.asarray(data, np.float32))
        f.create_dataset("label", data=np.asarray(labels, np.int64))


# Main parallelized pipeline
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--source_training", required=True)
    parser.add_argument("--source_testing", required=True)
    parser.add_argument("--dest_dir", required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()
    data_dir = args.data_dir
    os.makedirs(args.dest_dir, exist_ok=True)

    # download data if requested
    if args.download_data:
        # https://zenodo.org/records/4421500
        url = "https://zenodo.org/records/4421500/files/EMG-EPN612%20Dataset.zip?download=1"
        os.system(f"wget -O {data_dir}/EMG-EPN612_Dataset.zip {url}")
        os.system(f"unzip -o {data_dir}/EMG-EPN612_Dataset.zip -d {data_dir}")
        # move the contents one level up
        os.system(f"mv {data_dir}/EMG-EPN612_Dataset/* {data_dir}/")
        os.system(f"rmdir {data_dir}/EMG-EPN612_Dataset")
        # clean up zip file
        os.system(f"rm {data_dir}/EMG-EPN612_Dataset.zip")
        print(f"Downloaded and unzipped dataset\n{data_dir}/EMG-EPN612_Dataset.zip")
        sys.exit("Data downloaded and unzipped. Rerun without --download_data.")

    seq_len = args.window_size
    train_X, train_y, val_X, val_y, test_X, test_y = [], [], [], [], [], []

    paths = glob.glob(os.path.join(args.source_training, "user*", "user*.json"))

    # Parallel process training JSONs
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_user_training)(p, seq_len)
        for p in tqdm(paths, desc="Training files")
    )
    for tX, ty, vX, vy in results:
        train_X.extend(tX)
        train_y.extend(ty)
        val_X.extend(vX)
        val_y.extend(vy)

    # Parallel process testing JSONs
    test_results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_user_testing)(p, seq_len)
        for p in tqdm(
            glob.glob(os.path.join(args.source_testing, "user*", "user*.json")),
            desc="Testing files",
        )
    )
    for tX, ty, teX, tey in test_results:
        train_X.extend(tX)
        train_y.extend(ty)
        test_X.extend(teX)
        test_y.extend(tey)

    # Save datasets
    save_h5(os.path.join(args.dest_dir, "train.h5"), train_X, train_y)
    save_h5(os.path.join(args.dest_dir, "val.h5"), val_X, val_y)
    save_h5(os.path.join(args.dest_dir, "test.h5"), test_X, test_y)

    # Print distributions
    for split, X, y in [
        ("Train", train_X, train_y),
        ("Val", val_X, val_y),
        ("Test", test_X, test_y),
    ]:
        arr = np.array(y)
        uniq, cnt = np.unique(arr, return_counts=True)
        uniq = [i.item() for i in uniq]
        cnt = [i.item() for i in cnt]
        print(f"{split} → total={len(y)}, classes={{}}".format(dict(zip(uniq, cnt))))


if __name__ == "__main__":
    main()
