import os
import sys

import h5py
import numpy as np
import scipy.signal as signal
from scipy.signal import iirnotch


# ─────────────────────────────────────────────
# Filtering utilities
# ─────────────────────────────────────────────
def bandpass_filter_emg(emg, lowcut=20.0, highcut=90.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return signal.filtfilt(b, a, emg, axis=0)


def notch_filter_emg(emg, notch_freq=50.0, Q=30.0, fs=200.0):
    b, a = iirnotch(notch_freq / (0.5 * fs), Q)
    return signal.filtfilt(b, a, emg, axis=0)


# ─────────────────────────────────────────────
# Core I/O + preprocessing helpers
# ─────────────────────────────────────────────
def read_emg_txt(txt_path):
    """
    Read a txt file with columns: time ch1 … ch8 class.
    Return float32 array of shape (N, 10).
    """
    data = []
    with open(txt_path, "r") as f:
        for line in f.readlines()[1:]:  # skip header
            cols = line.strip().split()
            if len(cols) == 10:
                data.append(list(map(float, cols)))
    return np.asarray(data, dtype=np.float32)


def preprocess_emg(arr, fs=200.0, remove_class0=True):
    """
    1) optional removal of class-0 rows
    2) band-pass → notch → Z-score  (on 8 channels)
    """
    if remove_class0:
        arr = arr[arr[:, -1] >= 1]
    if arr.size == 0:
        return arr

    emg = arr[:, 1:9]  # (N, 8)
    emg = bandpass_filter_emg(emg, 20, 90, fs)
    emg = notch_filter_emg(emg, 50, 30, fs)

    mu = emg.mean(axis=0)
    sd = emg.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    emg = (emg - mu) / sd

    arr[:, 1:9] = emg
    return arr


def find_label_runs(arr):
    """Group consecutive rows with identical class labels."""
    runs = []
    if arr.size == 0:
        return runs
    curr_lbl = int(arr[0, -1])
    start = 0
    for i in range(1, len(arr)):
        lbl = int(arr[i, -1])
        if lbl != curr_lbl:
            runs.append((curr_lbl, arr[start:i]))
            curr_lbl, start = lbl, i
    runs.append((curr_lbl, arr[start:]))
    return runs


def sliding_window_majority(seg_arr, window_size=1000, stride=500):
    segs, labs = [], []
    for start in range(0, len(seg_arr) - window_size + 1, stride):
        win = seg_arr[start : start + window_size]
        maj = np.argmax(np.bincount(win[:, -1].astype(int)))
        segs.append(win[:, 1:9])  # keep 8-channel EMG
        labs.append(maj)
    return np.asarray(segs, dtype=np.float32), np.asarray(labs, dtype=np.int32)


# ─────────────────────────────────────────────
# Safe concatenation utilities
# ─────────────────────────────────────────────
def concat_data(lst):  # lst of (N,256,8)
    return np.concatenate(lst, axis=0) if lst else np.empty((0, 1000, 8), np.float32)


def concat_label(lst):
    return np.concatenate(lst, axis=0) if lst else np.empty((0,), np.int32)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser(description="Convert UCI EMG dataset to h5 format.")
    args.add_argument("--download_data", action="store_true")
    args.add_argument(
        "--data_dir", type=str, help="Root directory of the UCI EMG dataset"
    )
    args.add_argument(
        "--save_dir", type=str, help="Directory to save the output h5 files"
    )
    args.add_argument("--window_size", type=int, help="Window size for sliding window")
    args.add_argument("--stride", type=int, help="Stride for sliding window")
    args = args.parse_args()

    data_root = args.data_dir
    save_root = args.save_dir
    os.makedirs(save_root, exist_ok=True)

    # download data if requested
    if args.download_data:
        # https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures
        base_url = (
            "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
        )
        # download and unzip
        os.system(f"wget -O {data_root}/emg_gestures.zip '{base_url}'")
        os.system(f"unzip -o {data_root}/emg_gestures.zip -d {data_root}")
        os.system(f"rm {data_root}/emg_gestures.zip")
        print(f"Downloaded and unzipped dataset\n{data_root}/emg_gestures.zip")
        sys.exit("Data downloaded and unzipped. Rerun without --download_data.")

    fs = 200.0  # sampling rate of MYO bracelet
    window_size, stride = args.window_size, args.stride

    split_map = {
        "train": list(range(1, 25)),  # 1–24
        "val": list(range(25, 31)),  # 25–30
        "test": list(range(31, 37)),  # 31–36
    }

    datasets = {k: {"data": [], "label": []} for k in split_map}

    for subj in range(1, 37):
        subj_dir = os.path.join(data_root, f"{subj:02d}")
        if not os.path.isdir(subj_dir):
            continue
        split_key = next(k for k, v in split_map.items() if subj in v)

        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith(".txt"):
                continue
            arr = read_emg_txt(os.path.join(subj_dir, fname))
            arr = preprocess_emg(arr, fs)

            for lbl, seg_arr in find_label_runs(arr):
                segs, labs = sliding_window_majority(seg_arr, window_size, stride)
                if segs.size:
                    datasets[split_key]["data"].append(segs)
                    datasets[split_key]["label"].append(labs - 1)

    # concatenate, transpose & save
    for split in ["train", "val", "test"]:
        X = concat_data(datasets[split]["data"])  # (N,256,8)
        y = concat_label(datasets[split]["label"])
        X = X.transpose(0, 2, 1)  # (N,8,256)

        with h5py.File(os.path.join(save_root, f"{split}.h5"), "w") as f:
            f.create_dataset("data", data=X.astype(np.float32))
            f.create_dataset("label", data=y.astype(np.int32))
        uniq, cnt = np.unique(y, return_counts=True)
        print(
            f"{split.upper():5} → X={X.shape}, label dist:",
            dict(zip(uniq.tolist(), cnt.tolist())),
        )

    print("\nAll splits saved to:", save_root)
