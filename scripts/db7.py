import os
import sys

import h5py
import numpy as np
import scipy.io
import scipy.signal as signal
from scipy.signal import iirnotch


# ─────────────── Filtering ──────────────────
def notch_filter(data, notch_freq=50.0, Q=30.0, fs=2000.0):
    """Notch-filter every channel independently."""
    b, a = iirnotch(notch_freq, Q, fs)
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = signal.filtfilt(b, a, data[:, ch])
    return out


def bandpass_filter_emg(emg, lowcut=20.0, highcut=90.0, fs=2000.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    out = np.zeros_like(emg)
    for ch in range(emg.shape[1]):
        out[:, ch] = signal.filtfilt(b, a, emg[:, ch])
    return out


# ─────────────── Sliding window ──────────────
def sliding_window_segment(emg, label, rerepetition, window_size, stride):
    """
    Segment EMG with a sliding window.
    Use the frame at the window centre as the segment label / repetition index.
    """
    segments, labels, reps = [], [], []
    n_samples = len(label)

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        emg_segment = emg[start:end]  # (win, ch)
        centre_idx = (start + end) // 2
        segments.append(emg_segment)
        labels.append(label[centre_idx])
        reps.append(rerepetition[centre_idx])

    return np.array(segments), np.array(labels), np.array(reps)


# ─────────────── Main pipeline ───────────────
def main():
    import argparse

    args = argparse.ArgumentParser(description="Process EMG data from DB7.")
    args.add_argument("--download_data", action="store_true")
    args.add_argument("--data_dir", type=str)
    args.add_argument("--save_dir", type=str)
    args.add_argument(
        "--window_size",
        type=int,
        default=256,
        help="Size of the sliding window for segmentation.",
    )
    args.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride for the sliding window segmentation.",
    )
    args = args.parse_args()
    data_dir = args.data_dir  # input folder with .mat files
    save_dir = args.save_dir  # output folder for .h5 files
    os.makedirs(save_dir, exist_ok=True)

    # download data if requested
    if args.download_data:
        # https://ninapro.hevs.ch/instructions/DB7.html
        len_data = range(1, 23)  # 1–22
        base_url = "https://ninapro.hevs.ch/files/DB7_Preproc/"
        # download and unzip
        for i in len_data:
            url = f"{base_url}Subject_{i}.zip"
            os.system(f"wget -P {data_dir} {url}")
            os.system(f"unzip -o {data_dir}/Subject_{i}.zip -d {data_dir}/Subject_{i}")
            os.system(f"rm {data_dir}/Subject_{i}.zip")
            print(f"Downloaded and unzipped subject {i}\n{data_dir}/Subject_{i}.zip")
        sys.exit("Data downloaded and unzipped. Rerun without --download_data.")

    fs = 2000.0
    window_size, stride = args.window_size, args.stride

    train_reps = [1, 2, 3, 4]  # 1–4
    val_reps = [5]  # 5
    test_reps = [6]  # 6

    splits = {
        "train": {"data": [], "label": []},
        "val": {"data": [], "label": []},
        "test": {"data": [], "label": []},
    }

    # iterate subjects
    for subj in sorted(os.listdir(data_dir)):
        subj_path = os.path.join(data_dir, subj)
        if not os.path.isdir(subj_path):
            continue
        print(f"Processing subject {subj} ...")

        subj_seg, subj_lbl, subj_rep = [], [], []

        # iterate .mat files
        for mat_file in sorted(os.listdir(subj_path)):
            if not mat_file.endswith(".mat"):
                continue
            mat_path = os.path.join(subj_path, mat_file)
            mat = scipy.io.loadmat(mat_path)

            emg = mat["emg"]  # (N, 16)
            label = mat["restimulus"].ravel()
            rerep = mat["rerepetition"].ravel()

            # filtering
            emg = bandpass_filter_emg(emg, 20.0, 450.0, fs=fs)
            emg = notch_filter(emg, 50.0, 30.0, fs=fs)

            # z-score per channel
            mu = emg.mean(axis=0)
            sd = emg.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            emg = (emg - mu) / sd

            # windowing
            seg, lbl, rep = sliding_window_segment(
                emg, label, rerep, window_size, stride
            )
            subj_seg.append(seg)
            subj_lbl.append(lbl)
            subj_rep.append(rep)

        if not subj_seg:
            continue

        seg = np.concatenate(subj_seg, axis=0)  # (M, win, 14)
        lbl = np.concatenate(subj_lbl)
        rep = np.concatenate(subj_rep)

        # split by repetition id
        for split_name, mask in (
            ("train", np.isin(rep, train_reps)),
            ("val", np.isin(rep, val_reps)),
            ("test", np.isin(rep, test_reps)),
        ):
            X = seg[mask].transpose(0, 2, 1)  # (N, 14, 1024)
            y = lbl[mask]
            splits[split_name]["data"].append(X)
            splits[split_name]["label"].append(y)

    # concatenate, save, and report
    for split in ["train", "val", "test"]:
        X = (
            np.concatenate(splits[split]["data"], axis=0)
            if splits[split]["data"]
            else np.empty((0, 14, window_size))
        )
        y = (
            np.concatenate(splits[split]["label"], axis=0)
            if splits[split]["label"]
            else np.empty((0,), dtype=int)
        )

        with h5py.File(os.path.join(save_dir, f"{split}.h5"), "w") as f:
            f.create_dataset("data", data=X.astype(np.float32))
            f.create_dataset("label", data=y.astype(np.int64))

        uniq, cnt = np.unique(y, return_counts=True)
        print(f"\n{split.upper()} → X={X.shape}, label distribution:")
        for u, c in zip(uniq, cnt):
            print(f"  label {u}: {c} samples")

    print("\nSaved: train.h5, val.h5, test.h5")


if __name__ == "__main__":
    main()
