import os
import sys

import h5py
import numpy as np
import scipy.io
import scipy.signal as signal
from joblib import Parallel, delayed
from scipy.signal import iirnotch
from tqdm import tqdm

_MATRIX_DOF2DOA_TRANSPOSED = np.array(
    # https://www.frontiersin.org/articles/10.3389/fnins.2019.00891/full
    # Open supplemental data > Data Sheet 1.PDF >
    # > SUPPLEMENTARY METHODS > Eqn. S2
    # https://www.frontiersin.org/articles/file/downloadfile/461612_supplementary-materials_datasheets_1_pdf/octet-stream/Data%20Sheet%201.PDF/1/461612
    [
        [+0.6390, +0.0000, +0.0000, +0.0000, +0.0000],
        [+0.3830, +0.0000, +0.0000, +0.0000, +0.0000],
        [+0.0000, +1.0000, +0.0000, +0.0000, +0.0000],
        [-0.6390, +0.0000, +0.0000, +0.0000, +0.0000],
        [+0.0000, +0.0000, +0.4000, +0.0000, +0.0000],
        [+0.0000, +0.0000, +0.6000, +0.0000, +0.0000],
        [+0.0000, +0.0000, +0.0000, +0.4000, +0.0000],
        [+0.0000, +0.0000, +0.0000, +0.6000, +0.0000],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.1667],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.3333],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.1667],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.3333],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000],
        [-0.1900, +0.0000, +0.0000, +0.0000, +0.0000],
        [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000],
    ],
    dtype=np.float32,
)

MATRIX_DOF2DOA = _MATRIX_DOF2DOA_TRANSPOSED.T


# ─────────────── Filtering ──────────────────
def notch_filter(data, notch_freq=50.0, Q=30.0, fs=1111.0):
    """Notch-filter every channel independently."""
    b, a = iirnotch(notch_freq, Q, fs)
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = signal.filtfilt(b, a, data[:, ch])
    return out


def bandpass_filter_emg(emg, lowcut=20.0, highcut=90.0, fs=1111.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    out = np.zeros_like(emg)
    for ch in range(emg.shape[1]):
        out[:, ch] = signal.filtfilt(b, a, emg[:, ch])
    return out


# ─────────────── Sliding window ──────────────
def sliding_window_segment(emg, label, window_size, stride):
    """
    Segment EMG with a sliding window.
    Use the frame at the window centre as the segment label / repetition index.
    """
    segments, labels = [], []
    n_samples = len(label)

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        emg_segment = emg[start:end]  # (win, ch)
        label_segment = label[start:end]  # (win, ch)
        segments.append(emg_segment)
        labels.append(label_segment)

    return np.array(segments), np.array(labels)


# ─────────────── Main pipeline ───────────────
def process_mat_file(mat_path, window_size, stride, fs):
    """
    Load one .mat file, filter out NaNs, filter & normalize EMG, map DoF→DoA,
    segment, and return (split, segs, labels).
    """
    mat = scipy.io.loadmat(mat_path)
    emg = mat["emg"]  # (T, 16)
    label = mat["glove"]  # (T, DoF)

    # 1) Drop timesteps with any NaNs in glove data
    valid = ~np.isnan(label).any(axis=1)
    emg = emg[valid]
    label = label[valid]

    # 2) Filtering
    emg = bandpass_filter_emg(emg, 20.0, 90.0, fs)
    emg = notch_filter(emg, 50.0, 30.0, fs)

    # 3) Z-score per channel
    mu = emg.mean(axis=0)
    sd = emg.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    emg = (emg - mu) / sd

    # 4) DoF → DoA
    y_doa = (MATRIX_DOF2DOA @ label.T).T

    # 5) Windowing
    segs, labs = sliding_window_segment(emg, y_doa, window_size, stride)

    # 6) Determine split
    fname = os.path.basename(mat_path)
    if "_A1" in fname:
        split = "train"
    elif "_A2" in fname:
        split = "val"
    elif "_A3" in fname:
        split = "test"
    else:
        return None  # skip

    return split, segs, labs


def main():
    import argparse

    args = argparse.ArgumentParser(description="Process EMG data from DB8.")
    args.add_argument("--download_data", action="store_true")
    args.add_argument("--data_dir", type=str, required=True)
    args.add_argument("--save_dir", type=str, required=True)
    args.add_argument(
        "--window_size", type=int, help="Size of the sliding window for segmentation."
    )
    args.add_argument(
        "--stride", type=int, help="Stride for the sliding window segmentation."
    )
    args.add_argument(
        "--n_jobs", type=int, default=-1, help="Number of parallel jobs to run."
    )
    args = args.parse_args()
    data_dir = args.data_dir  # input folder with .mat files
    os.makedirs(args.save_dir, exist_ok=True)

    # download data if requested
    if args.download_data:
        # https://ninapro.hevs.ch/instructions/DB8.html
        len_data = range(1, 13)  # 1–12
        base_url = "https://ninapro.hevs.ch/files/DB8/"
        # download and unzip
        for i in len_data:
            url_a = f"{base_url}S{i}_E1_A1.mat"
            url_b = f"{base_url}S{i}_E1_A2.mat"
            url_c = f"{base_url}S{i}_E1_A3.mat"
            os.system(f"wget -P {data_dir} {url_a}")
            os.system(f"wget -P {data_dir} {url_b}")
            os.system(f"wget -P {data_dir} {url_c}")
            print(
                f"Downloaded subject {i}\n{data_dir}/S{i}_E1_A1.mat and {data_dir}/S{i}_E1_A2.mat and {data_dir}/S{i}_E1_A3.mat"
            )
        sys.exit("Data downloaded and unzipped. Rerun without --download_data.")

    fs = 1111.0

    # collect all .mat paths
    mat_paths = [
        os.path.join(args.data_dir, f)
        for f in sorted(os.listdir(args.data_dir))
        if f.endswith(".mat")
    ]

    # run in parallel
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_mat_file)(mp, args.window_size, args.stride, fs)
        for mp in mat_paths
    )

    # aggregate
    splits = {k: {"data": [], "label": []} for k in ("train", "val", "test")}
    for out in tqdm(results, desc="Processing files", unit="file"):
        if out is None:
            continue
        split, segs, labs = out
        splits[split]["data"].append(segs)
        splits[split]["label"].append(labs)

    # concatenate + save + stats
    for split, d in tqdm(splits.items(), desc="Saving splits", unit="split"):
        if not d["data"]:
            continue

        X = np.concatenate(d["data"], axis=0)
        y = np.concatenate(d["label"], axis=0)

        # transpose to [N, ch, window_size]
        X = X.transpose(0, 2, 1)

        print(f"Split: {split}, X shape: {X.shape}, y shape: {y.shape}")
        # save
        with h5py.File(os.path.join(args.save_dir, f"{split}.h5"), "w") as hf:
            hf.create_dataset("data", data=X)
            hf.create_dataset("label", data=y)


if __name__ == "__main__":
    main()
