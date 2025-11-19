import os
import sys

import h5py
import numpy as np
import scipy.io
import scipy.signal as signal
from scipy.signal import iirnotch


# ==== Data augmentation functions ====
def random_amplitude_scale(sig, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return sig * scale


def random_time_jitter(sig, jitter_ratio=0.01):
    T, D = sig.shape
    std_ch = np.std(sig, axis=0)
    noise = np.random.randn(T, D) * (jitter_ratio * std_ch)
    return sig + noise


def random_channel_dropout(sig, dropout_prob=0.05):
    T, D = sig.shape
    mask = np.random.rand(D) < dropout_prob
    sig[:, mask] = 0.0
    return sig


def augment_one_sample(seg):
    out = seg.copy()
    out = random_amplitude_scale(out, (0.9, 1.1))
    out = random_time_jitter(out, 0.01)
    out = random_channel_dropout(out, 0.05)
    return out


def augment_train_data(data, labels, factor=3):
    if factor <= 0 or data.shape[0] == 0:
        return data, labels
    aug_segs = [data]
    aug_lbls = [labels]
    N = data.shape[0]
    for i in range(N):
        seg = data[i]  # [window_size, n_ch]
        lab = labels[i]
        for _ in range(factor):
            aug_segs.append(augment_one_sample(seg)[None, ...])
            aug_lbls.append([lab])
    new_data = np.concatenate(aug_segs, axis=0)
    new_labels = np.concatenate(aug_lbls, axis=0).ravel()
    return new_data, new_labels


# ==== Filter functions (operate at original fs=200) ====
def notch_filter(data, notch_freq=50.0, Q=30.0, fs=200.0):
    b, a = iirnotch(notch_freq, Q, fs)
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = signal.filtfilt(b, a, data[:, ch])
    return out


def bandpass_filter_emg(emg, lowcut=20.0, highcut=90.0, fs=200.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    out = np.zeros_like(emg)
    for c in range(emg.shape[1]):
        out[:, c] = signal.filtfilt(b, a, emg[:, c])
    return out


# ==== Window segmentation ====
def process_emg_features(emg, label, rerep, window_size=1024, stride=512):
    segs, lbls, reps = [], [], []
    N = len(label)
    for start in range(0, N, stride):
        end = start + window_size
        if end > N:
            cut = emg[start:N]
            pad = np.zeros((end - N, emg.shape[1]))
            win = np.vstack([cut, pad])
        else:
            win = emg[start:end]

        segs.append(win)
        lbls.append(label[start])
        reps.append(rerep[start])
    return np.array(segs), np.array(lbls), np.array(reps)


# ==== Main pipeline ====
def main():
    import argparse

    args = argparse.ArgumentParser(description="Process EMG data from DB5.")
    args.add_argument("--download_data", action="store_true")
    args.add_argument("--data_dir", type=str)
    args.add_argument("--save_dir", type=str)
    args.add_argument(
        "--window_size", type=int, help="Size of the sliding window for segmentation."
    )
    args.add_argument(
        "--stride", type=int, help="Stride for the sliding window segmentation."
    )
    args = args.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # download data if requested
    if args.download_data:
        # https://ninapro.hevs.ch/instructions/DB5.html
        len_data = range(1, 11)  # 1–10
        base_url = "https://ninapro.hevs.ch/files/DB5_Preproc/"
        # download and unzip
        for i in len_data:
            url = f"{base_url}s{i}.zip"
            os.system(f"wget -P {data_dir} {url}")
            os.system(f"unzip -o {data_dir}/s{i}.zip -d {data_dir}")
            os.system(f"rm {data_dir}/s{i}.zip")
            print(f"Downloaded and unzipped subject {i}\n{data_dir}/s{i}.zip")
        sys.exit("Data downloaded and unzipped. Rerun without --download_data.")

    fs = 200.0  # original sampling rate
    window_size, stride = args.window_size, args.stride
    train_reps = [1, 2, 5]
    val_reps = [4, 6]
    test_reps = [3]

    all_data = {"train": [], "val": [], "test": []}
    all_lbls = {"train": [], "val": [], "test": []}

    for subj in sorted(os.listdir(data_dir)):
        subj_path = os.path.join(data_dir, subj)
        if not os.path.isdir(subj_path):
            continue
        print(f"Processing subject {subj}...")
        for mat in sorted(os.listdir(subj_path)):
            if not mat.endswith(".mat"):
                continue
            dd = scipy.io.loadmat(os.path.join(subj_path, mat))
            emg = dd["emg"]  # [N,16]
            label = dd["restimulus"].ravel().astype(int)
            rerep = dd["rerepetition"].ravel().astype(int)

            # label shift by exercise
            if "E2" in mat:
                label = np.where(label != 0, label + 12, 0)
            elif "E3" in mat:
                label = np.where(label != 0, label + 29, 0)

            # filtering at original 200 Hz
            emg_filt = bandpass_filter_emg(emg, 20, 90, fs=fs)
            emg_filt = notch_filter(emg_filt, 50, 30, fs=fs)

            # z-score
            mu = emg_filt.mean(axis=0)
            sd = emg_filt.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            emg_z = (emg_filt - mu) / sd

            # segment
            segs, lbls, reps = process_emg_features(
                emg_z, label, rerep, window_size, stride
            )

            # split by repetition index
            for seg, lab, rp in zip(segs, lbls, reps):
                if rp in train_reps:
                    all_data["train"].append(seg)
                    all_lbls["train"].append(lab)
                elif rp in val_reps:
                    all_data["val"].append(seg)
                    all_lbls["val"].append(lab)
                elif rp in test_reps:
                    all_data["test"].append(seg)
                    all_lbls["test"].append(lab)

    # stack, augment train, transpose, save, and print stats
    stats = {}
    for split in ["train", "val", "test"]:
        X = np.stack(all_data[split], axis=0)  # [N, window_size, ch]
        y = np.array(all_lbls[split], dtype=int)

        if split == "train":
            X, y = augment_train_data(X, y, factor=2)

        # transpose to [N, ch, window_size]
        X = X.transpose(0, 2, 1)

        # save
        with h5py.File(os.path.join(save_dir, f"{split}.h5"), "w") as hf:
            hf.create_dataset("data", data=X)
            hf.create_dataset("label", data=y)

        # compute stats
        uniq, cnt = np.unique(y, return_counts=True)
        stats[split] = (X.shape, dict(zip(uniq.tolist(), cnt.tolist())))

    # print stats
    for split, (shape, dist) in stats.items():
        print(f"\n{split} → X={shape}, label distribution:")
        for lab, count in dist.items():
            print(f"  label {lab}: {count} samples")


if __name__ == "__main__":
    main()
