"""Preprocess the STEW workload corpus into train/val/test LMDBs."""

import os

import mne
import numpy as np

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    build_arg_parser,
    referential_coordinates,
    run_jobs,
)

SOURCE_FREQ = 128
WINDOW_SECONDS = 4
OVERLAP_SECONDS = 2
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
HOP_SAMPLES = SAMPLING_FREQ * (WINDOW_SECONDS - OVERLAP_SECONDS)
HIGHPASS_FREQ = 1.0
ASR_CUTOFF = 20
NUM_SUBJECTS = 48

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, "AR")

SUBJECT_SPLITS = {
    "train": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25,
              26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 43, 44],
    "val": [35, 38, 45, 46],
    "test": [22, 40, 41, 47, 48],
}


def load_ratings(path: str) -> dict:
    """Read the per-subject workload ratings for the low and high workload sessions."""
    ratings = {}
    with open(path, "r") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split(",")]
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            ratings[int(parts[0])] = {
                "lo": None if parts[1] == "" else int(parts[1]),
                "hi": None if parts[2] == "" else int(parts[2]),
            }
    return ratings


def rating_to_label(rating):
    """Map a 1-9 workload rating onto three balanced classes."""
    if rating is None:
        return None
    if 1 <= rating <= 3:
        return 0
    if 4 <= rating <= 6:
        return 1
    if 7 <= rating <= 9:
        return 2
    return None


def clean_recording(data: np.ndarray) -> np.ndarray:
    """High-pass, artifact-correct with ASR, re-reference and resample one recording."""
    import asrpy

    info = mne.create_info(ch_names=CHANNELS, sfreq=SOURCE_FREQ, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(l_freq=HIGHPASS_FREQ, h_freq=None, fir_design="firwin", verbose=False)

    asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=ASR_CUTOFF)
    asr.fit(raw)
    raw = asr.transform(raw)

    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    if int(round(raw.info["sfreq"])) != SAMPLING_FREQ:
        raw.resample(SAMPLING_FREQ, npad="auto", verbose=False)
    return raw.get_data().astype(np.float32)


def sliding_windows(data: np.ndarray) -> np.ndarray:
    """Cut a recording into overlapping fixed-length windows."""
    length = data.shape[1]
    if length < WINDOW_SAMPLES:
        return np.empty((0, data.shape[0], WINDOW_SAMPLES), dtype=np.float32)
    starts = np.arange(0, length - WINDOW_SAMPLES + 1, HOP_SAMPLES, dtype=int)
    return np.stack([data[:, start : start + WINDOW_SAMPLES] for start in starts]).astype(np.float32)


def process_recording(task):
    """Clean one workload session and cut it into labelled windows."""
    split, path, subject, session, label = task
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != NUM_CHANNELS:
        return []

    cleaned = clean_recording(data.T)
    samples = []
    for index, window in enumerate(sliding_windows(cleaned)):
        samples.append((
            split,
            f"sub{subject:02d}_{session}-{index}".encode(),
            {
                "eeg": window,
                "label": int(label),
                "channel_coords": CHANNEL_COORDS,
                "subject_id": str(subject),
            },
        ))
    return samples


def main():
    """Split STEW by subject and write one LMDB per split."""
    args = build_arg_parser("STEW workload classification to LMDB").parse_args()
    mne.set_log_level("ERROR")

    ratings = load_ratings(os.path.join(args.input_dir, "ratings.txt"))
    subject_split = {
        subject: split for split, subjects in SUBJECT_SPLITS.items() for subject in subjects
    }

    tasks = []
    for subject in range(1, NUM_SUBJECTS + 1):
        split = subject_split.get(subject)
        if split is None or subject not in ratings:
            continue
        for session in ("lo", "hi"):
            path = os.path.join(args.input_dir, f"sub{subject:02d}_{session}.txt")
            label = rating_to_label(ratings[subject][session])
            if label is not None and os.path.isfile(path):
                tasks.append((split, path, subject, session, label))

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "STEW recordings"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("STEW")


if __name__ == "__main__":
    main()
