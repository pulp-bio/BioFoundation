"""Preprocess the mental arithmetic corpus into train/val/test LMDBs."""

import os
import re

import mne
import numpy as np

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    build_arg_parser,
    electrode_coordinate,
    index_splits,
    run_jobs,
)

WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
TRAIN_RECORDINGS = 56
VAL_RECORDINGS = 64

EDF_CHANNELS = [
    "EEG Fp1", "EEG Fp2", "EEG F3", "EEG F4", "EEG F7", "EEG F8",
    "EEG T3", "EEG T4", "EEG C3", "EEG C4", "EEG T5", "EEG T6",
    "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG Fz", "EEG Cz",
    "EEG Pz", "EEG A2-A1",
]
NUM_CHANNELS = len(EDF_CHANNELS)
REFERENCE = "A1"


def channel_coordinates() -> np.ndarray:
    """Build coordinates for the montage, treating the A2-A1 channel as bipolar."""
    coords = np.zeros((NUM_CHANNELS, 2, 3), dtype=np.float32)
    reference_position = electrode_coordinate(REFERENCE)
    for index, name in enumerate(EDF_CHANNELS):
        label = name.replace("EEG ", "").upper()
        if label == "A2-A1":
            coords[index, 0, :] = electrode_coordinate("A2")
            coords[index, 1, :] = electrode_coordinate("A1")
        else:
            coords[index, 0, :] = electrode_coordinate(label)
            coords[index, 1, :] = reference_position
    return coords


CHANNEL_COORDS = channel_coordinates()

SUBJECT_PATTERN = re.compile(r"Subject(\d+)", re.IGNORECASE)
LABEL_PATTERN = re.compile(r"_(\d)\.edf$", re.IGNORECASE)


def subject_of(filename: str) -> str:
    """Return the subject number encoded in a recording filename."""
    match = SUBJECT_PATTERN.search(filename)
    return (match.group(1).lstrip("0") or "0") if match else os.path.splitext(filename)[0]


def label_of(filename: str) -> int:
    """Return the task condition encoded in the filename suffix."""
    match = LABEL_PATTERN.search(os.path.basename(filename))
    if not match:
        raise ValueError(f"Cannot parse a condition label from {filename}")
    return int(match.group(1)) - 1


def process_recording(task):
    """Resample one recording and cut it into fixed-length windows."""
    split, root, filename = task
    raw = mne.io.read_raw_edf(os.path.join(root, filename), preload=True, verbose=False)
    raw.pick(EDF_CHANNELS)
    raw.reorder_channels(EDF_CHANNELS)
    raw.resample(SAMPLING_FREQ, verbose=False)

    data = raw.get_data(units="uV")
    if data.shape[0] != NUM_CHANNELS:
        raise RuntimeError(f"{filename} has {data.shape[0]} channels, expected {NUM_CHANNELS}")

    num_windows = data.shape[1] // WINDOW_SAMPLES
    if num_windows == 0:
        return []
    windows = data[:, : num_windows * WINDOW_SAMPLES]
    windows = windows.reshape(NUM_CHANNELS, num_windows, WINDOW_SAMPLES).transpose(1, 0, 2)

    subject = subject_of(filename)
    label = label_of(filename)
    stem = os.path.splitext(filename)[0]
    return [
        (
            split,
            f"{stem}-{index}".encode(),
            {
                "eeg": window.astype(np.float32),
                "label": label,
                "channel_coords": CHANNEL_COORDS,
                "subject_id": subject,
            },
        )
        for index, window in enumerate(windows)
    ]


def main():
    """Split the mental arithmetic corpus by recording and write one LMDB per split."""
    args = build_arg_parser("Mental arithmetic classification to LMDB").parse_args()

    recordings = sorted(name for name in os.listdir(args.input_dir) if name.lower().endswith(".edf"))
    splits = index_splits(recordings, TRAIN_RECORDINGS, VAL_RECORDINGS)
    tasks = [
        (split, args.input_dir, filename) for split, names in splits.items() for filename in names
    ]

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "Mental arithmetic recordings"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("MentalArithmetic")


if __name__ == "__main__":
    main()
