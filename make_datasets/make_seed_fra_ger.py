"""Preprocess the SEED-FRA and SEED-GER pre-training corpora into a pooled LMDB."""

import os

import mne
import numpy as np

from make_datasets.common import (
    SAMPLING_FREQ,
    build_arg_parser,
    referential_coordinates,
    slice_windows,
    write_pretraining_corpus,
)

SLICE_SECONDS = 30
WINDOW_SAMPLES = SAMPLING_FREQ * SLICE_SECONDS

CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, "AR")


def process_recording(task):
    """Filter and slice one Neuroscan session recording."""
    root, relative_path = task
    path = os.path.join(root, relative_path)
    raw = mne.io.read_raw_cnt(path, preload=True, verbose=False)
    if raw.n_times == 0:
        return []

    present = [name for name in CHANNELS if name in raw.ch_names]
    if len(present) != NUM_CHANNELS:
        return []
    raw.pick(present)
    raw.reorder_channels(CHANNELS)
    raw.notch_filter(50, verbose=False)
    raw.filter(l_freq=0.5, h_freq=30, method="fir", picks="eeg", verbose=False)
    if raw.info["sfreq"] != SAMPLING_FREQ:
        raw.resample(SAMPLING_FREQ, verbose=False)

    stem = os.path.splitext(os.path.basename(relative_path))[0]
    return [
        (
            f"{stem}-{index}".encode(),
            {"eeg": window, "channel_coords": CHANNEL_COORDS, "subject_id": stem},
        )
        for index, window in enumerate(slice_windows(raw.get_data(), WINDOW_SAMPLES))
    ]


def main():
    """Slice SEED-FRA or SEED-GER into 30-second pre-training windows."""
    args = build_arg_parser("SEED-FRA / SEED-GER pre-training corpus to LMDB").parse_args()
    mne.set_log_level("ERROR")

    from make_datasets.common import list_files

    tasks = [(args.input_dir, name) for name in list_files(args.input_dir, [".cnt"])]
    write_pretraining_corpus(
        "SEED-FRA", tasks, process_recording, args.output_dir, args.num_workers, args.dry_run
    )


if __name__ == "__main__":
    main()
