"""Preprocess the GWD pre-training corpus into a pooled LMDB."""

import os

import mne
import numpy as np
import scipy.io

from make_datasets.common import (
    SAMPLING_FREQ,
    build_arg_parser,
    electrode_coordinate,
    list_files,
    slice_windows,
    write_pretraining_corpus,
)

SLICE_SECONDS = 30
WINDOW_SAMPLES = SAMPLING_FREQ * SLICE_SECONDS
REFERENCE = "AR"


def known_channels(channel_names):
    """Return the indices and coordinates of channels with a known electrode position."""
    reference_position = electrode_coordinate(REFERENCE)
    indices, coords = [], []
    for index, name in enumerate(channel_names):
        position = electrode_coordinate(name.strip().upper())
        if position == (0.0, 0.0, 0.0):
            continue
        indices.append(index)
        coords.append([position, reference_position])
    return indices, np.asarray(coords, dtype=np.float32)


def process_recording(task):
    """Read one MATLAB recording, keep the locatable EEG channels, filter and slice."""
    root, relative_path = task
    contents = scipy.io.loadmat(os.path.join(root, relative_path))
    if "signal" not in contents or "header" not in contents:
        return []

    signal = np.asarray(contents["signal"], dtype=np.float32)
    header = contents["header"]

    try:
        source_freq = float(header["sample_rate"][0][0])
    except Exception:
        source_freq = float(SAMPLING_FREQ)

    try:
        eeg_indices = header["channels_eeg"][0][0].flatten() - 1
        raw_labels = header["channels_labels"][0][0].flatten()
        labels = [str(label[0]) if isinstance(label, np.ndarray) else str(label) for label in raw_labels]
        channel_names = [labels[index] for index in eeg_indices]
    except Exception:
        eeg_indices = np.arange(signal.shape[0])
        channel_names = [f"Ch{index + 1}" for index in eeg_indices]

    data = signal[eeg_indices, :]
    info = mne.create_info(ch_names=channel_names, sfreq=source_freq, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.filter(l_freq=0.5, h_freq=min(100.0, source_freq / 2 - 1), method="fir", picks="eeg", verbose=False)
    if raw.info["sfreq"] != SAMPLING_FREQ:
        raw.resample(SAMPLING_FREQ, verbose=False)

    indices, coords = known_channels(raw.ch_names)
    if not indices:
        return []

    stem = os.path.splitext(os.path.basename(relative_path))[0]
    return [
        (
            f"{stem}-{index}".encode(),
            {"eeg": window, "channel_coords": coords, "subject_id": stem},
        )
        for index, window in enumerate(slice_windows(raw.get_data()[indices], WINDOW_SAMPLES))
    ]


def main():
    """Slice GWD into 30-second pre-training windows."""
    args = build_arg_parser("GWD pre-training corpus to LMDB").parse_args()
    mne.set_log_level("ERROR")

    tasks = [(args.input_dir, name) for name in list_files(args.input_dir, [".mat"])]
    write_pretraining_corpus(
        "GWD", tasks, process_recording, args.output_dir, args.num_workers, args.dry_run
    )


if __name__ == "__main__":
    main()
