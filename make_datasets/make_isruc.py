"""Preprocess the ISRUC sleep-staging corpus into train/val/test LMDBs.

Unlike the other fine-tuning datasets, one ISRUC sample is a sequence of consecutive
30-second epochs rather than a single window, because sleep stage depends on
neighbouring context. Each sample therefore holds ``(sequence_length, channels,
timesteps)`` with one label per epoch, and the sequence classification head consumes it.
"""

import os

import numpy as np
from mne.io import read_raw_edf

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    bipolar_coordinates,
    build_arg_parser,
    run_jobs,
)

EPOCH_SECONDS = 30
EPOCH_SAMPLES = SAMPLING_FREQ * EPOCH_SECONDS
SEQUENCE_LENGTH = 20
NUM_SUBJECTS = 100
TRAIN_MAX_SUBJECT = 80
VAL_MAX_SUBJECT = 90

MASTOID_CHANNELS = ["F3-M2", "C3-M2", "O1-M2", "F4-M1", "C4-M1", "O2-M1"]
AURICULAR_CHANNELS = ["F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]
NUM_CHANNELS = len(MASTOID_CHANNELS)
STAGE_TO_LABEL = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}

MASTOID_COORDS = bipolar_coordinates([tuple(name.split("-")) for name in MASTOID_CHANNELS])
AURICULAR_COORDS = bipolar_coordinates([tuple(name.split("-")) for name in AURICULAR_CHANNELS])


def split_of(subject: int) -> str:
    """Assign a subject to a split, holding out the highest-numbered subjects for test."""
    if subject <= TRAIN_MAX_SUBJECT:
        return "train"
    if subject <= VAL_MAX_SUBJECT:
        return "val"
    return "test"


def recording_pairs(input_dir: str):
    """Return the recording and hypnogram paths for every available subject."""
    pairs = []
    for subject in range(1, NUM_SUBJECTS + 1):
        folder = os.path.join(input_dir, str(subject))
        recording = os.path.join(folder, f"{subject}.rec")
        hypnogram = os.path.join(folder, f"{subject}_1.txt")
        if os.path.isfile(recording) and os.path.isfile(hypnogram):
            pairs.append((subject, recording, hypnogram))
    return pairs


def resolve_channels(recording: str):
    """Choose the montage present in a recording, deriving it from unipolar channels if needed.

    Returns the channel names to pick and their coordinates, or ``(None, None, None)`` when
    the recording has none of the expected montages.
    """
    present = set(read_raw_edf(recording, preload=False, verbose=False).info["ch_names"])
    if all(name in present for name in MASTOID_CHANNELS):
        return MASTOID_CHANNELS, MASTOID_COORDS, False
    if all(name in present for name in AURICULAR_CHANNELS):
        return AURICULAR_CHANNELS, AURICULAR_COORDS, False
    electrodes = {name for channel in AURICULAR_CHANNELS for name in channel.split("-")}
    if electrodes <= present:
        return sorted(electrodes), AURICULAR_COORDS, True
    return None, None, None


def process_recording(task):
    """Cut one night into sequences of consecutive labelled epochs."""
    split, subject, recording, hypnogram, channels, coords, derive_bipolar = task

    raw = read_raw_edf(recording, preload=True, verbose=False)
    raw.filter(0.3, 35, fir_design="firwin", verbose=False)
    raw.notch_filter(50, verbose=False)
    raw.pick(channels)
    raw.reorder_channels(channels)
    data = raw.to_data_frame().values[:, 1:].T

    if derive_bipolar:
        pairs = [name.split("-") for name in AURICULAR_CHANNELS]
        derived = np.zeros((NUM_CHANNELS, data.shape[1]), dtype=np.float32)
        for index, (active, reference) in enumerate(pairs):
            derived[index] = data[channels.index(active)] - data[channels.index(reference)]
        data = derived

    transposed = data.T
    remainder = transposed.shape[0] % EPOCH_SAMPLES
    if remainder:
        transposed = transposed[:-remainder]
    epochs = transposed.reshape(-1, EPOCH_SAMPLES, NUM_CHANNELS)

    with open(hypnogram) as handle:
        labels = np.array([STAGE_TO_LABEL[line.strip()] for line in handle if line.strip()])

    usable = min(epochs.shape[0], labels.shape[0])
    epochs, labels = epochs[:usable], labels[:usable]

    trailing = epochs.shape[0] % SEQUENCE_LENGTH
    if trailing:
        epochs, labels = epochs[:-trailing], labels[:-trailing]
    if epochs.shape[0] == 0:
        return []

    num_sequences = epochs.shape[0] // SEQUENCE_LENGTH
    epochs = epochs.reshape(num_sequences, SEQUENCE_LENGTH, EPOCH_SAMPLES, NUM_CHANNELS)
    epochs = epochs.transpose(0, 1, 3, 2)
    labels = labels.reshape(num_sequences, SEQUENCE_LENGTH)

    return [
        (
            split,
            f"{subject}-{index}".encode(),
            {
                "eeg": epochs[index].astype(np.float32),
                "label": labels[index].tolist(),
                "channel_coords": coords,
                "subject_id": str(subject),
            },
        )
        for index in range(num_sequences)
    ]


def main():
    """Split ISRUC by subject and write one LMDB per split."""
    args = build_arg_parser("ISRUC sleep staging to LMDB").parse_args()

    tasks, skipped = [], []
    for subject, recording, hypnogram in recording_pairs(args.input_dir):
        channels, coords, derive_bipolar = resolve_channels(recording)
        if channels is None:
            skipped.append(subject)
            continue
        tasks.append(
            (split_of(subject), subject, recording, hypnogram, channels, coords, derive_bipolar)
        )

    print(f"[ISRUC] usable subjects: {len(tasks)}, skipped for missing channels: {len(skipped)}")

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "ISRUC nights"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("ISRUC")


if __name__ == "__main__":
    main()
