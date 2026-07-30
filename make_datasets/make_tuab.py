"""Preprocess the TUAB abnormality-detection corpus into train/val/test LMDBs."""

import os

import mne
import numpy as np

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    bipolar_coordinates,
    build_arg_parser,
    run_jobs,
)

WINDOW_SECONDS = 10
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
CHANNEL_FOLDER = "01_tcp_ar"
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 4523

BIPOLAR_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("FP1", "F3"),
    ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("T4", "A2"), ("C4", "T4"), ("CZ", "C4"), ("FP2", "F4"),
    ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
NUM_CHANNELS = len(BIPOLAR_PAIRS)
CHANNEL_COORDS = bipolar_coordinates(BIPOLAR_PAIRS)

EDF_PAIRS = [(f"EEG {a}-REF", f"EEG {b}-REF") for a, b in BIPOLAR_PAIRS]
EDF_CHANNELS = list(dict.fromkeys([name for pair in EDF_PAIRS for name in pair]))


def subject_of(filename: str) -> str:
    """Return the TUAB subject identifier encoded in a filename."""
    return os.path.basename(filename).split("_")[0]


def recording_folder(root: str, split: str, label: int) -> str:
    """Return the folder holding recordings for one split and label."""
    return os.path.join(
        root, "eval" if split == "test" else "train", "abnormal" if label else "normal", CHANNEL_FOLDER
    )


def process_subject(task):
    """Build the bipolar montage for one subject and cut it into fixed-length windows."""
    split, root, subject, label = task
    folder = recording_folder(root, split, label)
    samples, skipped = [], []

    for filename in sorted(os.listdir(folder)):
        if not filename.startswith(f"{subject}_") or not filename.endswith(".edf"):
            continue
        path = os.path.join(folder, filename)
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            missing = [name for name in EDF_CHANNELS if name not in set(raw.info["ch_names"])]
            if missing:
                skipped.append((filename, f"missing_channels:{len(missing)}"))
                continue

            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.pick(EDF_CHANNELS)
            raw.reorder_channels(EDF_CHANNELS)
            raw.notch_filter(60, verbose=False)
            raw.filter(l_freq=0.3, h_freq=75, verbose=False)
            if raw.info["sfreq"] != SAMPLING_FREQ:
                raw.resample(SAMPLING_FREQ, n_jobs=1)

            data = raw.get_data(units="uV")
            bipolar = np.stack([
                data[EDF_CHANNELS.index(active)] - data[EDF_CHANNELS.index(reference)]
                for active, reference in EDF_PAIRS
            ]).astype(np.float32)

            num_windows = bipolar.shape[1] // WINDOW_SAMPLES
            if num_windows == 0:
                skipped.append((filename, "too_short"))
                continue

            for index in range(num_windows):
                window = bipolar[:, index * WINDOW_SAMPLES : (index + 1) * WINDOW_SAMPLES]
                samples.append((
                    split,
                    f"{filename[:-4]}-{index}".encode(),
                    {
                        "eeg": window,
                        "label": int(label),
                        "channel_coords": CHANNEL_COORDS,
                        "subject_id": subject,
                    },
                ))
        except Exception as error:
            skipped.append((filename, f"exception:{type(error).__name__}"))

    return samples, skipped


def subject_ids(folder: str) -> list:
    """Return the sorted unique subject identifiers in a recording folder."""
    return sorted({subject_of(name) for name in os.listdir(folder) if name.endswith(".edf")})


def main():
    """Split TUAB by subject and write one LMDB per split.

    The official evaluation set becomes the test split. Training subjects are divided
    into train and validation by subject, so no subject appears in two splits.
    """
    args = build_arg_parser("TUAB abnormality detection to LMDB").parse_args()
    generator = np.random.default_rng(SPLIT_SEED)

    tasks = []
    for label in (0, 1):
        development = subject_ids(recording_folder(args.input_dir, "train", label))
        generator.shuffle(development)
        cut = int((1.0 - VALIDATION_FRACTION) * len(development))
        tasks += [("train", args.input_dir, subject, label) for subject in development[:cut]]
        tasks += [("val", args.input_dir, subject, label) for subject in development[cut:]]
        tasks += [
            ("test", args.input_dir, subject, label)
            for subject in subject_ids(recording_folder(args.input_dir, "test", label))
        ]

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    skipped = []
    for samples, reasons in run_jobs(process_subject, tasks, args.num_workers, "TUAB subjects"):
        skipped += reasons
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("TUAB")
    if skipped:
        print(f"  skipped {len(skipped)} recordings, first 5: {skipped[:5]}")


if __name__ == "__main__":
    main()
