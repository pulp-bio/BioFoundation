#*----------------------------------------------------------------------------*
#* Copyright (C) 2026 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Glenn Anta Bucagu                                                 *
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

"""Preprocess the Helsinki neonatal seizure corpus into train/val/test LMDBs."""

import os
import re

import mne
import numpy as np
import pandas as pd

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    build_arg_parser,
    electrode_coordinate,
    run_jobs,
)

WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
DROPPED_CHANNELS = ["ECG EKG", "Resp Effort", "ECG EKG-REF", "Resp Effort-REF"]
ANNOTATION_FILES = ["annotations_2017_A.csv", "annotations_2017_B.csv", "annotations_2017_C.csv"]
TIE_BREAK_SEED = 0

BIPOLAR_MONTAGE = [
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fz", "Cz"), ("Cz", "Pz"),
]
NUM_CHANNELS = len(BIPOLAR_MONTAGE)

SUBJECT_SPLITS = {
    "train": {2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28,
              29, 30, 31, 33, 35, 36, 37, 38, 40, 41, 43, 44, 45, 48, 49, 50, 51, 52, 54, 55,
              56, 58, 60, 61, 62, 63, 64, 65, 66, 67, 70, 71, 72, 73, 74, 75, 76, 77, 78},
    "val": {1, 3, 24, 32, 34, 42, 46, 69},
    "test": {7, 11, 27, 39, 47, 53, 57, 59, 68, 79},
}

SUBJECT_PATTERN = re.compile(r"eeg(\d+)\.edf$", re.IGNORECASE)


def subject_of(filename: str):
    """Return the subject number encoded in a recording filename, or None."""
    match = SUBJECT_PATTERN.search(filename)
    return int(match.group(1)) if match else None


def split_of(subject: int):
    """Return the split a subject belongs to, or None if it is not used."""
    for split, subjects in SUBJECT_SPLITS.items():
        if subject in subjects:
            return split
    return None


def consensus_annotations(input_dir: str) -> pd.DataFrame:
    """Combine the three expert annotation files into a per-second consensus label.

    A second is labelled when at least two annotators agree. Two-annotator ties are
    broken with a seeded draw, and seconds without a majority are left as NaN so the
    window is skipped.
    """
    tables = [pd.read_csv(os.path.join(input_dir, name), header=None) for name in ANNOTATION_FILES]
    if len({table.shape for table in tables}) != 1:
        raise ValueError("Annotation files must have identical shapes")

    generator = np.random.default_rng(TIE_BREAK_SEED)

    def consensus(first, second, third):
        values = np.array([first, second, third], dtype=float)
        present = values[~np.isnan(values)]
        if present.size == 0:
            return np.nan
        if present.size == 1:
            return present[0]
        if present.size == 2:
            return present[0] if present[0] == present[1] else float(generator.integers(2))
        return float(round(present.mean()))

    combined = np.vectorize(consensus)(*[table.values for table in tables])
    frame = pd.DataFrame(combined)
    return frame.where(frame.isin([0.0, 1.0]), np.nan)


def bipolar_montage(raw):
    """Derive the bipolar montage from a referential recording.

    Returns the montage signal and its channel coordinates, keeping only the pairs
    whose electrodes are both present in the recording.
    """
    available = {}
    for name in raw.ch_names:
        upper = name.upper()
        if upper.startswith("EEG ") and "-REF" in upper:
            available[upper[4:].split("-")[0].strip()] = name

    data = raw.get_data()
    channel_names = list(raw.ch_names)
    signals, coords = [], []

    for active, reference in BIPOLAR_MONTAGE:
        active_channel = available.get(active.upper())
        reference_channel = available.get(reference.upper())
        if active_channel is None or reference_channel is None:
            continue
        signals.append(data[channel_names.index(active_channel)] - data[channel_names.index(reference_channel)])
        coords.append([electrode_coordinate(active.upper()), electrode_coordinate(reference.upper())])

    if not signals:
        return None, None
    return np.stack(signals), np.asarray(coords, dtype=np.float32)


def process_recording(task):
    """Filter one recording, build its bipolar montage and emit one window per second."""
    split, input_dir, filename, annotations = task
    subject = subject_of(filename)
    if subject is None:
        return []

    raw = mne.io.read_raw_edf(os.path.join(input_dir, filename), preload=True, verbose="ERROR")
    kept = [name for name in raw.ch_names if name not in DROPPED_CHANNELS]
    if not kept:
        return []
    raw.pick(kept)
    raw.filter(l_freq=0.5, h_freq=None, method="iir", phase="forward",
               iir_params=dict(order=6, ftype="butter"), verbose="ERROR")
    raw.notch_filter(freqs=50, notch_widths=4.0, method="iir", phase="forward", verbose="ERROR")
    if raw.info["sfreq"] != SAMPLING_FREQ:
        raw.resample(SAMPLING_FREQ, n_jobs=1, verbose="ERROR")

    montage, coords = bipolar_montage(raw)
    if montage is None or montage.shape[1] < WINDOW_SAMPLES:
        return []

    labels = annotations.iloc[:, subject - 1].values
    seconds = min(int(montage.shape[1] // SAMPLING_FREQ), len(labels))
    stem = os.path.splitext(filename)[0]

    samples = []
    for second in range(seconds):
        label = labels[second]
        if label not in (0, 1):
            continue
        start = second * WINDOW_SAMPLES
        window = montage[:, start : start + WINDOW_SAMPLES]
        if window.shape[1] != WINDOW_SAMPLES:
            continue
        samples.append((
            split,
            f"{stem}-{second}".encode(),
            {
                "eeg": window.astype(np.float32),
                "label": int(label),
                "channel_coords": coords,
                "subject_id": str(subject),
            },
        ))
    return samples


def main():
    """Split the neonatal corpus by subject and write one LMDB per split."""
    args = build_arg_parser("Neonatal seizure detection to LMDB").parse_args()
    annotations = consensus_annotations(args.input_dir)

    tasks = []
    for filename in sorted(name for name in os.listdir(args.input_dir) if name.lower().endswith(".edf")):
        subject = subject_of(filename)
        split = split_of(subject) if subject is not None else None
        if split is not None:
            tasks.append((split, args.input_dir, filename, annotations))

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "Neonate recordings"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("Neonate")


if __name__ == "__main__":
    main()
