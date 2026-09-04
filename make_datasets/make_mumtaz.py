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
#* Author:  BioFoundation Contributors                                        *
#*                                                                            *
#* Imported from the S-CEReBrO reference implementation (TimeFM).             *
#*----------------------------------------------------------------------------*

"""Preprocess the Mumtaz depression corpus into train/val/test LMDBs."""

import os
import re

import mne
import numpy as np

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    build_arg_parser,
    referential_coordinates,
    run_jobs,
)

WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
REFERENCE = "LE"

EDF_CHANNELS = [
    "EEG Fp1-LE", "EEG Fp2-LE", "EEG F3-LE", "EEG F4-LE", "EEG C3-LE", "EEG C4-LE",
    "EEG P3-LE", "EEG P4-LE", "EEG O1-LE", "EEG O2-LE", "EEG F7-LE", "EEG F8-LE",
    "EEG T3-LE", "EEG T4-LE", "EEG T5-LE", "EEG T6-LE", "EEG Fz-LE", "EEG Cz-LE",
    "EEG Pz-LE",
]
CHANNELS = [name.replace("EEG ", "").replace("-LE", "").upper() for name in EDF_CHANNELS]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, REFERENCE)

HEALTHY_TRAIN, HEALTHY_VAL = 40, 48
DEPRESSED_TRAIN, DEPRESSED_VAL = 42, 52

SUBJECT_PATTERN = re.compile(r"\b(H|MDD)\s*S?(\d+)", re.IGNORECASE)


def subject_of(filename: str):
    """Return the group-qualified subject identifier, or None if it cannot be parsed."""
    match = SUBJECT_PATTERN.search(os.path.splitext(os.path.basename(filename))[0].replace("  ", " "))
    if not match:
        return None
    return f"{match.group(1).upper()}_S{match.group(2)}"


def label_of(subject: str) -> int:
    """Return 1 for depressed subjects and 0 for healthy controls."""
    return 1 if subject.startswith("MDD_") else 0


def process_recording(task):
    """Filter one resting-state recording and cut it into fixed-length windows."""
    split, root, filename = task
    if "TASK" in filename.upper():
        return []
    subject = subject_of(filename)
    if subject is None:
        return []

    raw = mne.io.read_raw_edf(os.path.join(root, filename), preload=True, verbose=False)
    present = [name for name in EDF_CHANNELS if name in raw.ch_names]
    if len(present) != NUM_CHANNELS:
        return []
    raw.pick(present)
    raw.reorder_channels(present)
    raw.resample(SAMPLING_FREQ, verbose=False)
    raw.filter(l_freq=0.3, h_freq=75, verbose=False)
    raw.notch_filter(50, verbose=False)

    data = raw.get_data(units="uV")
    num_windows = data.shape[1] // WINDOW_SAMPLES
    if num_windows == 0:
        return []

    windows = data[:, : num_windows * WINDOW_SAMPLES]
    windows = windows.reshape(NUM_CHANNELS, num_windows, WINDOW_SAMPLES).transpose(1, 0, 2)

    label = label_of(subject)
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
    """Split Mumtaz by recording within each diagnostic group and write one LMDB per split."""
    args = build_arg_parser("Mumtaz depression detection to LMDB").parse_args()

    healthy, depressed = [], []
    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.lower().endswith(".edf") or "TASK" in filename.upper():
            continue
        subject = subject_of(filename)
        if subject is None:
            continue
        (depressed if label_of(subject) else healthy).append(filename)

    splits = {
        "train": healthy[:HEALTHY_TRAIN] + depressed[:DEPRESSED_TRAIN],
        "val": healthy[HEALTHY_TRAIN:HEALTHY_VAL] + depressed[DEPRESSED_TRAIN:DEPRESSED_VAL],
        "test": healthy[HEALTHY_VAL:] + depressed[DEPRESSED_VAL:],
    }
    tasks = [
        (split, args.input_dir, filename) for split, names in splits.items() for filename in names
    ]

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "Mumtaz recordings"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("Mumtaz")


if __name__ == "__main__":
    main()
