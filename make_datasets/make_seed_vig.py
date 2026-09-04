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

"""Preprocess the SEED-VIG vigilance corpus into train/val/test LMDBs."""

import os
import re

import numpy as np
import scipy.io

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    build_arg_parser,
    index_splits,
    referential_coordinates,
    run_jobs,
)

WINDOW_SECONDS = 8
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
TRAIN_RECORDINGS = 15
VAL_RECORDINGS = 19

CHANNELS = [
    "FT7", "FT8", "T7", "T8", "TP7", "TP8", "CP1", "CP2",
    "P1", "PZ", "P2", "PO3", "POZ", "PO4", "O1", "OZ", "O2",
]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, "AR")

SUBJECT_PATTERN = re.compile(r"(?:sub(?:ject)?[_\- ]?|S)(\d+)", re.IGNORECASE)


def subject_of(filename: str) -> str:
    """Return the subject number encoded in a recording filename."""
    match = SUBJECT_PATTERN.search(os.path.basename(filename))
    return match.group(1) if match else os.path.splitext(os.path.basename(filename))[0]


def process_recording(task):
    """Pair each 8-second window of one session with its PERCLOS score."""
    split, data_dir, labels_dir, filename = task
    eeg = scipy.io.loadmat(os.path.join(data_dir, filename))["EEG"][0][0][0]
    labels = scipy.io.loadmat(os.path.join(labels_dir, filename))["perclos"][:, 0]

    total_points, channels = eeg.shape
    if channels != NUM_CHANNELS:
        raise ValueError(f"{filename} has {channels} channels, expected {NUM_CHANNELS}")

    num_windows = total_points // WINDOW_SAMPLES
    if total_points != num_windows * WINDOW_SAMPLES:
        raise ValueError(f"{filename} length {total_points} is not a multiple of {WINDOW_SAMPLES}")
    if len(labels) != num_windows:
        raise ValueError(f"{filename} has {len(labels)} labels for {num_windows} windows")

    windows = eeg.reshape(num_windows, WINDOW_SAMPLES, NUM_CHANNELS).transpose(0, 2, 1)
    subject = subject_of(filename)
    stem = os.path.splitext(os.path.basename(filename))[0]

    return [
        (
            split,
            f"{stem}-{index}".encode(),
            {
                "eeg": window.astype(np.float32),
                "label": float(label),
                "channel_coords": CHANNEL_COORDS,
                "subject_id": subject,
            },
        )
        for index, (window, label) in enumerate(zip(windows, labels))
    ]


def main():
    """Split SEED-VIG by recording and write one LMDB per split."""
    parser = build_arg_parser("SEED-VIG vigilance regression to LMDB")
    parser.add_argument("--labels_dir", required=True, help="Directory holding the PERCLOS label files")
    args = parser.parse_args()

    recordings = sorted(name for name in os.listdir(args.input_dir) if name.lower().endswith(".mat"))
    splits = index_splits(recordings, TRAIN_RECORDINGS, VAL_RECORDINGS)
    tasks = [
        (split, args.input_dir, args.labels_dir, filename)
        for split, names in splits.items()
        for filename in names
    ]

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "SEED-VIG sessions"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("SEED-VIG")


if __name__ == "__main__":
    main()
