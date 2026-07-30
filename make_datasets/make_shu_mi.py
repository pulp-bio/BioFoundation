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

"""Preprocess the SHU-MI motor imagery corpus into train/val/test LMDBs."""

import os

import numpy as np
import scipy.io
from scipy import signal

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    build_arg_parser,
    index_splits,
    list_files,
    referential_coordinates,
    run_jobs,
)

WINDOW_SECONDS = 4
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
TRAIN_RECORDINGS = 75
VAL_RECORDINGS = 100

CHANNELS = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "CZ", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2",
]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, "AR")


def process_recording(task):
    """Resample one session to the target rate and emit one sample per trial."""
    split, root, relative_path = task
    data = scipy.io.loadmat(os.path.join(root, relative_path))
    trials = signal.resample(data["data"], WINDOW_SAMPLES, axis=2)
    labels = data["labels"][0]
    stem = os.path.splitext(os.path.basename(relative_path))[0]

    samples = []
    for index in range(trials.shape[0]):
        samples.append((
            split,
            f"{stem}-{index}".encode(),
            {
                "eeg": trials[index].astype(np.float32),
                "label": int(labels[index] - 1),
                "channel_coords": CHANNEL_COORDS,
                "subject_id": stem,
            },
        ))
    return samples


def main():
    """Split SHU-MI by recording and write one LMDB per split."""
    args = build_arg_parser("SHU-MI motor imagery to LMDB").parse_args()

    recordings = list_files(args.input_dir, [".mat"])
    splits = index_splits(recordings, TRAIN_RECORDINGS, VAL_RECORDINGS)
    tasks = [
        (split, args.input_dir, relative_path)
        for split, paths in splits.items()
        for relative_path in paths
    ]

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "SHU-MI recordings"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("SHU-MI")


if __name__ == "__main__":
    main()
