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

"""Preprocess the SEED-V emotion corpus into train/val/test LMDBs."""

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

SEGMENT_SECONDS = 4
SEGMENT_SAMPLES = SAMPLING_FREQ * SEGMENT_SECONDS
TRIALS_PER_SESSION = 15

CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, "AR")

TRIAL_BOUNDS = {
    "1": {"start": [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
          "end": [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},
    "2": {"start": [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
          "end": [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},
    "3": {"start": [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
          "end": [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}

TRIAL_LABELS = {
    "1": [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    "2": [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    "3": [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}

SUBJECT_SPLITS = {
    "train": {str(index) for index in range(1, 11)},
    "val": {str(index) for index in range(11, 14)},
    "test": {str(index) for index in range(14, 17)},
}

SUBJECT_PATTERN = re.compile(r"(\d+)")


def subject_of(filename: str) -> str:
    """Return the subject number encoded at the start of a SEED-V filename."""
    match = SUBJECT_PATTERN.search(os.path.basename(filename))
    return match.group(1) if match else os.path.splitext(os.path.basename(filename))[0]


def split_of(subject: str):
    """Return the split a subject belongs to, or None if it is not used."""
    for split, subjects in SUBJECT_SPLITS.items():
        if subject in subjects:
            return split
    return None


def process_recording(task):
    """Cut each labelled trial of one session into fixed-length segments."""
    root, filename = task
    subject = subject_of(filename)
    split = split_of(subject)
    if split is None:
        return []

    raw = mne.io.read_raw_cnt(os.path.join(root, filename), preload=True, verbose=False)
    raw.pick(CHANNELS)
    raw.reorder_channels(CHANNELS)
    raw.resample(SAMPLING_FREQ, verbose=False)
    raw.filter(l_freq=0.3, h_freq=75, verbose=False)
    data = raw.get_data(units="uV")

    session = os.path.basename(filename).split("_")[1]
    bounds = TRIAL_BOUNDS[session]
    labels = TRIAL_LABELS[session]

    samples = []
    for trial in range(TRIALS_PER_SESSION):
        trial_data = data[:, bounds["start"][trial] * SAMPLING_FREQ : bounds["end"][trial] * SAMPLING_FREQ]
        num_segments = trial_data.shape[1] // SEGMENT_SAMPLES
        if num_segments == 0:
            continue
        trimmed = trial_data[:, : num_segments * SEGMENT_SAMPLES]
        segments = trimmed.reshape(NUM_CHANNELS, num_segments, SEGMENT_SAMPLES).transpose(1, 0, 2)
        for index, segment in enumerate(segments):
            samples.append((
                split,
                f"{os.path.basename(filename)[:-4]}-{trial}-{index}".encode(),
                {
                    "eeg": segment.astype(np.float32),
                    "label": int(labels[trial]),
                    "channel_coords": CHANNEL_COORDS,
                    "subject_id": subject,
                },
            ))
    return samples


def main():
    """Split SEED-V by subject and write one LMDB per split."""
    args = build_arg_parser("SEED-V emotion recognition to LMDB").parse_args()
    mne.set_log_level("ERROR")

    recordings = sorted(name for name in os.listdir(args.input_dir) if name.lower().endswith(".cnt"))
    tasks = [(args.input_dir, name) for name in recordings]

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "SEED-V sessions"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("SEED-V")


if __name__ == "__main__":
    main()
