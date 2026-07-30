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

"""Preprocess the PhysioNet motor imagery corpus into train/val/test LMDBs."""

import os

import mne
import numpy as np

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    build_arg_parser,
    index_splits,
    referential_coordinates,
    run_jobs,
)

WINDOW_SECONDS = 4
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
RUNS = ["04", "06", "08", "10", "12", "14"]
TRAIN_SUBJECTS = 70
VAL_SUBJECTS = 89
TEST_SUBJECTS = 109

EDF_CHANNELS = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp5.", "Cp3.", "Cp1.", "Cpz.", "Cp2.", "Cp4.", "Cp6.",
    "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.", "T7..", "T8..", "T9..", "T10.",
    "Tp7.", "Tp8.",
    "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..",
    "Po7.", "Po3.", "Poz.", "Po4.", "Po8.",
    "O1..", "Oz..", "O2..", "Iz..",
]
CHANNELS = [name.strip(".").upper() for name in EDF_CHANNELS]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, "AR")


def event_to_label(event: int, run: str) -> int:
    """Map an annotation code to a class index, accounting for the run's task pairing."""
    return int(event - 2) if run in ("04", "08", "12") else int(event)


def process_run(task):
    """Epoch one motor imagery run and drop rest epochs."""
    split, root, subject, run = task
    path = os.path.join(root, subject, f"{subject}R{run}.edf")
    if not os.path.isfile(path):
        return []

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick(EDF_CHANNELS)
    raw.reorder_channels(EDF_CHANNELS)
    if raw.info["bads"]:
        raw.interpolate_bads()
    raw.set_eeg_reference(ref_channels="average", verbose=False)
    raw.filter(l_freq=0.3, h_freq=None, verbose=False)
    raw.notch_filter(60, verbose=False)
    raw.resample(SAMPLING_FREQ, verbose=False)

    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(
        raw, events, event_ids, tmin=0, tmax=WINDOW_SECONDS - 1.0 / SAMPLING_FREQ,
        baseline=None, preload=True, verbose=False,
    )
    data = epochs.get_data(units="uV")[:, :, -WINDOW_SAMPLES:]

    samples = []
    for index, (window, event) in enumerate(zip(data, epochs.events[:, 2])):
        if event == 1:
            continue
        samples.append((
            split,
            f"{subject}R{run}-{index}".encode(),
            {
                "eeg": window.astype(np.float32),
                "label": event_to_label(event, run),
                "channel_coords": CHANNEL_COORDS,
                "subject_id": subject,
            },
        ))
    return samples


def main():
    """Split PhysioNet-MI by subject and write one LMDB per split."""
    args = build_arg_parser("PhysioNet motor imagery to LMDB").parse_args()

    subjects = sorted(
        name for name in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, name))
    )[:TEST_SUBJECTS]
    splits = index_splits(subjects, TRAIN_SUBJECTS, VAL_SUBJECTS)

    tasks = [
        (split, args.input_dir, subject, run)
        for split, names in splits.items()
        for subject in names
        for run in RUNS
    ]

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_run, tasks, args.num_workers, "PhysioNet-MI runs"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("PhysioNet-MI")


if __name__ == "__main__":
    main()
