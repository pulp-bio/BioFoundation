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

"""Preprocess the CHB-MIT seizure corpus into train/val/test LMDBs."""

import os
import pickle

import numpy as np
from scipy import signal

from make_datasets.common import (
    SAMPLING_FREQ,
    LMDBWriter,
    bipolar_coordinates,
    build_arg_parser,
    run_jobs,
)

SOURCE_FREQ = 256
WINDOW_SECONDS = 10
WINDOW_SAMPLES = SAMPLING_FREQ * WINDOW_SECONDS
SOURCE_WINDOW_SAMPLES = SOURCE_FREQ * WINDOW_SECONDS
SEIZURE_HOP_SECONDS = 5
SEIZURE_PAD_SECONDS = 1

CHANNEL_NAMES = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]
NUM_CHANNELS = len(CHANNEL_NAMES)
CHANNEL_COORDS = bipolar_coordinates([tuple(name.split("-")) for name in CHANNEL_NAMES])

VAL_PATIENTS = {"chb21", "chb22"}
TEST_PATIENTS = {"chb23", "chb24"}
PATIENT_ALIASES = {"chb21": "chb01"}


def patient_of(path: str) -> str:
    """Return the patient identifier for a recording, resolving known aliases."""
    patient = os.path.basename(os.path.dirname(path))
    return PATIENT_ALIASES.get(patient, patient)


def split_of(path: str) -> str:
    """Assign a recording to a split by patient, holding two patients out for each."""
    patient = os.path.basename(os.path.dirname(path))
    if patient in TEST_PATIENTS:
        return "test"
    if patient in VAL_PATIENTS:
        return "val"
    return "train"


def process_recording(task):
    """Window one recording, then re-window each seizure with overlap to enrich positives.

    Seizures are rare, so in addition to the non-overlapping pass every annotated
    seizure is re-sampled with a shorter hop, extending one second either side. A
    window is positive when a seizure boundary falls strictly inside it. Windows in
    the seizure pass are truncated at the end of the recording and resampled to the
    full window length regardless of how much signal they contain.
    """
    split, path = task
    with open(path, "rb") as handle:
        recording = pickle.load(handle)

    try:
        raw = np.stack([recording[name] for name in CHANNEL_NAMES], axis=0)
    except KeyError:
        return []

    seizures = recording.get("metadata", {}).get("times", [])
    record_id = os.path.basename(path).split(".")[0]
    patient = patient_of(path)
    length = raw.shape[1]
    samples = []

    def contains_boundary(window_start, window_end):
        return any(
            window_start < start < window_end or window_start < end < window_end
            for start, end in seizures
        )

    def emit(key, segment, label):
        resampled = signal.resample(segment, WINDOW_SAMPLES, axis=1)
        samples.append((
            split,
            key.encode(),
            {
                "eeg": resampled.astype(np.float32),
                "label": int(label),
                "channel_coords": CHANNEL_COORDS,
                "subject_id": patient,
            },
        ))

    for start in range(0, length, SOURCE_WINDOW_SAMPLES):
        end = start + SOURCE_WINDOW_SAMPLES
        if end > length:
            continue
        emit(f"{record_id}-{start}", raw[:, start:end], contains_boundary(start, end))

    hop = SEIZURE_HOP_SECONDS * SOURCE_FREQ
    pad = SEIZURE_PAD_SECONDS * SOURCE_FREQ
    for seizure_index, (seizure_start, seizure_end) in enumerate(seizures):
        first = max(0, seizure_start - pad)
        last = min(seizure_end + pad, length)
        for start in range(first, last, hop):
            end = min(start + SOURCE_WINDOW_SAMPLES, length)
            emit(f"{record_id}-s-{seizure_index}-add-{start}", raw[:, start:end], 1)

    return samples


def main():
    """Split CHB-MIT by patient and write one LMDB per split."""
    args = build_arg_parser("CHB-MIT seizure detection to LMDB").parse_args()

    recordings = []
    for directory, _, filenames in os.walk(args.input_dir):
        recordings += [
            os.path.join(directory, name) for name in filenames if name.lower().endswith(".pkl")
        ]
    tasks = [(split_of(path), path) for path in sorted(recordings)]

    writer = LMDBWriter(args.output_dir, dry_run=args.dry_run)
    for samples in run_jobs(process_recording, tasks, args.num_workers, "CHB-MIT recordings"):
        for split, key, sample in samples:
            writer.put(split, key, sample)

    writer.close()
    writer.summarise("CHB-MIT")


if __name__ == "__main__":
    main()
