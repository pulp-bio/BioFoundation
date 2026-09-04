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

"""Preprocess the TUEG pre-training corpus into a packed LMDB.

TUEG is the largest corpus, so each window is stored as a fixed-size byte blob rather
than a pickled dictionary, and the sample keys are written to a companion text file so
readers do not have to scan the database at start-up.

Subjects that also appear in TUAB are excluded, so that pre-training cannot see any
recording from a subject used in the TUAB fine-tuning splits.
"""

import os
from pathlib import Path

import mne
import numpy as np

from make_datasets.common import (
    SAMPLING_FREQ,
    PackedLMDBWriter,
    build_arg_parser,
    electrode_coordinate,
    run_jobs,
)

SLICE_SECONDS = 30
WINDOW_SAMPLES = SAMPLING_FREQ * SLICE_SECONDS
MAX_CHANNELS = 64

STANDARD_CHANNELS = [
    "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF", "EEG C4-REF",
    "EEG P3-REF", "EEG P4-REF", "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF",
    "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF", "EEG A1-REF", "EEG A2-REF",
    "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF", "EEG T1-REF", "EEG T2-REF",
]

ADDITIONAL_CHANNELS = [
    "EEG FP2-LE", "EEG P3-LE", "EEG C4P-REF", "EEG F8-LE", "EEG P4-LE", "EEG C3-LE",
    "EEG T6-LE", "EEG A1-LE", "EEG F4-LE", "EEG T5-LE", "EEG FZ-LE", "EEG O1-LE",
    "EEG PZ-LE", "EEG C4-LE", "EEG A2-LE", "EEG CZ-LE", "EEG T1-LE", "EEG F3-LE",
    "EEG T2-LE", "EEG FP1-LE", "EEG C3P-REF", "EEG O2-LE", "EEG OZ-LE", "EEG OZ-REF",
    "EEG T4-LE", "EEG F7-LE", "EEG SP1-LE", "EEG SP2-LE", "EEG T3-LE", "EEG SP1-REF",
    "EEG SP2-REF",
]

ALL_CHANNELS = list(dict.fromkeys(STANDARD_CHANNELS + ADDITIONAL_CHANNELS))
ELECTRODE_ALIASES = {"C4P": "CP4", "C3P": "CP3"}


def channel_coordinates(channel_name: str, session_reference: str):
    """Resolve the two electrode coordinates of one TUEG channel name.

    TUEG mixes average-reference and linked-ear recordings, and the suffix in the
    channel name only says which convention the file uses, so the session's reference
    is supplied separately.
    """
    stripped = channel_name[4:] if channel_name.startswith("EEG ") else channel_name
    active, reference = stripped.split("-")
    if reference.upper() == "REF":
        reference = "AR"
    elif reference.upper() == "LE":
        reference = session_reference

    def lookup(name: str):
        upper = name.upper()
        return electrode_coordinate(ELECTRODE_ALIASES.get(upper, upper))

    return lookup(active), lookup(reference)


def process_recording(task):
    """Filter, normalise and slice one recording, padding channels to the maximum.

    Each window is min-max normalised per channel to ``[-1, 1]`` before packing,
    because the packed format stores no per-sample statistics for a reader to apply.
    """
    path, tuab_subjects = task
    session = Path(path).stem
    subject = session.split("_s")[0]
    if subject in tuab_subjects:
        return []

    session_reference = "LE" if ("02_tcp_le" in str(path) or "04_tcp_le" in str(path)) else "AR"

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    if raw.n_times == 0:
        return []

    raw.drop_channels([name for name in raw.ch_names if name not in ALL_CHANNELS])
    if not raw.ch_names:
        return []
    raw.reorder_channels([name for name in ALL_CHANNELS if name in raw.ch_names])

    coords = np.asarray(
        [channel_coordinates(name, session_reference) for name in raw.ch_names], dtype=np.float32
    )

    raw.notch_filter(60, verbose=False)
    raw.filter(l_freq=0.3, h_freq=75.0, verbose=False)
    if raw.info["sfreq"] != SAMPLING_FREQ:
        raw.resample(SAMPLING_FREQ, n_jobs=1)

    data = raw.get_data().astype(np.float32) * 1e6
    num_channels, num_timesteps = data.shape
    if num_channels > MAX_CHANNELS or num_timesteps < WINDOW_SAMPLES:
        return []

    padded_coords = np.zeros((MAX_CHANNELS, 2, 3), dtype=np.float32)
    padded_coords[:num_channels] = coords

    windows = []
    for index in range(num_timesteps // WINDOW_SAMPLES):
        window = data[:, index * WINDOW_SAMPLES : (index + 1) * WINDOW_SAMPLES]
        minimum = window.min(axis=1, keepdims=True)
        maximum = window.max(axis=1, keepdims=True)
        normalised = ((window - minimum) / (maximum - minimum + 1e-10) - 0.5) * 2.0

        padded = np.zeros((MAX_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
        padded[:num_channels] = normalised
        windows.append((f"{session}_slice_{index:04d}", padded, padded_coords))

    return windows


def tuab_subject_ids(tuab_dir: str) -> set:
    """Return every subject identifier present in the raw TUAB corpus."""
    return {path.name.split("_s")[0] for path in Path(tuab_dir).rglob("*.edf")}


def main():
    """Slice TUEG into 30-second pre-training windows, excluding TUAB subjects."""
    parser = build_arg_parser("TUEG pre-training corpus to packed LMDB")
    parser.add_argument(
        "--tuab_dir", default=None,
        help="Root of the raw TUAB corpus. Subjects found here are excluded so they cannot "
             "leak into the TUAB fine-tuning splits.",
    )
    parser.add_argument(
        "--allow_tuab_overlap", action="store_true",
        help="Pre-train on all TUEG subjects, including those in TUAB. Only use this when "
             "TUAB is not among the downstream evaluation datasets.",
    )
    args = parser.parse_args()
    mne.set_log_level("ERROR")

    if args.allow_tuab_overlap:
        tuab_subjects = set()
        print("TUAB exclusion disabled; TUAB subjects may appear in pre-training")
    else:
        if args.tuab_dir is None:
            parser.error(
                "--tuab_dir is required so TUAB subjects can be held out of pre-training; "
                "pass --allow_tuab_overlap to opt out explicitly"
            )
        if not os.path.isdir(args.tuab_dir):
            parser.error(f"--tuab_dir does not exist: {args.tuab_dir}")
        tuab_subjects = tuab_subject_ids(args.tuab_dir)
        if not tuab_subjects:
            parser.error(
                f"No .edf files found under --tuab_dir {args.tuab_dir}; refusing to continue "
                "because this would silently leak TUAB into pre-training"
            )
        print(f"Excluding {len(tuab_subjects)} TUAB subjects from pre-training")

    recordings = sorted(str(path) for path in Path(args.input_dir).rglob("*.edf"))
    tasks = [(path, tuab_subjects) for path in recordings]

    writer = PackedLMDBWriter(
        lmdb_path=os.path.join(args.output_dir, "TUEG.lmdb"),
        keys_path=os.path.join(args.output_dir, "TUEG_keys.txt"),
        dry_run=args.dry_run,
    )
    for windows in run_jobs(process_recording, tasks, args.num_workers, "TUEG recordings"):
        for key, waveform, coords in windows:
            writer.put(key, waveform, coords)

    writer.close()
    writer.summarise("TUEG")


if __name__ == "__main__":
    main()
