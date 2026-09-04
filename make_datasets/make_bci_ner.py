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

"""Preprocess the BCI Challenge NER pre-training corpus into a pooled LMDB."""

import os

import mne
import pandas as pd

from make_datasets.common import (
    SAMPLING_FREQ,
    build_arg_parser,
    electrode_coordinate,
    list_files,
    slice_windows,
    write_pretraining_corpus,
)
import numpy as np

SLICE_SECONDS = 30
WINDOW_SAMPLES = SAMPLING_FREQ * SLICE_SECONDS
EXCLUDED_COLUMNS = ["EOG", "FeedBackEvent"]
REFERENCE = "AR"


def channel_coordinates(channel_names):
    """Build referential coordinates for the channels present in one session file."""
    reference_position = electrode_coordinate(REFERENCE)
    coords = np.zeros((len(channel_names), 2, 3), dtype=np.float32)
    for index, name in enumerate(channel_names):
        coords[index, 0, :] = electrode_coordinate(name.strip().upper())
        coords[index, 1, :] = reference_position
    return coords


def process_recording(task):
    """Filter and slice one comma-separated session recording."""
    root, relative_path = task
    frame = pd.read_csv(os.path.join(root, relative_path))
    if frame.shape[0] == 0:
        return []

    columns = [name for name in frame.columns[1:] if name not in EXCLUDED_COLUMNS]
    data = frame[columns].to_numpy().T

    info = mne.create_info(ch_names=columns, sfreq=SAMPLING_FREQ, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(l_freq=0.5, h_freq=30, method="fir", picks="eeg", verbose=False)

    coords = channel_coordinates(columns)
    stem = os.path.splitext(os.path.basename(relative_path))[0]
    return [
        (
            f"{stem}-{index}".encode(),
            {"eeg": window, "channel_coords": coords, "subject_id": stem},
        )
        for index, window in enumerate(slice_windows(raw.get_data(), WINDOW_SAMPLES))
    ]


def main():
    """Slice BCI-NER into 30-second pre-training windows."""
    args = build_arg_parser("BCI-NER pre-training corpus to LMDB").parse_args()
    mne.set_log_level("ERROR")

    tasks = [(args.input_dir, name) for name in list_files(args.input_dir, [".csv"])]
    write_pretraining_corpus(
        "BCI-NER", tasks, process_recording, args.output_dir, args.num_workers, args.dry_run
    )


if __name__ == "__main__":
    main()
