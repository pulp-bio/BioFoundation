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

"""Preprocess the SEED and SEED-IV pre-training corpora into a pooled LMDB."""

import os

import mne
import numpy as np

from make_datasets.common import (
    SAMPLING_FREQ,
    build_arg_parser,
    referential_coordinates,
    slice_windows,
    write_pretraining_corpus,
)

SLICE_SECONDS = 30
WINDOW_SAMPLES = SAMPLING_FREQ * SLICE_SECONDS

CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, "AR")


def process_recording(task):
    """Slice every 62-channel array stored in one MATLAB session file."""
    root, relative_path = task
    import scipy.io

    contents = scipy.io.loadmat(os.path.join(root, relative_path))
    stem = os.path.splitext(os.path.basename(relative_path))[0]
    samples = []

    for name, value in contents.items():
        if name.startswith("__"):
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != NUM_CHANNELS:
            continue

        info = mne.create_info(CHANNELS, sfreq=SAMPLING_FREQ, ch_types="eeg", verbose=False)
        raw = mne.io.RawArray(array, info, verbose=False)
        for index, window in enumerate(slice_windows(raw.get_data(), WINDOW_SAMPLES)):
            samples.append((
                f"{stem}-{name}-{index}".encode(),
                {"eeg": window, "channel_coords": CHANNEL_COORDS, "subject_id": stem},
            ))
    return samples


def main():
    """Slice SEED or SEED-IV into 30-second pre-training windows."""
    args = build_arg_parser("SEED / SEED-IV pre-training corpus to LMDB").parse_args()
    mne.set_log_level("ERROR")

    from make_datasets.common import list_files

    tasks = [(args.input_dir, name) for name in list_files(args.input_dir, [".mat"])]
    write_pretraining_corpus(
        "SEED", tasks, process_recording, args.output_dir, args.num_workers, args.dry_run
    )


if __name__ == "__main__":
    main()
