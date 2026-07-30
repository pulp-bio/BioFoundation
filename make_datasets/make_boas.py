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

"""Preprocess the BOAS headband pre-training corpus into a pooled LMDB."""

import os

import mne

from make_datasets.common import (
    SAMPLING_FREQ,
    build_arg_parser,
    list_files,
    referential_coordinates,
    slice_windows,
    write_pretraining_corpus,
)

SLICE_SECONDS = 30
WINDOW_SAMPLES = SAMPLING_FREQ * SLICE_SECONDS

CHANNEL_RENAMES = {"HB_1": "AF7", "HB_2": "AF8"}
CHANNELS = ["AF7", "AF8"]
NUM_CHANNELS = len(CHANNELS)
CHANNEL_COORDS = referential_coordinates(CHANNELS, "AR")


def process_recording(task):
    """Rename the headband channels, filter and slice one recording."""
    root, relative_path = task
    raw = mne.io.read_raw_edf(os.path.join(root, relative_path), preload=True, verbose=False)
    raw.rename_channels({key: value for key, value in CHANNEL_RENAMES.items() if key in raw.ch_names})

    present = [name for name in CHANNELS if name in raw.ch_names]
    if len(present) != NUM_CHANNELS:
        return []
    raw.pick(present)
    raw.reorder_channels(CHANNELS)
    raw.filter(l_freq=0.1, h_freq=10, method="fir", picks="eeg", verbose=False)
    if raw.info["sfreq"] != SAMPLING_FREQ:
        raw.resample(SAMPLING_FREQ, verbose=False)

    stem = os.path.splitext(os.path.basename(relative_path))[0]
    return [
        (
            f"{stem}-{index}".encode(),
            {"eeg": window, "channel_coords": CHANNEL_COORDS, "subject_id": stem},
        )
        for index, window in enumerate(slice_windows(raw.get_data(), WINDOW_SAMPLES))
    ]


def main():
    """Slice BOAS into 30-second pre-training windows."""
    args = build_arg_parser("BOAS pre-training corpus to LMDB").parse_args()
    mne.set_log_level("ERROR")

    tasks = [(args.input_dir, name) for name in list_files(args.input_dir, [".edf"])]
    write_pretraining_corpus(
        "BOAS", tasks, process_recording, args.output_dir, args.num_workers, args.dry_run
    )


if __name__ == "__main__":
    main()
