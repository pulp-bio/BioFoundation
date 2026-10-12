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

"""Convert raw CHB-MIT recordings into the per-recording pickles make_chb_mit.py reads.

This is the first of the two CHB-MIT steps. It reproduces the BIOT/CBraMod
``process1.py``, including its fixed table of patients and recordings, which leaves
out chb12, chb13 and chb17. Each recording becomes ``chbXX/chbXX_NN.edf.pkl`` under the
output directory: a dictionary from channel name to signal in the EDF's physical units,
plus ``metadata`` holding the seizure count and ``times``, each seizure's start and end
as sample indices at 256 Hz.

Reference: https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/CHB-MIT/process1.py
"""

import os
import pickle

import numpy as np

from make_datasets.common import build_arg_parser, run_jobs

SOURCE_FREQ = 256

# (patient, reference recording, first other recording, last other recording, summary
# block). The channels listed for the reference recording in the given "Channels
# changed" block of the patient's summary are the channels kept for all its recordings.
RECORDINGS = [
    ("01", "01", 2, 46, 0), ("02", "01", 2, 35, 0), ("03", "01", 2, 38, 0),
    ("05", "01", 2, 39, 0), ("06", "01", 2, 24, 0), ("07", "01", 2, 19, 0),
    ("08", "02", 3, 29, 0), ("10", "01", 2, 89, 0), ("11", "01", 2, 99, 0),
    ("14", "01", 2, 42, 0), ("20", "01", 2, 68, 0), ("21", "01", 2, 33, 0),
    ("22", "01", 2, 77, 0), ("23", "06", 7, 20, 0), ("24", "01", 3, 21, 0),
    ("04", "07", 1, 43, 1), ("09", "02", 1, 19, 1), ("15", "02", 1, 63, 1),
    ("16", "01", 2, 19, 0), ("18", "02", 1, 36, 1), ("19", "02", 1, 30, 1),
]


def read_edf_channels(path, indices=None):
    """Read physical signals, suffixing a repeated channel label with ``-2``."""
    from pyedflib import highlevel  # only this preprocessing step needs pyedflib

    signals, headers, _ = highlevel.read_edf(path, ch_nrs=indices, digital=False)
    channels = {}
    for signal, header in zip(signals, headers):
        label = header.get("label")
        if label in channels:
            label = label + "-2"
        channels[label] = signal
    return channels


def summary_channels(summary_path, block):
    """Return the channel names and EDF indices listed in one block of a summary file."""
    names, indices = [], []
    current_block = 0
    with open(summary_path) as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "Channels" and fields[1] == "changed:":
                current_block += 1
            # "-" and "." mark an empty channel slot.
            if fields[0] == "Channel" and current_block == block and fields[2] not in ("-", "."):
                names.append(fields[2] + "-2" if fields[2] in names else fields[2])
                indices.append(int(fields[1][:-1]) - 1)
    return names, indices


def seizure_metadata(summary_path, filename):
    """Read one recording's seizure count and seizure sample ranges from the summary.

    Times are converted to samples as ``seconds * 256 - 1``, as in the reference.
    """
    with open(summary_path) as handle:
        lines = handle.readlines()

    metadata = {}
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 3 or fields[2] != filename:
            continue

        seizures = 0
        for following in lines[index + 1 :]:
            parts = following.split()
            if parts and parts[0] == "Number":
                seizures = int(parts[-1])
                break

        times = []
        cursor = index + 1
        for _ in range(seizures):
            while cursor < len(lines):
                parts = lines[cursor].split()
                cursor += 1
                if parts and parts[0] == "Seizure" and "Start" in parts:
                    start = int(parts[-2]) * SOURCE_FREQ - 1
                    end = int(lines[cursor].split()[-2]) * SOURCE_FREQ - 1
                    times.append((start, end))
                    break

        metadata["seizures"] = seizures
        metadata["times"] = times
    return metadata


def convert_patient(task):
    """Write the pickles for one patient and return how many recordings were written."""
    patient, reference, first, last, block, input_dir, output_dir, dry_run = task
    folder = f"chb{patient}"
    summary = os.path.join(input_dir, folder, f"{folder}-summary.txt")
    if not os.path.isfile(summary):
        print(f"  skipping {folder}: no {os.path.basename(summary)} in {input_dir}")
        return 0
    channel_names, channel_indices = summary_channels(summary, block)
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    def write(number, channels):
        filename = f"{folder}_{number}.edf"
        metadata = seizure_metadata(summary, filename)
        metadata["channels"] = channel_names
        channels["metadata"] = metadata
        if not dry_run:
            with open(os.path.join(output_dir, folder, filename + ".pkl"), "wb") as handle:
                pickle.dump(channels, handle)

    from pyedflib import highlevel

    write(reference, read_edf_channels(os.path.join(input_dir, folder, f"{folder}_{reference}.edf"),
                                       channel_indices))
    written = {reference}

    # As in the reference, this range can include the reference recording, which is then
    # rewritten through the same path as every other recording.
    for number in (f"{n:02d}" for n in range(first, last + 1)):
        path = os.path.join(input_dir, folder, f"{folder}_{number}.edf")
        if not os.path.isfile(path):
            continue  # recording numbers have gaps
        labels = [header["label"] for header in highlevel.read_edf_header(path)["SignalHeaders"]]
        keep = [index for index, label in enumerate(labels) if label in channel_names]
        if not keep:
            continue

        # Keep exactly the reference channels, zero-filling any this recording lacks.
        channels = {
            name: signal for name, signal in read_edf_channels(path, keep).items()
            if name in channel_names
        }
        length = len(next(iter(channels.values())))
        for name in channel_names:
            channels.setdefault(name, np.zeros(length, dtype=float))
        write(number, channels)
        written.add(number)
    return len(written)


def main():
    """Convert every recording in the reference table."""
    args = build_arg_parser("CHB-MIT EDF recordings to per-recording pickles").parse_args()
    tasks = [
        (patient, reference, first, last, block, args.input_dir, args.output_dir, args.dry_run)
        for patient, reference, first, last, block in RECORDINGS
    ]
    total = sum(run_jobs(convert_patient, tasks, args.num_workers, "CHB-MIT patients"))
    verb = "found" if args.dry_run else "written"
    print(f"\nCHB-MIT: {total} recordings {verb} for {len(tasks)} patients in {args.output_dir}")


if __name__ == "__main__":
    main()
