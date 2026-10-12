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
#*----------------------------------------------------------------------------*

"""Preprocess the TUEG pre-training corpus into a packed LMDB.

TUEG is the largest corpus, so each window is stored as a fixed-size byte blob rather
than a pickled dictionary, and the sample keys are written to a companion text file so
readers do not have to scan the database at start-up.

Subjects that also appear in TUAB are excluded, so that pre-training cannot see any
recording from a subject used in the TUAB fine-tuning splits.

A corpus this size holds files that cannot be used: unreadable, sampled too slowly for
the 75 Hz low-pass, without any known channel, or shorter than one window. Each one is
skipped with its reason, printed and listed in TUEG_skipped.tsv, so that no single file
can end a run of many hours. The reference skipped such files too, but only printed
the error.
"""

import os
import re
import shutil
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import lmdb
import mne
import numpy as np
from tqdm import tqdm

from make_datasets.common import (
    DEFAULT_MAP_SIZE,
    SAMPLING_FREQ,
    PackedLMDBWriter,
    build_arg_parser,
    electrode_coordinate,
    run_jobs_isolated,
)

SLICE_SECONDS = 30
WINDOW_SAMPLES = SAMPLING_FREQ * SLICE_SECONDS
MAX_CHANNELS = 64
# One padded window and its electrode coordinates, as stored.
RECORD_BYTES = MAX_CHANNELS * (WINDOW_SAMPLES + 2 * 3) * np.dtype(np.float32).itemsize
# File names become key prefixes, in an ASCII key list with one key per line.
KEY_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,200}")

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


def describe(error: BaseException) -> str:
    """One line naming an exception, short enough for the skip list."""
    return f"{type(error).__name__}: {' '.join(str(error).split())}"[:300]


def process_recording(path: str):
    """Filter, normalise and slice one recording.

    Returns ``(path, windows, problem)``. Windows are unpadded, so that workers send
    back less; the writer pads them. Each is min-max normalised per channel to
    ``[-1, 1]`` before packing, because the packed format stores no per-sample
    statistics for a reader to apply. A recording that yields no windows comes back
    with ``problem``, a ``(reason, detail)`` pair, instead of raising.
    """
    session = Path(path).stem
    session_reference = "LE" if ("02_tcp_le" in path or "04_tcp_le" in path) else "AR"
    stage = "reading"
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
        if raw.n_times == 0:
            return path, [], ("no samples", "")
        # Checked before dropping: MNE refuses to drop every channel.
        known = [name for name in ALL_CHANNELS if name in raw.ch_names]
        if not known:
            return path, [], ("no channel with a known name", "")
        raw.drop_channels([name for name in raw.ch_names if name not in ALL_CHANNELS])
        raw.reorder_channels(known)

        stage = "computing coordinates"
        coords = np.asarray(
            [channel_coordinates(name, session_reference) for name in raw.ch_names], dtype=np.float32
        )

        # Recordings at 150 Hz or below fail here: the low-pass must sit below Nyquist.
        stage = "filtering"
        raw.notch_filter(60, verbose="error")
        raw.filter(l_freq=0.3, h_freq=75.0, verbose="error")
        if raw.info["sfreq"] != SAMPLING_FREQ:
            stage = "resampling"
            raw.resample(SAMPLING_FREQ, n_jobs=1, verbose="error")

        stage = "windowing"
        data = raw.get_data().astype(np.float32) * 1e6
        del raw
        num_channels, num_timesteps = data.shape
        if num_channels > MAX_CHANNELS:
            return path, [], (f"more than {MAX_CHANNELS} channels", str(num_channels))
        if num_timesteps < WINDOW_SAMPLES:
            return path, [], (f"shorter than {SLICE_SECONDS} s", f"{num_timesteps / SAMPLING_FREQ:.1f} s")
        # Not in the reference. EDF samples are finite, so this only catches broken
        # scaling, which would otherwise put NaN into pre-training.
        if not np.isfinite(data).all():
            return path, [], ("non-finite samples", "")

        windows = []
        for index in range(num_timesteps // WINDOW_SAMPLES):
            window = data[:, index * WINDOW_SAMPLES : (index + 1) * WINDOW_SAMPLES]
            minimum = window.min(axis=1, keepdims=True)
            maximum = window.max(axis=1, keepdims=True)
            normalised = ((window - minimum) / (maximum - minimum + 1e-10) - 0.5) * 2.0
            windows.append((f"{session}_slice_{index:04d}", normalised, coords))
        return path, windows, None
    except Exception as error:  # noqa: BLE001 - any failure skips this recording only
        return path, [], (f"failed while {stage}", describe(error))


def worker_failure(path: str, reason: str):
    """Result for a recording whose worker raised or died; see run_jobs_isolated."""
    return path, [], ("worker failed", reason)


def select_recordings(recordings: Sequence[str], tuab_subjects: set):
    """Choose the recordings to process.

    Returns ``(tasks, set_aside, held_out)``: the paths to process, a
    ``(path, reason, detail)`` entry for each file that cannot be used, and the number
    of files held out as TUAB subjects.
    """
    tasks, set_aside, first_seen, held_out = [], [], {}, 0
    for path in recordings:
        session = Path(path).stem
        if session.split("_s")[0] in tuab_subjects:
            held_out += 1
        elif not KEY_PATTERN.fullmatch(session):
            set_aside.append((path, "file name not usable as a key", ""))
        elif session in first_seen:
            # Both would write the same keys, the second overwriting the first.
            set_aside.append((path, "same file name as another recording", first_seen[session]))
        else:
            first_seen[session] = path
            tasks.append(path)
    return tasks, set_aside, held_out


def count_windows(recordings: Sequence[str]) -> int:
    """Count the windows a run can write, from the EDF headers alone.

    This is an upper bound, since it cannot know which recordings will fail. A header
    that cannot be read is not counted; its recording is reported when processed.
    """
    windows = unreadable = 0
    for path in tqdm(recordings, desc="Counting windows"):
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
            kept = sum(name in ALL_CHANNELS for name in raw.ch_names)
            if 0 < kept <= MAX_CHANNELS:
                windows += int(raw.n_times * SAMPLING_FREQ / raw.info["sfreq"]) // WINDOW_SAMPLES
        except Exception:  # noqa: BLE001 - reported when the recording is processed
            unreadable += 1
    if unreadable:
        print(f"{unreadable} headers could not be read; those recordings are reported when processed")
    return windows


def map_size_for(windows: int) -> int:
    """LMDB reservation for ``windows`` records, with 10% headroom for LMDB's own pages."""
    return max(int(RECORD_BYTES * windows * 1.1), DEFAULT_MAP_SIZE)


def warn_if_disk_short(output_dir: str, needed: int) -> None:
    """Warn when the output filesystem has less free space than the data will take.

    LMDB's reservation is address space, not disk, so only the data is compared. A
    cluster quota can be smaller than the free space and is not visible here.
    """
    existing = os.path.abspath(output_dir)
    while not os.path.exists(existing):
        existing = os.path.dirname(existing)
    try:
        free = shutil.disk_usage(existing).free
    except OSError:
        return
    if free < needed:
        print(
            f"WARNING: about {needed / 1e12:.2f} TB will be written, but only {free / 1e12:.2f} TB "
            f"is free on the filesystem holding {output_dir}"
        )


class SkipLog:
    """Reports each recording that yields no windows and lists it in a TSV file.

    Args:
        path: The TSV file to write, or None to only print, as in a dry run.
    """

    def __init__(self, path: Optional[str]):
        self.path = path
        self.reasons: Counter = Counter()
        self.handle = None
        if path is not None:
            self.handle = open(path, "w")
            self.handle.write("path\treason\tdetail\n")

    def add(self, recording: str, reason: str, detail: str = "") -> None:
        """Report one skipped recording."""
        self.reasons[reason] += 1
        tqdm.write(f"  skipped {recording}: {reason}" + (f" ({detail})" if detail else ""))
        if self.handle is not None:
            self.handle.write(f"{recording}\t{reason}\t{detail}\n")
            self.handle.flush()

    def close(self) -> None:
        """Close the TSV file."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def summarise(self) -> None:
        """Print how many recordings were skipped, by reason."""
        total = sum(self.reasons.values())
        listed = f", listed in {self.path}" if self.path else ""
        print(f"  {total} recordings skipped{listed}")
        for reason, count in self.reasons.most_common():
            print(f"    {count:8d}  {reason}")


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
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument(
        "--num_windows", type=int, default=None,
        help="Size the LMDB for this many 30 s windows, such as the count a dry run printed, "
             "instead of first reading every EDF header.",
    )
    sizing.add_argument(
        "--map_size_gb", type=float, default=None,
        help="Reserve exactly this many GB (10^9 bytes) for the LMDB.",
    )
    args = parser.parse_args()
    mne.set_log_level("ERROR")
    if args.num_windows is not None and args.num_windows < 1:
        parser.error("--num_windows must be positive")
    if args.map_size_gb is not None and args.map_size_gb <= 0:
        parser.error("--map_size_gb must be positive")

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
    if not recordings:
        parser.error(f"No .edf files found under --input_dir {args.input_dir}")
    tasks, set_aside, held_out = select_recordings(recordings, tuab_subjects)
    print(
        f"{len(recordings)} EDF files: {len(tasks)} to process, {held_out} held out as TUAB "
        f"subjects, {len(set_aside)} set aside"
    )

    lmdb_path = os.path.join(args.output_dir, "TUEG.lmdb")
    map_size = DEFAULT_MAP_SIZE
    if not args.dry_run:
        if os.path.exists(lmdb_path):
            parser.error(
                f"{lmdb_path} already exists; delete it or choose another --output_dir, "
                "since writing into it would mix two runs"
            )
        if args.map_size_gb is not None:
            map_size = int(args.map_size_gb * 1e9)
            expected = int(map_size / 1.1)
        else:
            windows = args.num_windows if args.num_windows is not None else count_windows(tasks)
            map_size = map_size_for(windows)
            expected = windows * RECORD_BYTES
            print(f"Sizing for {windows:,} windows: about {expected / 1e12:.2f} TB of data")
        print(f"LMDB map_size = {map_size:,} bytes; disk is used only as data is written")
        warn_if_disk_short(args.output_dir, expected)
        os.makedirs(args.output_dir, exist_ok=True)

    skips = SkipLog(None if args.dry_run else os.path.join(args.output_dir, "TUEG_skipped.tsv"))
    try:
        writer = PackedLMDBWriter(
            lmdb_path=lmdb_path,
            keys_path=os.path.join(args.output_dir, "TUEG_keys.txt"),
            map_size=map_size,
            dry_run=args.dry_run,
        )
    except (lmdb.Error, OSError) as error:
        sys.exit(
            f"Could not create {lmdb_path} with map_size {map_size:,}: {error}. The map is "
            "reserved as virtual memory, so `ulimit -v` must allow it."
        )
    for path, reason, detail in set_aside:
        skips.add(path, reason, detail)

    # A cluster stops a job at its time limit with SIGTERM: stop as on Ctrl-C, so that the
    # key list is still written for what was committed.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    padded = np.zeros((MAX_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
    padded_coords = np.zeros((MAX_CHANNELS, 2, 3), dtype=np.float32)
    results = run_jobs_isolated(process_recording, tasks, args.num_workers, "TUEG recordings", worker_failure)
    try:
        for path, windows, problem in results:
            if problem is not None:
                skips.add(path, *problem)
            for key, waveform, coords in windows:
                padded.fill(0)
                padded[: len(waveform)] = waveform
                padded_coords.fill(0)
                padded_coords[: len(coords)] = coords
                writer.put(key, padded, padded_coords)
        writer.close()
    except lmdb.Error as error:
        writer.abort()
        skips.close()
        results.close()
        advice = (
            "rerun with a larger --map_size_gb or --num_windows" if isinstance(error, lmdb.MapFullError)
            else "check the disk and its free space, then rerun"
        )
        sys.exit(
            f"Writing the LMDB failed after {writer.committed:,} committed windows, which "
            f"{writer.keys_path} lists: {error}. Delete {lmdb_path} and {advice}."
        )
    except BaseException:
        # The writer first: shutting the workers down can wait for a long recording.
        writer.abort()
        skips.close()
        results.close()
        raise
    skips.close()

    writer.summarise("TUEG")
    size = writer.count * RECORD_BYTES / 1e12
    if args.dry_run:
        print(f"  a real run writes about {size:.2f} TB; pass --num_windows {writer.count} to size it")
    else:
        print(f"  {size:.2f} TB of data")
    skips.summarise()


if __name__ == "__main__":
    main()
