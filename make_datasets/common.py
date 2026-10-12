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

import argparse
import json
import multiprocessing
import os
import pickle
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import lmdb
import numpy as np
from tqdm import tqdm

from make_datasets import electrode_positions

SAMPLING_FREQ = 200
SPLITS = ("train", "val", "test")
DEFAULT_MAP_SIZE = 60 * 10 ** 9
COMMIT_INTERVAL = 4000


def electrode_coordinate(name: str) -> Tuple[float, float, float]:
    """Return the 3D coordinate of one electrode, or the origin if it is unknown.

    The origin fallback matches the reference preprocessing that produced the published
    corpora, so it is kept deliberately. It is also what padded channels carry, so an
    unrecognised name is indistinguishable from padding; every electrode in every
    montage shipped here resolves, and a new montage should be checked against
    ELECTRODE_ANGLES and SPECIAL_REFERENCE_POSITIONS before use.
    """
    if name in electrode_positions.ELECTRODE_ANGLES:
        angles = electrode_positions.ELECTRODE_ANGLES[name]
        return electrode_positions.get_electrode_3d_positions(angles["theta"], angles["phi"])
    return electrode_positions.SPECIAL_REFERENCE_POSITIONS.get(name, (0.0, 0.0, 0.0))


def referential_coordinates(channel_names: Sequence[str], reference: str) -> np.ndarray:
    """Build channel coordinates for a montage where every channel shares one reference.

    Args:
        channel_names: Electrode names in the order the channels appear in the data.
        reference: Name of the shared reference electrode.

    Returns:
        Array of shape ``(num_channels, 2, 3)`` holding the active and reference
        coordinates of each channel.
    """
    reference_position = electrode_coordinate(reference)
    coords = np.zeros((len(channel_names), 2, 3), dtype=np.float32)
    for index, name in enumerate(channel_names):
        coords[index, 0, :] = electrode_coordinate(name)
        coords[index, 1, :] = reference_position
    return coords


def bipolar_coordinates(pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
    """Build channel coordinates for a bipolar montage.

    Args:
        pairs: ``(active, reference)`` electrode names for each channel.

    Returns:
        Array of shape ``(num_channels, 2, 3)``.
    """
    coords = np.zeros((len(pairs), 2, 3), dtype=np.float32)
    for index, (active, reference) in enumerate(pairs):
        coords[index, 0, :] = electrode_coordinate(active)
        coords[index, 1, :] = electrode_coordinate(reference)
    return coords


def index_splits(items: Sequence[Any], train_end: int, val_end: int) -> Dict[str, List[Any]]:
    """Split an ordered sequence of recordings into train, validation and test.

    Splitting on recordings rather than windows keeps every window of a recording,
    and therefore of a subject, inside a single split.

    Args:
        items: Recordings in a deterministic order.
        train_end: Number of leading recordings used for training.
        val_end: Index at which validation ends and test begins.

    Returns:
        Mapping from split name to the recordings assigned to it.
    """
    if not 0 < train_end <= val_end <= len(items):
        raise ValueError(
            f"Invalid split boundaries train_end={train_end} val_end={val_end} for {len(items)} recordings"
        )
    return {"train": list(items[:train_end]), "val": list(items[train_end:val_end]), "test": list(items[val_end:])}


class LMDBWriter:
    """Writes pickled EEG samples into one LMDB per split and reports the result.

    Transactions are committed every ``COMMIT_INTERVAL`` samples so that a long job does
    not hold one unbounded write transaction. Closing writes ``{split}.meta.json`` next to
    each LMDB, in the reference format ``{"keys", "subject_ids", "labels"}``, so readers
    index a split without scanning it.

    Args:
        output_dir: Directory to create the LMDB files in.
        splits: Split names to open an LMDB for. A single-element sequence produces one
            pooled database, which is what the pre-training corpora use.
        map_size: Maximum size of each LMDB in bytes.
        dry_run: Collect statistics without writing anything.
    """

    def __init__(
        self,
        output_dir: str,
        splits: Sequence[str] = SPLITS,
        map_size: int = DEFAULT_MAP_SIZE,
        dry_run: bool = False,
    ):
        self.output_dir = output_dir
        self.splits = tuple(splits)
        self.dry_run = dry_run
        self.written = 0
        self.counts: Dict[str, int] = {split: 0 for split in self.splits}
        self.labels: Dict[str, Counter] = {split: Counter() for split in self.splits}
        self.shapes: Counter = Counter()
        self.subjects: Dict[str, set] = {split: set() for split in self.splits}
        # Insertion-ordered, and a repeated key replaces its entry exactly as LMDB does.
        self.index: Dict[str, Dict[str, Tuple[str, Any]]] = {split: {} for split in self.splits}

        self.envs: Dict[str, Any] = {}
        self.txns: Dict[str, Any] = {}
        if not dry_run:
            os.makedirs(output_dir, exist_ok=True)
            for split in self.splits:
                self.envs[split] = lmdb.open(os.path.join(output_dir, f"{split}.lmdb"), map_size=map_size)
                self.txns[split] = self.envs[split].begin(write=True)

    def put(self, split: str, key: bytes, sample: Dict[str, Any]) -> None:
        """Record one sample, writing it unless this is a dry run."""
        self.counts[split] += 1
        eeg = sample["eeg"]
        self.shapes[tuple(eeg.shape)] += 1

        label = sample.get("label")
        if label is not None:
            for value in np.atleast_1d(np.asarray(label)).ravel():
                self.labels[split][value.item()] += 1
        if "subject_id" in sample:
            self.subjects[split].add(str(sample["subject_id"]))

        if self.dry_run:
            return

        self.index[split][key.decode()] = (
            str(sample.get("subject_id", "unknown")),
            None if label is None else np.asarray(label).tolist(),
        )
        self.txns[split].put(key, pickle.dumps(sample))
        self.written += 1
        if self.written % COMMIT_INTERVAL == 0:
            for name in self.splits:
                self.txns[name].commit()
                self.txns[name] = self.envs[name].begin(write=True)

    def close(self) -> None:
        """Commit and close every LMDB, then write its index file."""
        if self.dry_run:
            return
        for split in self.splits:
            self.txns[split].commit()
            self.envs[split].close()
            entries = self.index[split]
            meta = {
                "keys": list(entries),
                "subject_ids": [subject for subject, _ in entries.values()],
                "labels": [label for _, label in entries.values()],
            }
            with open(os.path.join(self.output_dir, f"{split}.meta.json"), "w") as handle:
                json.dump(meta, handle)

    def summarise(self, name: str) -> None:
        """Print per-split sample counts, label distributions and window shapes."""
        verb = "computed" if self.dry_run else "written"
        print(f"\n{name}: {sum(self.counts.values())} samples {verb} to {self.output_dir}")
        for split in self.splits:
            subjects = self.subjects[split]
            suffix = f", {len(subjects)} subjects" if subjects else ""
            print(f"  {split:5s}: {self.counts[split]} samples{suffix}")
            if self.labels[split]:
                distribution = ", ".join(
                    f"{label}:{count}" for label, count in sorted(self.labels[split].items())
                )
                print(f"           labels {distribution}")
        print("  window shapes: " + ", ".join(f"{shape}:{count}" for shape, count in sorted(self.shapes.items())))
        # Pairwise: intersecting all splits at once misses a train/test-only leak.
        names = sorted(self.splits)
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                overlap = self.subjects[first] & self.subjects[second]
                if overlap:
                    sample = ", ".join(sorted(overlap)[:5])
                    print(
                        f"  WARNING: {len(overlap)} subjects appear in both "
                        f"{first} and {second} ({sample}...)"
                    )


class PackedLMDBWriter:
    """Writes fixed-size raw byte records into one LMDB alongside a key list.

    Used for the largest pre-training corpus, where pickling every sample is a
    measurable overhead. Each value is a waveform followed by its channel
    coordinates, both float32, so readers can slice the blob without unpickling.

    The key list is written when the writer closes or aborts, by listing the LMDB
    itself, so it names exactly the committed records however the run ended, in
    sorted order, which keeps the dataset order independent of worker timing. Until
    then it is empty. LMDB keeps an open write transaction's records in memory, about
    1.5 MB each here, which is why commits are more frequent than for the other corpora.

    Args:
        lmdb_path: Path of the LMDB to create.
        keys_path: Path of the newline-separated key list to write.
        map_size: Maximum size of the LMDB in bytes.
        dry_run: Collect statistics without writing anything.
        commit_interval: Records per write transaction.
    """

    def __init__(
        self,
        lmdb_path: str,
        keys_path: str,
        map_size: int = DEFAULT_MAP_SIZE,
        dry_run: bool = False,
        commit_interval: int = 1000,
    ):
        self.lmdb_path = lmdb_path
        self.keys_path = keys_path
        self.dry_run = dry_run
        self.commit_interval = commit_interval
        self.count = 0
        self.committed = 0
        self.uncommitted = 0
        self.env = None
        self.txn = None
        if not dry_run:
            os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
            self.env = lmdb.open(lmdb_path, map_size=map_size)
            self.txn = self.env.begin(write=True)
            open(keys_path, "w").close()

    def put(self, key: str, waveform: np.ndarray, channel_coords: np.ndarray) -> None:
        """Record one window as a packed byte blob."""
        self.count += 1
        if self.dry_run:
            return
        payload = waveform.astype(np.float32).tobytes() + channel_coords.astype(np.float32).tobytes()
        self.txn.put(key.encode("ascii"), payload)
        self.uncommitted += 1
        if self.uncommitted >= self.commit_interval:
            self.txn.commit()
            self.committed += self.uncommitted
            self.uncommitted = 0
            self.txn = self.env.begin(write=True)

    def _write_key_list(self) -> None:
        """List every stored key, in sorted order, replacing the key list in one step."""
        temporary = f"{self.keys_path}.tmp"
        listed = 0
        with self.env.begin() as txn, open(temporary, "w") as handle:
            for key in txn.cursor().iternext(keys=True, values=False):
                handle.write(bytes(key).decode("ascii") + "\n")
                listed += 1
        os.replace(temporary, self.keys_path)
        self.committed = listed

    def close(self) -> None:
        """Commit the last transaction, write the key list and close the LMDB."""
        if self.dry_run:
            return
        self.txn.commit()
        try:
            self._write_key_list()
        finally:
            self.env.close()

    def abort(self) -> None:
        """Discard the open transaction, list what was committed and close the LMDB."""
        if self.dry_run:
            return
        try:
            self.txn.abort()
        except lmdb.Error:
            pass  # A failed commit has already ended the transaction.
        try:
            self._write_key_list()
        finally:
            self.env.close()

    def summarise(self, name: str) -> None:
        """Print the number of windows written."""
        verb = "computed" if self.dry_run else "written"
        print(f"\n{name}: {self.count} windows {verb} to {self.lmdb_path}")


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Create the argument parser shared by every preprocessing script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_dir", required=True, help="Directory holding the raw dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to write LMDB files into")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--dry_run", action="store_true", help="Report statistics without writing")
    return parser


def run_jobs(
    worker: Callable[[Any], Iterable[Any]], tasks: Sequence[Any], num_workers: int, description: str
) -> Iterable[Any]:
    """Map ``worker`` over ``tasks``, yielding each result as it completes.

    Runs in the calling process when ``num_workers`` is 1, which keeps tracebacks
    readable while debugging a new dataset.
    """
    if num_workers <= 1:
        for task in tqdm(tasks, desc=description):
            yield worker(task)
        return

    from multiprocessing import Pool

    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc=description):
            yield result


def run_jobs_isolated(
    worker: Callable[[Any], Any],
    tasks: Sequence[Any],
    num_workers: int,
    description: str,
    on_failure: Callable[[Any, str], Any],
) -> Iterable[Any]:
    """Map ``worker`` over ``tasks`` in worker processes, yielding each result as it completes.

    For runs of many hours over files that cannot all be trusted, where run_jobs falls
    short twice. A task that raises, or whose worker process dies, for instance at the
    hands of the out-of-memory killer, yields ``on_failure(task, reason)`` and the run
    goes on; multiprocessing.Pool would instead wait forever for the dead worker's task.
    And at most two tasks per worker are in flight, so a consumer slower than the
    workers holds them back, where Pool would keep every finished result in memory.
    Workers start as fresh interpreters, so they inherit neither threads nor open LMDBs;
    the calling script therefore needs the usual ``if __name__ == "__main__"`` guard.
    """
    workers = max(1, num_workers)
    queue = deque(tasks)
    with tqdm(total=len(queue), desc=description) as progress:
        while queue:
            suspects = yield from _run_until_broken(worker, queue, workers, progress, on_failure)
            # Rerun alone, so that a task which kills its process again can only kill its own.
            for task in suspects:
                yield _run_alone(worker, task, on_failure)
                progress.update()


def _worker_context():
    """Multiprocessing context for run_jobs_isolated."""
    return multiprocessing.get_context("spawn")


def _run_until_broken(worker, queue, workers, progress, on_failure):
    """Feed tasks from ``queue`` to one pool until the queue is empty or a worker dies.

    Yields each outcome. A dead worker breaks the whole pool and fails every task in
    flight. The pool starts tasks in submission order, so the tasks that were running
    are the earliest of the failed ones: the first ``workers + 1`` are returned as
    suspects, and the rest go back to the front of the queue. Should the culprit be
    sent back regardless, it breaks the next pool and is caught then.
    """
    pool = ProcessPoolExecutor(workers, mp_context=_worker_context())
    pending: Dict[Future, Tuple[int, Any]] = {}
    submitted = 0
    broken = False
    try:
        while (queue or pending) and not broken:
            while queue and len(pending) < 2 * workers:
                task = queue.popleft()
                try:
                    future = pool.submit(worker, task)
                except BrokenProcessPool:
                    queue.appendleft(task)
                    broken = True
                    break
                pending[future] = (submitted, task)
                submitted += 1
            if broken:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                if isinstance(future.exception(), BrokenProcessPool):
                    broken = True  # Settled with the rest of the pending tasks below.
                    continue
                _, task = pending.pop(future)
                yield _outcome(future, task, on_failure)
                progress.update()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    # Shut down, every pending task has either finished or failed with the pool.
    failed = []
    for future, (order, task) in pending.items():
        if future.cancelled() or isinstance(future.exception(), BrokenProcessPool):
            failed.append((order, task))
        else:
            yield _outcome(future, task, on_failure)
            progress.update()
    failed.sort(key=lambda entry: entry[0])
    queue.extendleft(task for _, task in reversed(failed[workers + 1 :]))
    return [task for _, task in failed[: workers + 1]]


def _run_alone(worker, task, on_failure):
    """Run one task in a process of its own and return its outcome."""
    pool = ProcessPoolExecutor(1, mp_context=_worker_context())
    try:
        future = pool.submit(worker, task)
        wait([future])
    finally:
        pool.shutdown(wait=True)
    return _outcome(future, task, on_failure)


def _outcome(future, task, on_failure):
    """The task's result, or ``on_failure(task, reason)`` when it has none."""
    error = future.exception()
    if error is None:
        return future.result()
    if isinstance(error, BrokenProcessPool):
        return on_failure(task, "worker process died, possibly out of memory")
    if isinstance(error, Exception):
        return on_failure(task, f"{type(error).__name__}: {error}")
    raise error


def list_files(root: str, extensions: Sequence[str]) -> List[str]:
    """Return every file under ``root`` with one of ``extensions``, sorted by relative path."""
    wanted = tuple(extension.lower() for extension in extensions)
    found = []
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(wanted):
                found.append(os.path.relpath(os.path.join(directory, filename), root))
    return sorted(found)


def resample_to_target(signal_array: np.ndarray, source_freq: int, axis: int = -1) -> np.ndarray:
    """Resample along ``axis`` from ``source_freq`` to the project-wide 200 Hz."""
    if source_freq == SAMPLING_FREQ:
        return signal_array
    from scipy import signal as scipy_signal

    num_samples = int(round(signal_array.shape[axis] * SAMPLING_FREQ / source_freq))
    return scipy_signal.resample(signal_array, num_samples, axis=axis)

def slice_windows(data: np.ndarray, window_samples: int) -> np.ndarray:
    """Cut a ``(channels, timesteps)`` recording into non-overlapping windows.

    Returns:
        Array of shape ``(num_windows, channels, window_samples)``; empty when the
        recording is shorter than one window.
    """
    channels, timesteps = data.shape
    num_windows = timesteps // window_samples
    if num_windows == 0:
        return np.empty((0, channels, window_samples), dtype=np.float32)
    trimmed = data[:, : num_windows * window_samples]
    return trimmed.reshape(channels, num_windows, window_samples).transpose(1, 0, 2).astype(np.float32)


def write_pretraining_corpus(name, tasks, worker, output_dir, num_workers, dry_run):
    """Run a pre-training corpus job and write every window into one pooled LMDB.

    Pre-training corpora carry no labels and no splits: the pre-training data module
    holds out a fraction of windows for validation itself.
    """
    writer = LMDBWriter(output_dir, splits=("all",), dry_run=dry_run)
    for samples in run_jobs(worker, tasks, num_workers, f"{name} recordings"):
        for key, sample in samples:
            writer.put("all", key, sample)
    writer.close()
    writer.summarise(name)
