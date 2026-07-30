import argparse
import os
import pickle
from collections import Counter
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
    """Return the 3D coordinate of one electrode, or the origin if it is unknown."""
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
    not hold one unbounded write transaction.

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

        self.txns[split].put(key, pickle.dumps(sample))
        self.written += 1
        if self.written % COMMIT_INTERVAL == 0:
            for name in self.splits:
                self.txns[name].commit()
                self.txns[name] = self.envs[name].begin(write=True)

    def close(self) -> None:
        """Commit and close every LMDB."""
        if self.dry_run:
            return
        for split in self.splits:
            self.txns[split].commit()
            self.envs[split].close()

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
        if len(self.splits) > 1:
            overlap = set.intersection(*(self.subjects[s] for s in self.splits)) if all(
                self.subjects[s] for s in self.splits
            ) else set()
            if overlap:
                print(f"  WARNING: {len(overlap)} subjects appear in more than one split")


class PackedLMDBWriter:
    """Writes fixed-size raw byte records into one LMDB alongside a key list.

    Used for the largest pre-training corpus, where pickling every sample is a
    measurable overhead. Each value is a waveform followed by its channel
    coordinates, both float32, so readers can slice the blob without unpickling.

    Args:
        lmdb_path: Path of the LMDB to create.
        keys_path: Path of the newline-separated key list to write.
        map_size: Maximum size of the LMDB in bytes.
        dry_run: Collect statistics without writing anything.
    """

    def __init__(self, lmdb_path: str, keys_path: str, map_size: int = DEFAULT_MAP_SIZE, dry_run: bool = False):
        self.lmdb_path = lmdb_path
        self.keys_path = keys_path
        self.dry_run = dry_run
        self.keys: List[str] = []
        self.written = 0
        self.env = None
        self.txn = None
        if not dry_run:
            os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
            self.env = lmdb.open(lmdb_path, map_size=map_size)
            self.txn = self.env.begin(write=True)

    def put(self, key: str, waveform: np.ndarray, channel_coords: np.ndarray) -> None:
        """Record one window as a packed byte blob."""
        self.keys.append(key)
        if self.dry_run:
            return
        payload = waveform.astype(np.float32).tobytes() + channel_coords.astype(np.float32).tobytes()
        self.txn.put(key.encode("ascii"), payload)
        self.written += 1
        if self.written % COMMIT_INTERVAL == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> None:
        """Commit the LMDB and write the key list."""
        if self.dry_run:
            return
        self.txn.commit()
        self.env.close()
        with open(self.keys_path, "w") as handle:
            handle.write("\n".join(self.keys) + "\n")

    def summarise(self, name: str) -> None:
        """Print the number of windows written."""
        verb = "computed" if self.dry_run else "written"
        print(f"\n{name}: {len(self.keys)} windows {verb} to {self.lmdb_path}")


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
