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
#* Author:  BioFoundation Contributors                                       *
#*----------------------------------------------------------------------------*

"""Generate synthetic S-CEReBrO datasets for smoke-testing the training pipeline.

This writes data in exactly the on-disk formats the real readers expect, so a run
started against it exercises the same code path as a run against prepared data. The
signals are band-limited noise and the labels are random: the output is for checking
that a pipeline runs, not for measuring anything.

Two formats are produced.

``LMDBDataset`` corpora
    An LMDB directory per split whose values are pickled dictionaries with ``eeg``,
    ``channel_coords`` and optionally ``label``, plus a ``{split}.meta.json`` index
    listing the keys so start-up does not scan the database.

``TUEGDataset`` corpus
    A single LMDB whose values are fixed-size raw blobs holding a padded
    ``float32[max_channels, timesteps]`` waveform followed by
    ``float32[max_channels, 2, 3]`` coordinates, plus a newline-separated key list.
    Padded channels are written with all-zero coordinates, which is how the reader
    recovers ``num_padded_channels``.

Example:
    python -m make_datasets.make_dummy_scerebro_dataset --output $DATA_PATH
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Optional, Sequence

import lmdb
import numpy as np

SAMPLING_RATE = 200
MAP_SIZE = 1 << 30

# Pre-training corpora named by config/data_module/pretrain_data_module_SCEReBrO.yaml.
PRETRAIN_CORPORA = ("SEED", "SEED-IV", "SEED-GER", "SEED-FRA", "BOAS", "GWD", "SleepEDFx", "BCI-NER")


def make_channel_coords(num_channels: int, rng: np.random.Generator) -> np.ndarray:
    """Return plausible unit-sphere electrode-pair coordinates of shape (C, 2, 3).

    Each channel is described by two electrodes. The first is placed on a sphere of
    unit radius and the second is a nearby point, which is what a bipolar derivation
    or a scalp-electrode-plus-reference channel looks like to the encoder.
    """
    theta = rng.uniform(0, np.pi, size=num_channels)
    phi = rng.uniform(0, 2 * np.pi, size=num_channels)
    first = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
    )
    second = first + rng.normal(scale=0.15, size=(num_channels, 3))
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack([first, second], axis=1).astype(np.float32)


def make_eeg(num_channels: int, num_timesteps: int, rng: np.random.Generator) -> np.ndarray:
    """Return band-limited noise of shape (C, T), loosely resembling scalp EEG."""
    t = np.arange(num_timesteps) / SAMPLING_RATE
    signal = np.zeros((num_channels, num_timesteps), dtype=np.float32)
    for freq in (2.0, 6.0, 10.0, 21.0):
        amplitude = rng.uniform(0.2, 1.0, size=(num_channels, 1))
        phase = rng.uniform(0, 2 * np.pi, size=(num_channels, 1))
        signal += (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    signal += rng.normal(scale=0.3, size=signal.shape).astype(np.float32)
    return signal * 20.0


def write_lmdb_split(
    directory: Path,
    split: str,
    num_samples: int,
    num_channels: int,
    num_timesteps: int,
    rng: np.random.Generator,
    label_kind: str = "none",
    num_classes: int = 2,
    sequence_length: Optional[int] = None,
) -> None:
    """Write one ``{split}.lmdb`` of pickled samples plus its ``{split}.meta.json`` index.

    ``label_kind`` is one of ``none``, ``classification`` or ``regression``. It must be
    ``none`` for pre-training corpora: those samples are concatenated with TUEG, which
    carries no label, and a batch mixing samples with and without a ``label`` key
    cannot be collated.
    """
    if label_kind not in {"none", "classification", "regression"}:
        raise ValueError(f"label_kind must be none, classification or regression, got {label_kind!r}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}.lmdb"
    env = lmdb.open(str(path), map_size=MAP_SIZE)
    keys = []

    with env.begin(write=True) as txn:
        for index in range(num_samples):
            coords = make_channel_coords(num_channels, rng)
            if sequence_length is None:
                eeg = make_eeg(num_channels, num_timesteps, rng)
                if label_kind == "classification":
                    label = int(rng.integers(num_classes))
                elif label_kind == "regression":
                    label = float(rng.uniform(0.0, 1.0))
                else:
                    label = None
            else:
                eeg = np.stack(
                    [make_eeg(num_channels, num_timesteps, rng) for _ in range(sequence_length)]
                )
                label = (
                    rng.integers(num_classes, size=sequence_length).astype(np.int64).tolist()
                    if label_kind == "classification"
                    else None
                )

            entry = {"eeg": eeg, "channel_coords": coords, "subject_id": f"S{index % 4:02d}"}
            if label is not None:
                entry["label"] = label

            key = f"{split}_{index:06d}".encode()
            txn.put(key, pickle.dumps(entry))
            keys.append(key.decode())

    env.close()
    (directory / f"{split}.meta.json").write_text(json.dumps({"keys": keys}))
    print(f"  {path}  ({num_samples} samples)")


def write_tueg(
    directory: Path,
    num_samples: int,
    max_channels: int,
    real_channels: int,
    num_timesteps: int,
    rng: np.random.Generator,
) -> None:
    """Write the fixed-size-blob LMDB and key list that ``TUEGDataset`` reads."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "TUEG.lmdb"
    env = lmdb.open(str(path), map_size=MAP_SIZE)
    keys = []

    with env.begin(write=True) as txn:
        for index in range(num_samples):
            waveform = np.zeros((max_channels, num_timesteps), dtype=np.float32)
            coords = np.zeros((max_channels, 2, 3), dtype=np.float32)

            # Vary how many channels are real so padding is genuinely exercised.
            active = real_channels if index % 2 == 0 else max(4, real_channels // 2)
            signal = make_eeg(active, num_timesteps, rng)
            # TUEGDataset returns the stored waveform unchanged, so windows are
            # min-max normalised offline exactly as make_tueg does. Without this the
            # TUEG windows would be on a different scale from the LMDB corpora they
            # are concatenated with.
            minimum = signal.min(axis=1, keepdims=True)
            maximum = signal.max(axis=1, keepdims=True)
            waveform[:active] = ((signal - minimum) / (maximum - minimum + 1e-10) - 0.5) * 2.0
            coords[:active] = make_channel_coords(active, rng)

            key = f"tueg_{index:06d}".encode()
            txn.put(key, waveform.tobytes() + coords.tobytes())
            keys.append(key.decode())

    env.close()
    (directory / "TUEG_keys.txt").write_text("\n".join(keys) + "\n")
    print(f"  {path}  ({num_samples} windows, padded to {max_channels} channels)")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output", required=True,
        help="Root directory; pretraining/ and finetuning/ are created beneath it. "
             "Use the same path you export as DATA_PATH.",
    )
    parser.add_argument("--pretrain-samples", type=int, default=64, help="Windows per pre-training corpus.")
    parser.add_argument("--finetune-samples", type=int, default=64, help="Windows in the fine-tuning train split.")
    parser.add_argument("--max-channels", type=int, default=64, help="Channel capacity of the pre-training corpora.")
    parser.add_argument("--slice-duration", type=int, default=30, help="Pre-training window length in seconds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--datasets", nargs="+", default=["pretrain", "tuab"],
        choices=["pretrain", "tuab", "isruc", "seed-vig"],
        help="Which corpora to generate. 'pretrain' builds TUEG plus the eight LMDB corpora.",
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    root = Path(args.output).expanduser().resolve()
    pretrain_timesteps = SAMPLING_RATE * args.slice_duration

    if "pretrain" in args.datasets:
        print(f"Pre-training corpora -> {root/'pretraining'}")
        write_tueg(
            root / "pretraining" / "TUEG", args.pretrain_samples,
            args.max_channels, args.max_channels // 2, pretrain_timesteps, rng,
        )
        for corpus in PRETRAIN_CORPORA:
            # Fewer channels than capacity, so the loader's zero-padding path runs.
            write_lmdb_split(
                root / "pretraining" / corpus, "train", args.pretrain_samples,
                num_channels=args.max_channels // 2, num_timesteps=pretrain_timesteps,
                rng=rng, label_kind="none",
            )

    if "tuab" in args.datasets:
        print(f"TUAB (22 channels, 10 s, 2 classes) -> {root/'finetuning'/'TUAB'}")
        for split, count in (("train", args.finetune_samples),
                             ("val", max(8, args.finetune_samples // 4)),
                             ("test", max(8, args.finetune_samples // 4))):
            write_lmdb_split(
                root / "finetuning" / "TUAB", split, count,
                num_channels=22, num_timesteps=10 * SAMPLING_RATE,
                rng=rng, label_kind="classification", num_classes=2,
            )

    if "isruc" in args.datasets:
        print(f"ISRUC (6 channels, 30 s x 20 epochs, 5 classes) -> {root/'finetuning'/'ISRUC'}")
        for split, count in (("train", args.finetune_samples // 4),
                             ("val", 8), ("test", 8)):
            write_lmdb_split(
                root / "finetuning" / "ISRUC", split, max(4, count),
                num_channels=6, num_timesteps=30 * SAMPLING_RATE,
                rng=rng, label_kind="classification", num_classes=5, sequence_length=20,
            )

    if "seed-vig" in args.datasets:
        print(f"SEED-VIG (17 channels, 8 s, continuous) -> {root/'finetuning'/'SEED-VIG'}")
        for split, count in (("train", args.finetune_samples),
                             ("val", 8), ("test", 8)):
            write_lmdb_split(
                root / "finetuning" / "SEED-VIG", split, count,
                num_channels=17, num_timesteps=8 * SAMPLING_RATE,
                rng=rng, label_kind="regression",
            )

    print(f"\nDone. Point DATA_PATH at {root}")


if __name__ == "__main__":
    main()
