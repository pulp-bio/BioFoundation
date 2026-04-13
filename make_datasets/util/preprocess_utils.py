import os
import mne
import h5py
import numpy as np
from tqdm import tqdm


def segment_raw(raw, window_sec=5):
    """
    Segment an MNE Raw object into fixed-length windows.

    Returns:
        segments: np.ndarray of shape (n_segments, n_channels, window_samples)
    """
    sfreq = int(raw.info['sfreq'])
    window_samples = round(window_sec * sfreq)

    data = raw.get_data()  # (n_channels, n_times)
    n_channels, n_times = data.shape

    n_segments = n_times // window_samples
    if n_segments == 0:
        return np.empty((0, n_channels, window_samples))

    segments = np.stack([
        data[:, i * window_samples:(i + 1) * window_samples]
        for i in range(n_segments)
    ])

    return segments

def write_segments_to_h5_append(
    h5_path,
    segments,
    label=None,
    finetune=True,
    group_size=1000
):
    """
    Appends segmented EEG data to an HDF5 file.
    """

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    # Open in append mode
    with h5py.File(h5_path, "a") as h5f:

        # Determine starting group index
        existing_groups = [
            int(k.split("_")[-1])
            for k in h5f.keys()
            if k.startswith("data_group_")
        ]
        start_idx = max(existing_groups) + 1 if existing_groups else 0

        n_segments = segments.shape[0]

        for i in tqdm(range(0, n_segments, group_size),
                      desc=f"Appending to {os.path.basename(h5_path)}"):

            grp_idx = start_idx + (i // group_size)
            grp = h5f.create_group(f"data_group_{grp_idx}")

            X = segments[i:i + group_size]
            grp.create_dataset("X", data=X, compression="gzip")

            if finetune and label is not None:
                y = np.full((X.shape[0],), label)
                grp.create_dataset("y", data=y)