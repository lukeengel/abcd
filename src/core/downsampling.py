"""Downsampling utilities for class imbalance."""

import numpy as np


def get_balanced_indices(y, seed):
    """Downsample majority class to 1:1 ratio with minority class."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_min = min(len(pos_idx), len(neg_idx))

    # downsample whichever class has more samples
    if len(pos_idx) < len(neg_idx):
        sampled_neg = rng.choice(neg_idx, size=n_min, replace=False)
        balanced_idx = np.concatenate([pos_idx, sampled_neg])
    else:
        sampled_pos = rng.choice(pos_idx, size=n_min, replace=False)
        balanced_idx = np.concatenate([sampled_pos, neg_idx])

    rng.shuffle(balanced_idx)
    return balanced_idx
