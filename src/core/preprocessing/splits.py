"""Functions that build modeling splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def timepoint_split(env, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return baseline-only rows and a longitudinal copy."""

    config = env.configs.data
    timepoint_col = config["columns"]["mapping"]["timepoint"]
    baseline_timepoint = config["timepoints"]["baseline"]

    baseline_df = df[df[timepoint_col] == baseline_timepoint].copy()
    longitudinal_df = df.copy()
    return baseline_df, longitudinal_df


def create_modeling_splits(
    env, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split using anxiety groups."""

    config = env.configs.data
    run_cfg = getattr(env.configs, "run", {}) or {}
    seed = run_cfg.get("seed", 42)

    ratio_cfg = config.get("splits", {})
    ratios = np.array(
        [
            float(ratio_cfg.get("train", 0.8)),
            float(ratio_cfg.get("val", 0.1)),
            float(ratio_cfg.get("test", 0.1)),
        ]
    )
    ratios /= ratios.sum()

    anxiety_col = config["columns"]["mapping"]["anx_group"]
    id_col = config["columns"]["mapping"]["id"]
    timepoint_col = config["columns"]["mapping"]["timepoint"]

    qc_pass_df = df[df["qc_pass"]].copy().reset_index(drop=True)
    if qc_pass_df.empty:
        raise ValueError("No QC-pass rows available for splitting")

    labels = qc_pass_df[anxiety_col].astype("string").fillna("unknown")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=ratios[0],
        random_state=seed,
    )
    train_idx, remaining_idx = next(sss.split(qc_pass_df, labels))

    train = qc_pass_df.iloc[train_idx].copy()
    remaining = qc_pass_df.iloc[remaining_idx].copy()
    remaining_labels = labels.iloc[remaining_idx]

    if np.isclose(ratios[2], 0.0):
        val = remaining.copy()
        test = remaining.iloc[0:0].copy()
    else:
        remainder_ratio = ratios[1] + ratios[2]
        second_split = StratifiedShuffleSplit(
            n_splits=1,
            train_size=ratios[1] / remainder_ratio,
            random_state=seed,
        )
        val_idx, test_idx = next(second_split.split(remaining, remaining_labels))
        val = remaining.iloc[val_idx].copy()
        test = remaining.iloc[test_idx].copy()

    split_map = pd.concat(
        [
            train[[id_col, timepoint_col]].assign(split="train"),
            val[[id_col, timepoint_col]].assign(split="val"),
            test[[id_col, timepoint_col]].assign(split="test"),
        ],
        ignore_index=True,
    )

    return train, val, test, split_map
