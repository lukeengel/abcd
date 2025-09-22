"""Functions that build modeling splits."""

from __future__ import annotations

import pandas as pd


def timepoint_split(env, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into baseline-only"""

    config = env.configs.data
    mapping = config["columns"]["mapping"]
    timepoint_col = mapping["timepoint"]

    baseline_timepoint = config["timepoints"]["baseline"]
    baseline_df = df[df[timepoint_col] == baseline_timepoint].copy()
    longitudinal_df = df.copy()

    return baseline_df, longitudinal_df
