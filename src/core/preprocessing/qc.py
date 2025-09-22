"""Quality control helpers for preprocessing."""

from __future__ import annotations

import pandas as pd


def quality_control(
    env, df: pd.DataFrame, *, copy: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate dataset with QC pass/fail and return the mask."""

    config = env.configs.data
    column_map = config["columns"]["mapping"]
    id_col = column_map["id"]
    timepoint_col = column_map["timepoint"]
    qc_col = column_map.get("qc")

    if qc_col is None or qc_col not in df.columns:
        raise KeyError("QC column not present in dataframe")

    threshold = config["qc_thresholds"]["surface_holes_max"]
    qc_df = df.copy() if copy else df

    qc_df["qc_reason"] = None
    qc_df.loc[qc_df[qc_col].isna(), "qc_reason"] = "missing_surface_holes"
    qc_df.loc[qc_df[qc_col] > threshold, "qc_reason"] = "surface_holes_gt_threshold"

    qc_df["qc_pass"] = qc_df["qc_reason"].isna()

    mask = qc_df[[id_col, timepoint_col, qc_col, "qc_pass", "qc_reason"]].copy()

    return qc_df, mask
