"""Data ingestion helpers for preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def get_columns_for_file(env, filepath: str | Path) -> list[str] | None:
    """Return columns to load for a file based on config."""

    config = env.configs.data
    columns_cfg = config["columns"]["mapping"]
    files_cfg = config["files"]
    str_path = str(filepath)

    metadata_files: Sequence[str] = files_cfg.get("metadata", [])
    imaging_files: Sequence[str] = files_cfg.get("imaging", [])

    if str_path in metadata_files:
        metadata_cols = list(config["columns"].get("metadata", []))
        derived_cols = list(config["columns"].get("derived", []))
        mapping_cols = [
            value for value in columns_cfg.values() if isinstance(value, str)
        ]
        cols = metadata_cols + derived_cols + mapping_cols
        unique_cols = list(dict.fromkeys(cols))
        return unique_cols

    if str_path in imaging_files:
        return None

    raise ValueError(f"File {filepath} not found in config")


def load_and_merge(env) -> pd.DataFrame:
    """Load and outer-merge metadata and imaging CSVs."""

    config = env.configs.data
    columns_cfg = config["columns"]["mapping"]
    files_cfg = config["files"]
    id_col = columns_cfg["id"]
    timepoint_col = columns_cfg["timepoint"]
    baseline_value = config["timepoints"]["baseline"]

    merged: pd.DataFrame | None = None
    for file in files_cfg["metadata"] + files_cfg["imaging"]:
        path = env.repo_root / file
        usecols = get_columns_for_file(env, file)

        if usecols is not None:
            available_cols = pd.read_csv(path, nrows=0).columns
            cols_to_read = [col for col in usecols if col in available_cols]
        else:
            cols_to_read = None

        df = pd.read_csv(path, usecols=cols_to_read, engine="python")

        # Handle baseline-only files (e.g., RVI_groups.csv) that lack eventname column
        if timepoint_col not in df.columns:
            df[timepoint_col] = baseline_value
            print(
                f"Added {timepoint_col}={baseline_value} to {file} (baseline-only data)"
            )

        if merged is None:
            merged = df
            continue

        merge_keys = [id_col, timepoint_col]

        if not merge_keys:
            raise ValueError(f"Cannot merge {file}: no overlap on ID/timepoint columns")

        merged = merged.merge(df, on=merge_keys, how="outer")

    if merged is None:
        raise ValueError("No files loaded during merge")

    return merged
