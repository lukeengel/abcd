"""Data ingestion helpers for preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def get_columns_for_file(env, filepath: str | Path) -> list[str] | None:
    """Return columns to load for a file based on config."""

    config = env.configs.data
    files_cfg = config["files"]
    str_path = str(filepath)

    metadata_files: Sequence[str] = files_cfg.get("metadata", [])
    imaging_files: Sequence[str] = files_cfg.get("imaging", [])

    if str_path in metadata_files:
        cols = [value for value in config["columns"].values() if isinstance(value, str)]
        unique_cols = list(dict.fromkeys(cols))
        return unique_cols

    if str_path in imaging_files:
        return None

    raise ValueError(f"File {filepath} not found in config")


def load_and_merge(env) -> pd.DataFrame:
    """Load and outer-merge metadata and imaging CSVs."""

    config = env.configs.data
    files_cfg = config["files"]
    id_col = config["columns"]["id"]
    timepoint_col = config["columns"]["timepoint"]

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
