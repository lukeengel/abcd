"""Missing-data utilities used during preprocessing."""

from __future__ import annotations

import pandas as pd


QC_COLUMNS = {"qc_pass", "qc_reason"}


def summarize_missing(env, df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and rates sorted descending."""

    counts = df.isna().sum()
    rates = counts.div(len(df) or 1).fillna(0.0)
    summary = pd.DataFrame({"column": counts.index, "missing": counts, "rate": rates})
    return summary.sort_values(by="rate", ascending=False).reset_index(drop=True)


def missing_rows(
    env,
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return rows containing at least one missing value in columns (all if None)."""

    view = df if columns is None else df[columns]
    mask = view.isna().any(axis=1)
    return df.loc[mask]


def drop_high_missing(
    env,
    df: pd.DataFrame,
    *,
    threshold: float = 0.3,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose missing-rate exceeds threshold."""

    if not columns:
        return df, []

    rates = df[columns].isna().mean()
    to_drop = rates[rates > threshold].index.tolist()
    if not to_drop:
        return df, []
    trimmed = df.drop(columns=to_drop)
    return trimmed, to_drop


def drop_rows_with_missing(
    env,
    df: pd.DataFrame,
    *,
    columns: list[str],
) -> tuple[pd.DataFrame, int]:
    """Drop rows that contain any missing values in the selected columns."""

    if not columns:
        return df, 0

    mask = df[columns].isna().any(axis=1)
    trimmed = df.loc[~mask].copy()
    return trimmed, int(mask.sum())


def _canonical_to_raw(env, keys: list[str]) -> list[str]:
    mapping = env.configs.data["columns"]["mapping"]
    return [mapping[key] for key in keys if key in mapping]


def _feature_family_columns(env, df: pd.DataFrame, family: str) -> list[str]:
    family_cfg = env.configs.data["columns"].get("imaging", {}).get(family, {})
    prefixes = family_cfg.get("prefixes", [])
    if not prefixes:
        return []
    cols = [
        col for col in df.columns if any(col.startswith(prefix) for prefix in prefixes)
    ]
    return cols


def drop_required_metadata(env, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required_keys = env.configs.data.get("missing", {}).get(
        "require_complete_metadata", []
    )
    if not required_keys:
        return df, 0

    required_cols = [
        col for col in _canonical_to_raw(env, required_keys) if col in df.columns
    ]
    if not required_cols:
        return df, 0
    mask = df[required_cols].notna().all(axis=1)
    trimmed = df.loc[mask].copy()
    return trimmed, int((~mask).sum())


def drop_rows_for_families(
    env, df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    families = env.configs.data.get("missing", {}).get("require_complete_families", [])
    dropped: dict[str, int] = {}
    trimmed = df

    family_columns: dict[str, list[str]] = {}

    for family in families:
        family_cols = _feature_family_columns(env, trimmed, family)
        trimmed, removed = drop_rows_with_missing(env, trimmed, columns=family_cols)
        dropped[family] = removed
        family_columns[family] = family_cols

    return trimmed, {"dropped": dropped, "columns": family_columns}


def handle_missing(
    env,
    df: pd.DataFrame,
    *,
    column_threshold: float = 0.3,
    drop_rows: bool = True,
) -> pd.DataFrame:
    """Basic missing-data strategy that preserves QC columns.

    Drops high-missing-rate columns (excluding metadata/QC) and, optionally,
    removes rows that still contain missing values in the remaining feature set.
    """

    config = env.configs.data
    column_map = config["columns"]["mapping"]

    trimmed_df, _ = drop_required_metadata(env, df)
    trimmed_df, _ = drop_rows_for_families(env, trimmed_df)

    protected = set(column_map.values())
    protected.update(config["columns"].get("metadata", []))
    protected.update(config["columns"].get("derived", []))
    protected.update(QC_COLUMNS)

    candidate_cols = [col for col in trimmed_df.columns if col not in protected]

    trimmed_df, dropped_cols = drop_high_missing(
        env, trimmed_df, threshold=column_threshold, columns=candidate_cols
    )

    if not drop_rows:
        return trimmed_df

    remaining_cols = [col for col in candidate_cols if col not in dropped_cols]
    trimmed_df, _ = drop_rows_with_missing(env, trimmed_df, columns=remaining_cols)
    return trimmed_df
