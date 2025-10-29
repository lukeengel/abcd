"""Column transformations applied during preprocessing."""

from __future__ import annotations

import pandas as pd


def recode(env, df: pd.DataFrame) -> pd.DataFrame:
    """Apply recodings defined in the config."""

    config = env.configs.data
    column_map = config["columns"]["mapping"]

    for mapping_cfg in config["derived"].values():
        if "map" not in mapping_cfg:
            continue

        source_key = mapping_cfg["source"]
        # Check if source is a mapping key or a direct column name
        source_col = column_map.get(source_key, source_key)
        output_col = mapping_cfg["output"]
        value_map = mapping_cfg["map"]

        if source_col in df.columns:
            df.loc[:, output_col] = df[source_col].map(value_map)

    return df


def binning(env, df: pd.DataFrame) -> pd.DataFrame:
    """Create bins from configured thresholds."""

    config = env.configs.data
    column_map = config["columns"]["mapping"]

    for binning_cfg in config["derived"].values():
        if "thresholds" not in binning_cfg:
            continue

        score_col = column_map[binning_cfg["source"]]
        if score_col not in df.columns:
            continue

        thresholds = binning_cfg["thresholds"]
        labels = binning_cfg["labels"]

        threshold_values = sorted(thresholds.values())
        bins = [-float("inf")] + threshold_values + [float("inf")]
        label_list = list(labels.values())

        df.loc[:, binning_cfg["output"]] = pd.cut(
            df[score_col],
            bins=bins,
            labels=label_list,
            include_lowest=True,
        )

    return df


def create_comorbid_group(df: pd.DataFrame) -> pd.DataFrame:
    """Create comorbid group: Clinical/Subclinical/Control based on anxiety OR psychosis."""
    if "anx_group" not in df.columns or "psych_group" not in df.columns:
        return df

    df = df.copy()

    # Clinical: Clinical in anxiety OR psychosis (includes comorbid clinical)
    is_clinical = (df["anx_group"] == "Clinical") | (df["psych_group"] == "Clinical")

    # Subclinical: Subclinical in anxiety OR psychosis, but NOT clinical
    is_subclinical = (
        (df["anx_group"] == "Subclinical") | (df["psych_group"] == "Subclinical")
    ) & ~is_clinical

    # Control: Control in BOTH anxiety AND psychosis
    is_control = (df["anx_group"] == "Control") & (df["psych_group"] == "Control")

    df.loc[:, "comorbid_group"] = None
    df.loc[is_control, "comorbid_group"] = "Control"
    df.loc[is_subclinical, "comorbid_group"] = "Subclinical"
    df.loc[is_clinical, "comorbid_group"] = "Clinical"

    return df
