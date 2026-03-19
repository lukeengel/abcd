"""Column transformations applied during preprocessing."""

from __future__ import annotations

from pathlib import Path

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


def load_family_history(fhx_path: str | Path, scope: str = "psychosis") -> pd.DataFrame:
    """Load and derive family history variables from mh_p_fhx.csv.

    Q8 (visions) items encode psychosis family history:
        a = mother, b = father, c-f = grandparents
        1 = Yes, 0 = No

    Args:
        fhx_path: Path to mh_p_fhx.csv
        scope: 'psychosis' for Q8 only, 'broad' to include Q6-Q7, Q10-Q13

    Returns:
        DataFrame with src_subject_id + derived binary/count variables.
    """
    fhx = pd.read_csv(fhx_path, low_memory=False)

    # Q8: visions (psychosis) — a=mother, b=father, c-f=grandparents
    psychosis_cols = [f"fam_history_q8{s}_visions" for s in "abcdef"]
    parent_cols = [f"fam_history_q8{s}_visions" for s in "ab"]

    # Coerce to numeric (may contain NaN / non-numeric)
    for col in psychosis_cols:
        if col in fhx.columns:
            fhx[col] = pd.to_numeric(fhx[col], errors="coerce")

    present = [c for c in psychosis_cols if c in fhx.columns]
    parent_present = [c for c in parent_cols if c in fhx.columns]

    out = fhx[["src_subject_id", "eventname"]].copy()
    out["fhx_psychosis_any"] = (fhx[present].eq(1).any(axis=1)).astype(int)
    out["fhx_psychosis_parent"] = (fhx[parent_present].eq(1).any(axis=1)).astype(int)
    out["fhx_psychosis_count"] = fhx[present].eq(1).sum(axis=1)

    if scope == "broad":
        broad_items = {
            "depression": "q6",
            "mania": "q7",
            "nerves": "q10",
            "professional": "q11",
            "hospitalized": "q12",
            "suicide": "q13",
        }
        for label, q_prefix in broad_items.items():
            cols = [f"fam_history_{q_prefix}{s}_{label}" for s in "abcdef"]
            for col in cols:
                if col in fhx.columns:
                    fhx[col] = pd.to_numeric(fhx[col], errors="coerce")
            cols_present = [c for c in cols if c in fhx.columns]
            if cols_present:
                out[f"fhx_{label}_any"] = (fhx[cols_present].eq(1).any(axis=1)).astype(int)

    return out
