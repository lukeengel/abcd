"""Brain feature name mapping from ABCD data dictionary."""

import pandas as pd


def load_data_dictionary(env) -> pd.DataFrame:
    """Load ABCD data dictionary from config path."""
    dict_path_str = env.configs.data["files"].get("data_dictionary")
    if not dict_path_str:
        return None

    dict_path = env.repo_root / dict_path_str
    if not dict_path.exists():
        return None

    return pd.read_csv(dict_path, encoding="utf-8-sig", low_memory=False)


def enrich_brain_regions(brain_regions_df: pd.DataFrame, env) -> pd.DataFrame:
    """Add human-readable labels from data dictionary."""
    data_dict = load_data_dictionary(env)

    if data_dict is None:
        brain_regions_df["region_label"] = brain_regions_df["brain_region"]
        return brain_regions_df[["region_label", "brain_region", "importance"]]

    # Merge with data dictionary
    merged = brain_regions_df.merge(
        data_dict[["var_name", "var_label"]],
        left_on="brain_region",
        right_on="var_name",
        how="left",
    )

    # Use var_label if available, else keep var_name
    merged["region_label"] = merged["var_label"].fillna(merged["brain_region"])

    return merged[["region_label", "brain_region", "importance"]]
