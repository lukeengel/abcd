"""Data preparation for harmonization."""

from __future__ import annotations

import pandas as pd
import numpy as np


def prepare_all_splits(
    env,
) -> tuple[
    np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame
]:
    """Prepare all splits for harmonization."""
    run_cfg = env.configs.run
    harm_cfg = env.configs.harmonize

    # Build output directory path
    output_dir = (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
        / "datasets"
    )

    # Get imaging prefixes once (remove duplicates)
    imaging_prefixes = []
    for modality_cfg in env.configs.data["columns"]["imaging"].values():
        if "prefixes" in modality_cfg:
            imaging_prefixes.extend(modality_cfg["prefixes"])
    imaging_prefixes = list(set(imaging_prefixes))

    # Get covariate columns once
    site_col = harm_cfg["site_column"]
    covariate_cols = [site_col] + harm_cfg["covariates"]

    # Process all splits
    results = []
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(output_dir / f"{split}.parquet")

        # Extract imaging data (no metadata)
        imaging_cols = [
            col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in imaging_prefixes)
        ]
        data_matrix = df[imaging_cols].values

        # Remove zero variance features (only check on first split)
        if split == "train":
            global valid_features
            feature_vars = np.var(data_matrix, axis=0)
            valid_features = (
                feature_vars > 1e-10
            )  # Keep features with non-zero variance
            n_removed = np.sum(~valid_features)
            if n_removed > 0:
                print(f"Removing {n_removed} zero/near-zero variance features")

        # Apply feature filter to all splits
        data_matrix = data_matrix[:, valid_features]

        # Extract covariates
        covars = df[covariate_cols].copy()
        covars = covars.rename(columns={site_col: "SITE"})

        results.extend([data_matrix, covars])

    return tuple(results)
