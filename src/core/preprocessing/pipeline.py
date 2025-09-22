"""High-level orchestration for ABCD preprocessing."""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from .artifacts import (
    save_processed_data,
    save_provenance,
    save_qc_artifacts,
    save_split_map,
)
from .ingest import load_and_merge
from .qc import quality_control
from .splits import create_modeling_splits, timepoint_split
from .transforms import binning, recode
from .missing import handle_missing


def preprocess_abcd_data(env) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main preprocessing pipeline for ABCD baseline data."""

    total_steps = 8

    with tqdm(
        total=total_steps,
        desc="Data Preprocessing",
        unit="step",
        leave=False,
        ncols=40,
        position=0,
        dynamic_ncols=True,
        file=None,
    ) as pbar:
        pbar.set_description("Step 1/8: Loading and merging datasets")
        merged_df = load_and_merge(env)
        pbar.update(1)

        pbar.set_description("Step 2/8: Extracting baseline data")
        baseline_df, _ = timepoint_split(env, merged_df)
        pbar.update(1)

        pbar.set_description("Step 3/8: Recoding variables")
        recoded_df = recode(env, baseline_df)
        pbar.update(1)

        pbar.set_description("Step 4/8: Binning variables")
        binned_df = binning(env, recoded_df)
        pbar.update(1)

        pbar.set_description("Step 5/8: Running quality control")
        qc_augmented_df, qc_mask = quality_control(env, binned_df, copy=True)
        pbar.update(1)

        pbar.set_description("Step 6/8: Handling missing values")
        clean_pre_qc = handle_missing(env, qc_augmented_df, drop_rows=True)
        clean_df = clean_pre_qc[clean_pre_qc["qc_pass"]].copy()
        pbar.update(1)

        pbar.set_description("Step 7/8: Creating splits")
        train, val, test, split_map = create_modeling_splits(env, clean_df)
        pbar.update(1)

        pbar.set_description("Step 8/8: Saving data")
        save_processed_data(
            env,
            baseline=clean_df,
            baseline_preqc=clean_pre_qc,
            train=train,
            val=val,
            test=test,
        )
        save_qc_artifacts(env, merged_df, qc_mask)
        save_split_map(env, split_map)
        save_provenance(env, qc_mask, split_map)
        pbar.update(1)
    qc_pass_count = (
        int(clean_df["qc_pass"].sum()) if "qc_pass" in clean_df else len(clean_df)
    )
    print(f"  - QC-pass baseline participants: {qc_pass_count}")
    print(f"  - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    return train, val, test
