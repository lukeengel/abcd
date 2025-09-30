"""High-level orchestration for ABCD harmonization pipeline."""

from __future__ import annotations

from tqdm import tqdm

from .prepare import prepare_all_splits
from .harmonize import harmonize_all_splits
from .artifacts import save_harmonized_data, save_harmonization_artifacts


def run_harmonization_pipeline(env) -> dict:
    """Main harmonization pipeline for ABCD data."""

    total_steps = 4

    with tqdm(
        total=total_steps,
        desc="Harmonization Pipeline",
        unit="step",
        leave=False,
        ncols=40,
        position=0,
        dynamic_ncols=True,
        file=None,
    ) as pbar:
        pbar.set_description("Step 1/4: Preparing data splits")
        (
            train_data,
            train_covars,
            val_data,
            val_covars,
            test_data,
            test_covars,
        ) = prepare_all_splits(env)
        pbar.update(1)

        pbar.set_description("Step 2/4: Running harmonization")
        harmonized_results = harmonize_all_splits(
            env, train_data, train_covars, val_data, val_covars, test_data, test_covars
        )
        pbar.update(1)

        pbar.set_description("Step 3/4: Saving harmonized data")
        save_harmonized_data(env, harmonized_results)
        pbar.update(1)

        pbar.set_description("Step 4/4: Saving artifacts")
        save_harmonization_artifacts(
            env, harmonized_results, train_covars, val_covars, test_covars
        )
        pbar.update(1)

    print(f"  - Harmonized features: {harmonized_results['train'].shape[1]}")
    print(
        f"  - Train: {harmonized_results['train'].shape[0]}, "
        f"Val: {harmonized_results['val'].shape[0]}, "
        f"Test: {harmonized_results['test'].shape[0]}"
    )

    return harmonized_results
