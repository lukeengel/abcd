"""Harmonization implementation using neuroHarmonize."""

import numpy as np
from neuroHarmonize import (
    harmonizationLearn,
    harmonizationApply,
    saveHarmonizationModel,
)


def harmonize_all_splits(
    env, train_data, train_covars, val_data, val_covars, test_data, test_covars
) -> dict:
    """Train harmonization on train split, apply to val/test."""
    print("Starting harmonization...")
    run_cfg = env.configs.run
    harm_cfg = env.configs.harmonize

    # Create output paths
    model_path = (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
        / "models"
        / "harmonization.pkl"
    )
    harmonized_dir = (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
        / "harmonized"
    )

    # Train harmonization model
    model, train_harmonized = harmonizationLearn(
        train_data,
        train_covars,
        eb=harm_cfg.get("empirical_bayes", True),
        smooth_terms=harm_cfg.get("smooth_terms", []),
        ref_batch=harm_cfg.get("reference_site"),
    )

    # Apply to val/test splits
    val_harmonized = harmonizationApply(val_data, val_covars, model)
    test_harmonized = harmonizationApply(test_data, test_covars, model)

    # Save model and results
    if harm_cfg.get("save_model", True):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove existing model file if it exists
        if model_path.exists():
            model_path.unlink()
        saveHarmonizationModel(model, str(model_path))

    harmonized_dir.mkdir(parents=True, exist_ok=True)
    np.save(harmonized_dir / "train_harmonized.npy", train_harmonized)
    np.save(harmonized_dir / "val_harmonized.npy", val_harmonized)
    np.save(harmonized_dir / "test_harmonized.npy", test_harmonized)

    print(
        f"Complete: Train {train_harmonized.shape}, "
        f"Val {val_harmonized.shape}, Test {test_harmonized.shape}"
    )

    return {
        "train": train_harmonized,
        "val": val_harmonized,
        "test": test_harmonized,
        "model": model,
    }
