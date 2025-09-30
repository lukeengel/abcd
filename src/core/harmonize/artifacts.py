"""Artifacts and output management for harmonization."""

import numpy as np
import pandas as pd
import json


def save_harmonized_data(env, harmonized_results):
    """Save harmonized data arrays to run-specific directory."""
    run_cfg = env.configs.run

    # Create run-specific harmonized directory
    harmonized_dir = (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
        / "harmonized"
    )
    harmonized_dir.mkdir(parents=True, exist_ok=True)

    # Save harmonized arrays
    np.save(harmonized_dir / "train_harmonized.npy", harmonized_results["train"])
    np.save(harmonized_dir / "val_harmonized.npy", harmonized_results["val"])
    np.save(harmonized_dir / "test_harmonized.npy", harmonized_results["test"])

    print(f"Harmonized data saved to: {harmonized_dir}")


def save_harmonization_artifacts(
    env, harmonized_results, train_covars, val_covars, test_covars
):
    """Save harmonization artifacts and metadata."""
    run_cfg = env.configs.run

    # Create artifacts directory
    artifacts_dir = (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
        / "artifacts"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save covariate information
    covariates_summary = {
        "train_sites": train_covars["SITE"].value_counts().to_dict(),
        "val_sites": val_covars["SITE"].value_counts().to_dict(),
        "test_sites": test_covars["SITE"].value_counts().to_dict(),
        "total_sites": len(
            pd.concat([train_covars, val_covars, test_covars])["SITE"].unique()
        ),
        "harmonization_config": dict(env.configs.harmonize),
    }

    # Save as JSON
    with open(artifacts_dir / "harmonization_summary.json", "w") as f:
        json.dump(covariates_summary, f, indent=2)

    # Save harmonization statistics
    harmonization_stats = {
        "original_features": harmonized_results["train"].shape[1],
        "train_samples": harmonized_results["train"].shape[0],
        "val_samples": harmonized_results["val"].shape[0],
        "test_samples": harmonized_results["test"].shape[0],
        "train_variance": float(np.var(harmonized_results["train"])),
        "val_variance": float(np.var(harmonized_results["val"])),
        "test_variance": float(np.var(harmonized_results["test"])),
    }

    with open(artifacts_dir / "harmonization_stats.json", "w") as f:
        json.dump(harmonization_stats, f, indent=2)

    print(f"Harmonization artifacts saved to: {artifacts_dir}")


def get_harmonized_data_path(env):
    """Get the path to harmonized data for a given run."""
    run_cfg = env.configs.run
    return (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
        / "harmonized"
    )


def load_harmonized_data(env):
    """Load harmonized data for a given run."""
    harmonized_dir = get_harmonized_data_path(env)

    if not harmonized_dir.exists():
        raise FileNotFoundError(f"No harmonized data found at: {harmonized_dir}")

    train_harm = np.load(harmonized_dir / "train_harmonized.npy")
    val_harm = np.load(harmonized_dir / "val_harmonized.npy")
    test_harm = np.load(harmonized_dir / "test_harmonized.npy")

    return {
        "train": train_harm,
        "val": val_harm,
        "test": test_harm,
        "all": np.vstack([train_harm, val_harm, test_harm]),
    }
