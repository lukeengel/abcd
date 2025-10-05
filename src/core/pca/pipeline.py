"""PCA analysis pipeline integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..tsne.embeddings import get_imaging_columns, save_metadata
from .embeddings import load_or_fit_pca, transform_and_save
from .plotting import plot_pca_comparison, plot_scree, setup_plotting_style


def run_pca_analysis(env) -> dict:
    """Run complete PCA analysis pipeline."""

    pca_config = env.configs.pca
    run_cfg = env.configs.run
    data_dir = (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
    )

    # Setup plotting style
    setup_plotting_style(pca_config["figure"])

    total_steps = 5
    with tqdm(
        total=total_steps,
        desc="PCA Analysis",
        unit="step",
        leave=False,
        ncols=40,
        position=0,
        dynamic_ncols=True,
        file=None,
    ) as pbar:
        pbar.set_description("Step 1/5: Loading data")
        # Load original datasets for metadata
        train_orig = pd.read_parquet(data_dir / "datasets" / "train.parquet")
        val_orig = pd.read_parquet(data_dir / "datasets" / "val.parquet")
        test_orig = pd.read_parquet(data_dir / "datasets" / "test.parquet")

        # Load harmonized data for PCA
        train_harm = np.load(data_dir / "harmonized" / "train_harmonized.npy")
        val_harm = np.load(data_dir / "harmonized" / "val_harmonized.npy")
        test_harm = np.load(data_dir / "harmonized" / "test_harmonized.npy")

        # Get imaging columns for summary stats
        imaging_cols = get_imaging_columns(train_orig, pca_config["imaging_prefixes"])
        pbar.update(1)

        pbar.set_description("Step 2/5: Fitting PCA")
        # Setup directories
        pca_dir = data_dir / "pca"
        plots_dir = pca_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Fit PCA on harmonized training data
        seed = run_cfg["seed"]
        pca, scaler = load_or_fit_pca(train_harm, pca_dir, pca_config, seed)
        pbar.update(1)

        pbar.set_description("Step 3/5: Transforming data")
        # Transform all datasets
        pca_data = {
            "train": transform_and_save(train_harm, scaler, pca, "train", pca_dir),
            "val": transform_and_save(val_harm, scaler, pca, "val", pca_dir),
            "test": transform_and_save(test_harm, scaler, pca, "test", pca_dir),
        }
        pbar.update(1)

        pbar.set_description("Step 4/5: Preparing metadata")
        # Prepare metadata
        research_question = run_cfg["run_name"]

        def extract_metadata(df: pd.DataFrame) -> dict:
            return {
                "research_groups": df[
                    env.configs.data["columns"]["mapping"]["research_group"]
                ].values,
            }

        metadata = {
            "train": extract_metadata(train_orig),
            "val": extract_metadata(val_orig),
            "test": extract_metadata(test_orig),
        }
        save_metadata(metadata, pca_dir, research_question)
        pbar.update(1)

        pbar.set_description("Step 5/5: Generating plots")
        # Generate scree plot
        plot_scree(pca, plots_dir, pca_config["n_components"])

        # Generate PCA scatter plots
        plot_pca_comparison(
            pca_data,
            metadata,
            1,
            2,
            "research_groups",
            f"PCA: {research_question.title()} Group Separation (PC1 vs PC2)",
            pca_config["colors"][research_question],
            f"pca_{research_question}_pc1_pc2",
            plots_dir,
            pca_config["plots"],
        )

        plot_pca_comparison(
            pca_data,
            metadata,
            2,
            3,
            "research_groups",
            f"PCA: {research_question.title()} Group Separation (PC2 vs PC3)",
            pca_config["colors"][research_question],
            f"pca_{research_question}_pc2_pc3",
            plots_dir,
            pca_config["plots"],
        )
        pbar.update(1)

    # Print summary
    print(f"  - Training samples: {len(train_orig):,}")
    print(f"  - Validation samples: {len(val_orig):,}")
    print(f"  - Test samples: {len(test_orig):,}")
    print(f"  - Original features: {len(imaging_cols):,}")
    print(f"  - PCA components: {pca.n_components_}")
    print(f"  - Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    return {
        "pca_data": pca_data,
        "metadata": metadata,
        "pca_model": pca,
        "scaler": scaler,
        "plots_dir": plots_dir,
        "pca_dir": pca_dir,
    }
