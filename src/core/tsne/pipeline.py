"""t-SNE analysis pipeline integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .embeddings import (
    get_imaging_columns,
    load_or_compute_tsne,
    prepare_metadata,
    save_metadata,
)
from .plotting import plot_comparison, plot_qc_comparison, setup_plotting_style


def run_tsne_analysis(env) -> dict:
    """Run complete t-SNE analysis pipeline."""

    tsne_config = env.configs.tsne
    run_cfg = env.configs.run
    data_dir = env.repo_root / "outputs" / run_cfg["run_name"] / run_cfg["run_id"] / f"seed_{run_cfg['seed']}"

    # Setup plotting style
    setup_plotting_style(tsne_config["figure"])

    total_steps = 6
    with tqdm(
        total=total_steps,
        desc="t-SNE Analysis",
        unit="step",
        leave=False,
        ncols=40,
        position=0,
        dynamic_ncols=True,
        file=None,
    ) as pbar:

        pbar.set_description("Step 1/6: Loading data")
        # Load all datasets
        baseline_preqc = pd.read_parquet(data_dir / "datasets" / "baseline_preqc.parquet")
        train_orig = pd.read_parquet(data_dir / "datasets" / "train.parquet")
        val_orig = pd.read_parquet(data_dir / "datasets" / "val.parquet")
        test_orig = pd.read_parquet(data_dir / "datasets" / "test.parquet")
        all_orig = pd.concat([train_orig, val_orig, test_orig], ignore_index=True)

        # Load harmonized data
        train_harm = np.load(data_dir / "harmonized" / "train_harmonized.npy")
        val_harm = np.load(data_dir / "harmonized" / "val_harmonized.npy")
        test_harm = np.load(data_dir / "harmonized" / "test_harmonized.npy")
        all_harm = np.vstack([train_harm, val_harm, test_harm])

        # Get imaging columns
        imaging_cols = get_imaging_columns(all_orig, tsne_config["imaging_prefixes"])
        pbar.update(1)

        pbar.set_description("Step 2/6: Computing embeddings")
        # Setup directories
        embeddings_dir = data_dir / "tsne_embeddings"
        plots_dir = embeddings_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Compute all embeddings
        embeddings = {
            'preqc': load_or_compute_tsne(
                baseline_preqc[imaging_cols].values, 'preqc', embeddings_dir, tsne_config
            ),
            'postqc': load_or_compute_tsne(
                all_orig[imaging_cols].values, 'postqc', embeddings_dir, tsne_config
            ),
            'harmonized': load_or_compute_tsne(
                all_harm, 'harmonized', embeddings_dir, tsne_config
            )
        }
        pbar.update(1)

        pbar.set_description("Step 3/6: Preparing metadata")
        # Prepare and save metadata
        metadata = prepare_metadata(baseline_preqc, all_orig)
        save_metadata(metadata, embeddings_dir)
        pbar.update(1)

        pbar.set_description("Step 4/6: QC comparison plots")
        # Generate QC comparison plot
        qc_threshold = env.configs.data["qc_thresholds"]["surface_holes_max"]
        plot_qc_comparison(embeddings, metadata, plots_dir, tsne_config["complexity"], qc_threshold)
        pbar.update(1)

        pbar.set_description("Step 5/6: Harmonization plots")
        # Generate harmonization comparison plots
        plot_comparison(
            embeddings, metadata, 'postqc', 'harmonized', 'anxiety',
            'Harmonization Impact on Anxiety Group Clustering',
            tsne_config["colors"]["anxiety"], 'harmonization_anxiety',
            plots_dir, tsne_config["complexity"], tsne_config["plots"]
        )

        plot_comparison(
            embeddings, metadata, 'postqc', 'harmonized', 'scanner',
            'Harmonization Impact on Scanner Effects',
            tsne_config["colors"]["scanner"], 'harmonization_scanner',
            plots_dir, tsne_config["complexity"], tsne_config["plots"]
        )
        pbar.update(1)

        pbar.set_description("Step 6/6: Demographics plots")
        # Generate demographics plots
        plot_comparison(
            embeddings, metadata, 'postqc', 'harmonized', 'age',
            'Age Distribution in t-SNE Space',
            tsne_config["colors"]["age"], 'demographics_age',
            plots_dir, tsne_config["complexity"], tsne_config["plots"]
        )

        plot_comparison(
            embeddings, metadata, 'postqc', 'harmonized', 'sex',
            'Sex Distribution in t-SNE Space',
            tsne_config["colors"]["sex"], 'demographics_sex',
            plots_dir, tsne_config["complexity"], tsne_config["plots"]
        )
        pbar.update(1)

    # Print summary
    print(f"  - Pre-QC samples: {len(baseline_preqc):,}")
    print(f"  - Post-QC samples: {len(all_orig):,}")
    print(f"  - Features analyzed: {len(imaging_cols):,}")
    print(f"  - t-SNE complexity: {tsne_config['complexity']}")

    return {
        'embeddings': embeddings,
        'metadata': metadata,
        'plots_dir': plots_dir,
        'embeddings_dir': embeddings_dir
    }