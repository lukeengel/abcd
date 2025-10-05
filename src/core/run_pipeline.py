#!/usr/bin/env python3
"""Run the full ABCD pipeline from command line."""

import argparse
from .config import initialize_notebook
from .preprocessing.pipeline import preprocess_abcd_data
from .harmonize.pipeline import run_harmonization_pipeline
from .tsne.pipeline import run_tsne_analysis
from .pca.pipeline import run_pca_analysis


def main():
    parser = argparse.ArgumentParser(description="Run ABCD pipeline")
    parser.add_argument("--run-name", help="Research run name")
    parser.add_argument("--new-run-id", action="store_true", help="Generate new run ID")

    args = parser.parse_args()

    env = initialize_notebook(run_name=args.run_name, regenerate_run_id=args.new_run_id)

    print("Step 1: Preprocessing...")
    train, val, test = preprocess_abcd_data(env)

    print("Step 2: Harmonization...")
    run_harmonization_pipeline(env)

    print("Step 3: t-SNE Analysis...")
    run_tsne_analysis(env)

    print("Step 4: PCA Analysis...")
    run_pca_analysis(env)

    run_path = f"outputs/{env.configs.run['run_name']}/{env.configs.run['run_id']}"
    print(f"Complete. Results in: {run_path}")


if __name__ == "__main__":
    main()
