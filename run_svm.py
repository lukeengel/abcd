#!/usr/bin/env python3
"""Run SVM classification pipeline with optional W&B sweep support."""

import argparse
from core.config import initialize_notebook
from core.svm.pipeline import run_svm_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--sweep", action="store_true", help="Run as W&B sweep agent")
    parser.add_argument(
        "--test", action="store_true", help="Test mode: run only first task"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SVM Classification Pipeline")
    print("=" * 60)

    print("\nInitializing configuration...")
    env = initialize_notebook()

    print(f"Research Question: {env.configs.run['run_name'].upper()}")
    print(f"Seed: {env.configs.run['seed']}")
    print(f"SVM Kernel: {env.configs.svm['model']['kernel']}")
    print(f"C: {env.configs.svm['model']['C']}")
    print(f"CV Folds: {env.configs.svm['cv']['n_splits']}")

    use_wandb = args.wandb or args.sweep

    if use_wandb:
        print("\nInitializing Weights & Biases...")
        import wandb

        if args.sweep:
            print("Running in sweep mode...")
            wandb.init()
            # Override config with sweep parameters
            if wandb.config:
                print(f"Sweep hyperparameters: {dict(wandb.config)}")
                for key, value in wandb.config.items():
                    if key in env.configs.svm["model"]:
                        env.configs.svm["model"][key] = value
        else:
            run_name = f"{env.configs.run['run_name']}-{env.configs.run['run_id']}"
            print("W&B Project: abcd-svm")
            print(f"W&B Run: {run_name}")
            wandb.init(
                project="abcd-svm",
                name=run_name,
                config={
                    "run_name": env.configs.run["run_name"],
                    "seed": env.configs.run["seed"],
                    **env.configs.svm["model"],
                },
            )

    print("\n" + "=" * 60)
    print("Starting SVM Pipeline")
    if args.test:
        print("TEST MODE: Running only first task")
    if args.sweep:
        print("SWEEP MODE: Metrics only, no plots")
    print("=" * 60 + "\n")

    # Limit to first task in test mode
    if args.test:
        original_tasks = env.configs.svm["tasks"]
        env.configs.svm["tasks"] = [original_tasks[0]]
        print(f"Testing with task: {original_tasks[0]['name']}\n")

    run_svm_pipeline(env, use_wandb=use_wandb, sweep_mode=args.sweep)

    if use_wandb:
        print("\nFinalizing W&B run...")
        wandb.finish()

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
