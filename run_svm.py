#!/usr/bin/env python3
"""Run SVM classification pipeline with optional W&B sweep support."""

import argparse
import logging
from src.core.config import initialize_notebook
from src.core.svm.pipeline import run_svm_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--sweep", action="store_true", help="Run as W&B sweep agent")
    parser.add_argument("--test", action="store_true", help="Test mode: run only first task")
    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help="Task index to run in test mode (default: 0)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SVM Classification Pipeline")
    logger.info("=" * 60)

    logger.info("Initializing configuration...")
    env = initialize_notebook()

    logger.info(f"Research Question: {env.configs.run['run_name'].upper()}")
    logger.info(f"Seed: {env.configs.run['seed']}")
    logger.info(f"SVM Kernel: {env.configs.svm['model']['kernel']}")
    logger.info(f"C: {env.configs.svm['model']['C']}")
    logger.info(f"CV Folds: {env.configs.svm['cv']['n_outer_splits']}")

    use_wandb = args.wandb or args.sweep

    if use_wandb:
        logger.info("Initializing Weights & Biases...")
        import wandb

        if args.sweep:
            logger.info("Running in sweep mode...")
            wandb.init()
            # override config with sweep parameters
            if wandb.config:
                logger.info(f"Sweep hyperparameters: {dict(wandb.config)}")
                for key, value in wandb.config.items():
                    if key in env.configs.svm["model"]:
                        env.configs.svm["model"][key] = value
        else:
            run_name = f"{env.configs.run['run_name']}-{env.configs.run['run_id']}"
            logger.info("W&B Project: abcd-svm")
            logger.info(f"W&B Run: {run_name}")
            wandb.init(
                project="abcd-svm",
                name=run_name,
                config={
                    "run_name": env.configs.run["run_name"],
                    "seed": env.configs.run["seed"],
                    **env.configs.svm["model"],
                },
            )

    logger.info("=" * 60)
    logger.info("Starting SVM Pipeline")
    if args.test:
        logger.info("TEST MODE: Running only first task")
    if args.sweep:
        logger.info("SWEEP MODE: Metrics only, no plots")
    logger.info("=" * 60)

    # limit to specific task in test mode
    if args.test:
        original_tasks = env.configs.svm["tasks"]
        task_idx = args.task
        if task_idx >= len(original_tasks):
            logger.error(f"Task index {task_idx} out of range (0-{len(original_tasks)-1})")
            logger.info("Available tasks:")
            for i, t in enumerate(original_tasks):
                logger.info(f"  {i}: {t['name']}")
            return
        env.configs.svm["tasks"] = [original_tasks[task_idx]]
        logger.info(f"Testing with task {task_idx}: {original_tasks[task_idx]['name']}")

    run_svm_pipeline(env, use_wandb=use_wandb, sweep_mode=args.sweep)

    if use_wandb:
        logger.info("Finalizing W&B run...")
        wandb.finish()

    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
