#!/usr/bin/env python3
"""Run complete analysis pipeline across all models and generate comparison report."""

import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.core.config import initialize_notebook
from src.core.svm.pipeline import run_svm_pipeline
from src.core.randomforest.pipeline import run_randomforest_pipeline
from src.core.mlp.pipeline import run_mlp_pipeline
from src.core.regression.pipeline import run_regression_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_classification_metrics(results_dict, model_name):
    """Extract key metrics from classification results."""
    metrics = {}

    for task_name, task_results in results_dict.items():
        model_key = model_name.lower()

        if model_key in task_results:
            overall = task_results[model_key].get("overall", {})
            per_fold = task_results[model_key].get("per_fold", {})

            metrics[task_name] = {
                "bal_acc": overall.get("balanced_accuracy", 0),
                "bal_acc_std": per_fold.get("balanced_accuracy_std", 0),
                "roc_auc": overall.get("roc_auc", 0),
                "roc_auc_std": per_fold.get("roc_auc_std", 0),
                "precision": overall.get("precision", 0),
                "recall": overall.get("recall", 0),
            }

    return metrics


def extract_regression_metrics(results_dict):
    """Extract key metrics from regression results."""
    metrics = {}

    for target_name, target_results in results_dict.items():
        for model_name, model_results in target_results.items():
            if model_name == "baseline":
                continue

            if target_name not in metrics:
                metrics[target_name] = {}

            overall = model_results.get("overall", {})

            metrics[target_name][model_name] = {
                "r2": overall.get("r2", 0),
                "mae": overall.get("mae", 0),
                "rmse": overall.get("rmse", 0),
                "pearson_r": overall.get("pearson_r", 0),
            }

    return metrics


def create_comparison_chart(all_results, env, output_dir):
    """Create comprehensive comparison chart across all methods."""
    logger.info("Creating comparison chart...")

    seed = env.configs.run["seed"]
    run_name = env.configs.run["run_name"]

    # setup plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 12))

    # extract classification metrics
    classification_metrics = {}
    for model in ["svm", "randomforest", "mlp"]:
        if model in all_results and all_results[model]:
            classification_metrics[model] = extract_classification_metrics(all_results[model], model)

    # get all tasks
    all_tasks = set()
    for model_metrics in classification_metrics.values():
        all_tasks.update(model_metrics.keys())
    all_tasks = sorted(all_tasks)

    # create subplots
    n_tasks = len(all_tasks)
    if n_tasks > 0:
        # 1. balanced accuracy comparison
        ax1 = plt.subplot(2, 3, 1)
        plot_data = []
        for task in all_tasks:
            for model, metrics in classification_metrics.items():
                if task in metrics:
                    plot_data.append(
                        {
                            "Task": task.replace("_", " ").title(),
                            "Model": model.upper(),
                            "Balanced Accuracy": metrics[task]["bal_acc"],
                            "std": metrics[task]["bal_acc_std"],
                        }
                    )

        if plot_data:
            import pandas as pd

            df = pd.DataFrame(plot_data)
            x = np.arange(len(all_tasks))
            width = 0.25

            for i, model in enumerate(["svm", "randomforest", "mlp"]):
                model_data = df[df["Model"] == model.upper()]
                if not model_data.empty:
                    values = [
                        model_data[model_data["Task"] == task.replace("_", " ").title()]["Balanced Accuracy"].values[0]
                        if task.replace("_", " ").title() in model_data["Task"].values
                        else 0
                        for task in all_tasks
                    ]
                    stds = [
                        model_data[model_data["Task"] == task.replace("_", " ").title()]["std"].values[0]
                        if task.replace("_", " ").title() in model_data["Task"].values
                        else 0
                        for task in all_tasks
                    ]
                    ax1.bar(
                        x + i * width,
                        values,
                        width,
                        label=model.upper(),
                        alpha=0.8,
                        yerr=stds,
                        capsize=3,
                    )

            ax1.set_xlabel("Task")
            ax1.set_ylabel("Balanced Accuracy")
            ax1.set_title("Balanced Accuracy by Model and Task", fontweight="bold")
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(
                [t.replace("_", " ").title()[:15] for t in all_tasks],
                rotation=45,
                ha="right",
            )
            ax1.legend()
            ax1.set_ylim([0, 1])
            ax1.axhline(0.5, color="red", linestyle="--", alpha=0.3, linewidth=1)

        # 2. roc auc comparison
        ax2 = plt.subplot(2, 3, 2)
        plot_data = []
        for task in all_tasks:
            for model, metrics in classification_metrics.items():
                if task in metrics:
                    plot_data.append(
                        {
                            "Task": task.replace("_", " ").title(),
                            "Model": model.upper(),
                            "ROC-AUC": metrics[task]["roc_auc"],
                            "std": metrics[task]["roc_auc_std"],
                        }
                    )

        if plot_data:
            df = pd.DataFrame(plot_data)
            x = np.arange(len(all_tasks))

            for i, model in enumerate(["svm", "randomforest", "mlp"]):
                model_data = df[df["Model"] == model.upper()]
                if not model_data.empty:
                    values = [
                        model_data[model_data["Task"] == task.replace("_", " ").title()]["ROC-AUC"].values[0]
                        if task.replace("_", " ").title() in model_data["Task"].values
                        else 0
                        for task in all_tasks
                    ]
                    stds = [
                        model_data[model_data["Task"] == task.replace("_", " ").title()]["std"].values[0]
                        if task.replace("_", " ").title() in model_data["Task"].values
                        else 0
                        for task in all_tasks
                    ]
                    ax2.bar(
                        x + i * width,
                        values,
                        width,
                        label=model.upper(),
                        alpha=0.8,
                        yerr=stds,
                        capsize=3,
                    )

            ax2.set_xlabel("Task")
            ax2.set_ylabel("ROC-AUC")
            ax2.set_title("ROC-AUC by Model and Task", fontweight="bold")
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(
                [t.replace("_", " ").title()[:15] for t in all_tasks],
                rotation=45,
                ha="right",
            )
            ax2.legend()
            ax2.set_ylim([0, 1])
            ax2.axhline(0.5, color="red", linestyle="--", alpha=0.3, linewidth=1)

    # 3. precision-recall tradeoff
    ax3 = plt.subplot(2, 3, 3)
    for model, model_metrics in classification_metrics.items():
        precisions = []
        recalls = []
        labels = []
        for task, metrics in model_metrics.items():
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            labels.append(task.replace("_", " ").title()[:10])

        if precisions:
            ax3.scatter(recalls, precisions, label=model.upper(), alpha=0.7, s=100)

    ax3.set_xlabel("Recall (Sensitivity)")
    ax3.set_ylabel("Precision (PPV)")
    ax3.set_title("Precision-Recall Tradeoff", fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])

    # 4. regression metrics (if available)
    if "regression" in all_results and all_results["regression"]:
        reg_metrics = extract_regression_metrics(all_results["regression"])

        ax4 = plt.subplot(2, 3, 4)
        targets = list(reg_metrics.keys())
        models = set()
        for target_metrics in reg_metrics.values():
            models.update(target_metrics.keys())
        models = sorted(models)

        x = np.arange(len(targets))
        width = 0.2

        for i, model in enumerate(models):
            r2_values = [reg_metrics[t].get(model, {}).get("r2", 0) for t in targets]
            ax4.bar(x + i * width, r2_values, width, label=model.upper(), alpha=0.8)

        ax4.set_xlabel("Target")
        ax4.set_ylabel("R²")
        ax4.set_title("Regression R² by Target and Model", fontweight="bold")
        ax4.set_xticks(x + width * (len(models) - 1) / 2)
        ax4.set_xticklabels([t.replace("_", " ").title()[:10] for t in targets], rotation=45, ha="right")
        ax4.legend()
        ax4.axhline(0, color="red", linestyle="--", alpha=0.3, linewidth=1)

    # 5. model comparison summary table
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis("off")

    # prepare summary table data
    table_data = []
    table_data.append(["Model", "Avg Bal Acc", "Avg ROC-AUC", "Tasks"])

    for model, model_metrics in classification_metrics.items():
        if model_metrics:
            bal_accs = [m["bal_acc"] for m in model_metrics.values()]
            roc_aucs = [m["roc_auc"] for m in model_metrics.values()]

            avg_bal_acc = np.mean(bal_accs) if bal_accs else 0
            avg_roc_auc = np.mean(roc_aucs) if roc_aucs else 0

            table_data.append(
                [
                    model.upper(),
                    f"{avg_bal_acc:.3f}",
                    f"{avg_roc_auc:.3f}",
                    str(len(model_metrics)),
                ]
            )

    if len(table_data) > 1:
        table = ax5.table(
            cellText=table_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.25, 0.25, 0.25, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # header row styling
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")

    ax5.set_title("Model Performance Summary", fontweight="bold", pad=20)

    plt.suptitle(
        f"ABCD Pipeline: {run_name.upper()} - Comprehensive Results (Seed {seed})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # save figure
    output_path = output_dir / f"comparison_report_{run_name}_seed{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Comparison chart saved to {output_path}")

    return output_path


def save_summary_report(all_results, env, output_dir):
    """Save comprehensive JSON summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_name": env.configs.run["run_name"],
        "run_id": env.configs.run["run_id"],
        "seed": env.configs.run["seed"],
        "results": {},
    }

    # add classification results
    for model in ["svm", "randomforest", "mlp"]:
        if model in all_results and all_results[model]:
            summary["results"][model] = extract_classification_metrics(all_results[model], model)

    # add regression results
    if "regression" in all_results and all_results["regression"]:
        summary["results"]["regression"] = extract_regression_metrics(all_results["regression"])

    output_path = output_dir / f"summary_report_{summary['run_name']}_seed{summary['seed']}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary report saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svm", "randomforest", "mlp"],
        choices=["svm", "randomforest", "mlp", "regression"],
        help="Models to run",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ABCD Complete Pipeline Runner")
    logger.info("=" * 60)

    # initialize environment
    logger.info("Initializing configuration...")
    env = initialize_notebook()

    seed = env.configs.run["seed"]
    run_name = env.configs.run["run_name"]
    logger.info(f"Research Question: {run_name.upper()}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Models to run: {', '.join(args.models)}")

    # create output directory
    output_dir = env.repo_root / "outputs" / run_name / env.configs.run["run_id"] / f"seed_{seed}" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # run classification models
    if "svm" in args.models:
        logger.info("\n" + "=" * 60)
        logger.info("Running SVM Pipeline")
        logger.info("=" * 60)
        all_results["svm"] = run_svm_pipeline(env, use_wandb=args.wandb)

    if "randomforest" in args.models:
        logger.info("\n" + "=" * 60)
        logger.info("Running Random Forest Pipeline")
        logger.info("=" * 60)
        all_results["randomforest"] = run_randomforest_pipeline(env, use_wandb=args.wandb)

    if "mlp" in args.models:
        logger.info("\n" + "=" * 60)
        logger.info("Running MLP Pipeline")
        logger.info("=" * 60)
        all_results["mlp"] = run_mlp_pipeline(env, use_wandb=args.wandb)

    if "regression" in args.models:
        logger.info("\n" + "=" * 60)
        logger.info("Running Regression Pipeline")
        logger.info("=" * 60)
        all_results["regression"] = run_regression_pipeline(env)

    # create comparison chart
    logger.info("\n" + "=" * 60)
    logger.info("Generating Comparison Report")
    logger.info("=" * 60)
    chart_path = create_comparison_chart(all_results, env, output_dir)
    summary_path = save_summary_report(all_results, env, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Comparison chart: {chart_path}")
    logger.info(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
