"""Visualization utilities for SVM results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import RocCurveDisplay


def plot_confusion_matrix(cm, class_names: list[str], title: str, save_path: Path):
    """Plot and save confusion matrix with counts and percentages."""
    # Create text annotations with both count and percentage
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            labels[i, j] = f"{count}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(model, X, y, title: str, save_path: Path):
    """Plot and save ROC curve (binary classification only)."""
    if len(np.unique(y)) != 2:
        return  # Skip for multiclass

    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cv_scores(cv_results: dict, model_name: str, save_path: Path):
    """Plot cross-validation scores across folds."""
    metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    available_metrics = [m for m in metrics if f"test_{m}" in cv_results]

    if not available_metrics:
        return

    fig, axes = plt.subplots(
        1, len(available_metrics), figsize=(4 * len(available_metrics), 4)
    )
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        scores = cv_results[f"test_{metric}"]
        ax.bar(range(1, len(scores) + 1), scores, alpha=0.7)
        ax.axhline(
            np.mean(scores),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(scores):.3f}",
        )
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{model_name} - {metric.replace('_', ' ').title()}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame, title: str, save_path: Path, top_n: int = 20
):
    """Plot feature importance bar chart."""
    df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(range(len(df)), df["importance"].values, alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.iloc[:, 0].values)  # First column (feature name)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_table(results_dict: dict, save_path: Path):
    """Create comparison table of model performances."""
    df = pd.DataFrame(results_dict).T

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
