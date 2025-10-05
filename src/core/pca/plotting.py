"""PCA plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_scree(
    pca: PCA,
    plots_dir: Path,
    n_components_config: float | int,
) -> None:
    """Plot scree plot showing explained variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)

    # Individual variance explained
    ax1.bar(
        components,
        pca.explained_variance_ratio_ * 100,
        alpha=0.7,
        color="#4A90E2",
    )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Variance Explained by Each Component", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Cumulative variance explained
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(components, cumsum, marker="o", linewidth=2, color="#4A90E2")
    ax2.axhline(y=95, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(y=90, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance Explained", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"PCA Scree Plot (n_components={n_components_config})",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_scree.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary
    var_95 = np.argmax(cumsum >= 95) + 1
    var_90 = np.argmax(cumsum >= 90) + 1
    print(f"Components for 90% variance: {var_90}")
    print(f"Components for 95% variance: {var_95}")
    print(f"Total variance explained: {cumsum[-1]:.2f}%")


def plot_pca_comparison(
    pca_data: dict,
    metadata: dict,
    pc_x: int,
    pc_y: int,
    grouping: str,
    title: str,
    colors: dict,
    filename: str,
    plots_dir: Path,
    plot_config: dict,
) -> None:
    """Plot PCA components colored by grouping variable."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    splits = ["train", "val", "test"]

    for i, (split, ax) in enumerate(zip(splits, axes)):
        groups = metadata[split][grouping]
        embedding = pca_data[split]

        # Auto-detect baseline group (most common)
        if grouping == "research_groups":
            unique_groups, counts = np.unique(
                groups[~pd.isna(groups)], return_counts=True
            )
            baseline_group = unique_groups[np.argmax(counts)]
        else:
            baseline_group = None

        for group in np.unique(groups):
            if pd.isna(group):
                continue
            mask = groups == group
            color = colors.get(group, "#666666")

            if grouping == "research_groups":
                is_baseline = group == baseline_group
                alpha = (
                    plot_config["alpha"]["control"]
                    if is_baseline
                    else plot_config["alpha"]["highlighted"]
                )
                size = (
                    plot_config["size"]["control"]
                    if is_baseline
                    else plot_config["size"]["highlighted"]
                )
                edge = (
                    plot_config["edge"]["control"]
                    if is_baseline
                    else plot_config["edge"]["highlighted"]
                )
                linewidth = (
                    plot_config["linewidth"]["control"]
                    if is_baseline
                    else plot_config["linewidth"]["highlighted"]
                )
            else:
                alpha = 0.7
                size = 8
                edge = None
                linewidth = 0

            ax.scatter(
                embedding[mask, pc_x - 1],
                embedding[mask, pc_y - 1],
                c=color,
                alpha=alpha,
                s=size,
                label=group,
                edgecolors=edge,
                linewidths=linewidth,
            )

        ax.set_title(f"{split.title()} (n = {len(embedding):,})", fontweight="bold")
        ax.set_xlabel(f"PC{pc_x}")
        if i == 0:
            ax.set_ylabel(f"PC{pc_y}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i == 2:  # Only add legend to last plot
            handles = []
            for group in np.unique(groups):
                if not pd.isna(group):
                    color = colors.get(group, "#666666")
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=10,
                            label=group,
                            markeredgecolor="black",
                            markeredgewidth=0.5,
                        )
                    )

            ax.legend(
                handles=handles,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=12,
                markerscale=1.2,
                frameon=False,
            )

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.show()
