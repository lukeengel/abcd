"""Visualization utilities for regression."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from typing import Optional


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
):
    """Brain-behavior scatterplot.

    Creates a scatterplot with:
    - Observed scores (x-axis) vs Predicted scores (y-axis)
    - Identity line (perfect prediction)
    - Line of best fit with 95% CI
    - statistics
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot with better aesthetics
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.4,
        s=30,
        color="steelblue",
        edgecolors="navy",
        linewidth=0.5,
        label="Subjects",
    )

    # Identity line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        lw=2,
        alpha=0.7,
        label="Perfect prediction",
        zorder=10,
    )

    # Line of best fit with confidence interval
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(y_true)
    y_fit = p(x_sorted)
    ax.plot(x_sorted, y_fit, "r-", lw=2.5, label="Best fit", zorder=9)

    # 95% confidence interval for the fit
    from scipy import stats

    predict_error = y_pred - p(y_true)
    degrees_of_freedom = len(y_true) - 2
    residual_std = np.sqrt(np.sum(predict_error**2) / degrees_of_freedom)
    t_val = stats.t.ppf(0.975, degrees_of_freedom)
    ci = (
        t_val
        * residual_std
        * np.sqrt(1 / len(y_true) + (x_sorted - y_true.mean()) ** 2 / np.sum((y_true - y_true.mean()) ** 2))
    )
    ax.fill_between(x_sorted, y_fit - ci, y_fit + ci, color="red", alpha=0.15, label="95% CI")

    # Compute comprehensive metrics
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r, p_val = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)

    # Statistics box with better formatting
    textstr = "\n".join(
        [
            f"R² = {r2:.3f}",
            f"Pearson r = {r:.3f}",
            f"p = {p_val:.4f}" if p_val >= 0.001 else "p < 0.001",
            f"Spearman ρ = {rho:.3f}",
            f"MAE = {mae:.2f}",
            f"RMSE = {rmse:.2f}",
            f"n = {len(y_true):,}",
        ]
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=1.5)
    ax.text(
        0.05,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        family="monospace",
    )

    ax.set_xlabel("Observed Clinical Score", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted Clinical Score", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # Equal aspect ratio for better interpretation
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
):
    """Plot residuals (errors) vs predicted values."""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Residual plot
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color="r", linestyle="--", lw=2)
    ax1.set_xlabel("Predicted Values", fontsize=12)
    ax1.set_ylabel("Residuals (True - Predicted)", fontsize=12)
    ax1.set_title("Residual Plot", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Histogram of residuals
    ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Residuals", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Distribution of Residuals", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)

    # Add stats
    textstr = f"Mean = {residuals.mean():.2f}\nStd = {residuals.std():.2f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax2.text(
        0.70,
        0.95,
        textstr,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    importance_df,
    title: str,
    save_path: Path,
    top_n: int = 20,
):
    """Plot feature importance."""
    # Select top N features
    top_features = importance_df.nlargest(top_n, "importance")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Horizontal bar plot
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features["importance"].values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_coefficients(
    coefficients: np.ndarray,
    feature_names: list[str],
    title: str,
    save_path: Path,
    top_n: int = 30,
):
    """Coefficient plot for linear models.

    Creates a stem/lollipop plot showing:
    - Top N most important coefficients by absolute value
    - Positive (predictive of higher scores) vs negative (protective)
    - Color-coded by sign
    """
    # Create DataFrame and get top features by absolute value
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coef": np.abs(coefficients),
        }
    )
    top_coef = coef_df.nlargest(top_n, "abs_coef").sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))

    # Color by sign
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in top_coef["coefficient"]]

    # Stem plot (lollipop chart)
    y_pos = np.arange(len(top_coef))
    ax.barh(
        y_pos,
        top_coef["coefficient"].values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add vertical line at zero
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_coef["feature"].values, fontsize=9)
    ax.set_xlabel("Regression Coefficient (β)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d62728", alpha=0.7, label="Positive (↑ symptoms)"),
        Patch(facecolor="#1f77b4", alpha=0.7, label="Negative (↓ symptoms)"),
    ]
    ax.legend(handles=legend_elements, loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    save_path: Path,
    title: str = "Brain-Behavior Correlation Matrix",
    max_features: int = 50,
):
    """Correlation heatmap between brain features and clinical subscales.

    Args:
        features_df: DataFrame with brain features (PCA components or ROIs)
        targets_df: DataFrame with clinical subscales
        save_path: Path to save figure
        title: Plot title
        max_features: Maximum number of features to show
    """
    # Limit features if too many
    if features_df.shape[1] > max_features:
        # Select features with highest variance
        feature_vars = features_df.var()
        top_features = feature_vars.nlargest(max_features).index
        features_df = features_df[top_features]

    # Compute correlations
    correlations = pd.DataFrame(index=features_df.columns, columns=targets_df.columns)

    for feat in features_df.columns:
        for target in targets_df.columns:
            r, _ = pearsonr(features_df[feat], targets_df[target])
            correlations.loc[feat, target] = r

    correlations = correlations.astype(float)

    # Create heatmap
    fig, ax = plt.subplots(
        figsize=(
            max(8, len(targets_df.columns) * 1.5),
            max(10, len(features_df.columns) * 0.3),
        )
    )

    sns.heatmap(
        correlations,
        cmap="RdBu_r",
        center=0,
        vmin=-0.3,
        vmax=0.3,
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Pearson r"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel("Clinical Subscales", fontsize=13, fontweight="bold")
    ax.set_ylabel("Brain Features", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_permutation_importance(
    feature_names: list[str],
    importance_mean: np.ndarray,
    importance_std: np.ndarray,
    title: str,
    save_path: Path,
    top_n: int = 20,
):
    """Plot permutation feature importance with error bars.

    Args:
        feature_names: List of feature names
        importance_mean: Mean importance across permutations
        importance_std: Std of importance across permutations
        title: Plot title
        save_path: Path to save figure
        top_n: Number of top features to show
    """
    # Create DataFrame
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance_mean, "std": importance_std})

    # Get top features
    top_imp = imp_df.nlargest(top_n, "importance").sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))

    y_pos = np.arange(len(top_imp))
    ax.barh(
        y_pos,
        top_imp["importance"].values,
        xerr=top_imp["std"].values,
        color="steelblue",
        alpha=0.7,
        edgecolor="navy",
        linewidth=0.5,
        error_kw={"linewidth": 1.5, "ecolor": "black", "capsize": 3},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_imp["feature"].values, fontsize=9)
    ax.set_xlabel("Permutation Importance (decrease in R²)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coefficients: Optional[np.ndarray],
    feature_names: Optional[list[str]],
    title: str,
    save_path: Path,
):
    """Create a 2x2 summary figure.

    Combines:
    1. Brain-behavior scatterplot
    2. Residual plot
    3. Coefficient plot (if available)
    4. Distribution comparison
    """
    if coefficients is not None and feature_names is not None:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # (1) Scatterplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        y_true,
        y_pred,
        alpha=0.4,
        s=25,
        color="steelblue",
        edgecolors="navy",
        linewidth=0.5,
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, alpha=0.7)

    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(y_true)
    ax1.plot(x_sorted, p(x_sorted), "r-", lw=2)

    r2 = r2_score(y_true, y_pred)
    r, p_val = pearsonr(y_true, y_pred)
    textstr = f"R² = {r2:.3f}\nr = {r:.3f}\np = {p_val:.4f}"
    ax1.text(
        0.05,
        0.95,
        textstr,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax1.set_xlabel("Observed Score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Predicted Score", fontsize=12, fontweight="bold")
    ax1.set_title("(A) Brain-Behavior Prediction", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)

    # (2) Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_true - y_pred
    ax2.scatter(
        y_pred,
        residuals,
        alpha=0.4,
        s=25,
        color="coral",
        edgecolors="darkred",
        linewidth=0.5,
    )
    ax2.axhline(y=0, color="black", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted Score", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Residuals", fontsize=12, fontweight="bold")
    ax2.set_title("(B) Residual Analysis", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)

    if coefficients is not None and feature_names is not None:
        # (3) Coefficients
        ax3 = fig.add_subplot(gs[1, 0])
        coef_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coef": np.abs(coefficients),
            }
        )
        top_coef = coef_df.nlargest(15, "abs_coef").sort_values("coefficient")

        colors = ["#d62728" if c > 0 else "#1f77b4" for c in top_coef["coefficient"]]
        y_pos = np.arange(len(top_coef))
        ax3.barh(y_pos, top_coef["coefficient"].values, color=colors, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(top_coef["feature"].values, fontsize=9)
        ax3.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax3.set_xlabel("Coefficient (β)", fontsize=12, fontweight="bold")
        ax3.set_title("(C) Top Feature Coefficients", fontsize=13, fontweight="bold")
        ax3.grid(axis="x", alpha=0.3)

        # (4) Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(
            y_true,
            bins=30,
            alpha=0.5,
            label="Observed",
            density=True,
            color="steelblue",
        )
        ax4.hist(y_pred, bins=30, alpha=0.5, label="Predicted", density=True, color="coral")
        ax4.set_xlabel("Clinical Score", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax4.set_title("(D) Score Distributions", fontsize=13, fontweight="bold")
        ax4.legend(framealpha=0.9)
        ax4.grid(alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
