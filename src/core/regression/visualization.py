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
    residualized: bool = False,
):
    """Brain-behavior scatterplot.

    Creates a scatterplot with:
    - Observed scores (x-axis) vs Predicted scores (y-axis)
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
    r, p_pearson = pearsonr(y_true, y_pred)
    rho, p_spearman = spearmanr(y_true, y_pred)

    # Format p-values
    def format_p(p):
        if p < 0.001:
            return "p < 0.001"
        else:
            return f"p = {p:.4f}"

    # Statistics box with better formatting
    textstr = "\n".join(
        [
            f"R² = {r2:.3f}",
            f"Pearson r = {r:.3f} ({format_p(p_pearson)})",
            f"Spearman ρ = {rho:.3f} ({format_p(p_spearman)})",
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

    # Set equal axis limits so the square figure has matching scales
    all_vals = np.concatenate([y_true, y_pred])
    margin = (all_vals.max() - all_vals.min()) * 0.05
    axis_min = all_vals.min() - margin
    axis_max = all_vals.max() + margin
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    score_label = "Residualized Score" if residualized else "Clinical Score"
    ax.set_xlabel(f"Observed {score_label}", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"Predicted {score_label}", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

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

    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(y_true)
    ax1.plot(x_sorted, p(x_sorted), "r-", lw=2)

    r2 = r2_score(y_true, y_pred)
    r, p_pearson = pearsonr(y_true, y_pred)
    rho, p_spearman = spearmanr(y_true, y_pred)
    p_str = "p < .001" if p_pearson < 0.001 else f"p = {p_pearson:.3f}"
    sp_str = "p < .001" if p_spearman < 0.001 else f"p = {p_spearman:.3f}"
    textstr = f"R² = {r2:.3f}\nr = {r:.3f} ({p_str})\nρ = {rho:.3f} ({sp_str})"
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


# ---------------------------------------------------------------------------
# Asymmetry visualization functions (NB11)
# ---------------------------------------------------------------------------

def plot_asymmetry_scatter(
    X_harm: np.ndarray,
    y: np.ndarray,
    bilateral_pairs: list,
    valid_cols: list,
    ai_feat: str = "pallidum_AI",
    save_path=None,
):
    """Annotated scatter: AI feature vs psychosis severity with regression line.

    Args:
        X_harm: Harmonized feature matrix.
        y: Target array.
        bilateral_pairs: List of (name, lcol, rcol).
        valid_cols: Feature column names for X_harm.
        ai_feat: Which AI feature to plot (e.g. 'pallidum_AI').
        save_path: Optional path to save figure.
    """
    from .univariate import compute_asymmetry_features
    from scipy.stats import pearsonr, linregress

    asym = compute_asymmetry_features(X_harm, valid_cols, bilateral_pairs)
    if ai_feat not in asym:
        raise KeyError(f"Feature '{ai_feat}' not in computed asymmetry dict. "
                       f"Available: {list(asym.keys())}")

    ai = asym[ai_feat]
    valid = np.isfinite(ai) & np.isfinite(y)
    ai, y_v = ai[valid], y[valid]
    r, p = pearsonr(ai, y_v)
    slope, intercept, *_ = linregress(ai, y_v)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(ai, y_v, c=y_v, cmap="plasma", alpha=0.5, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="PQ-BC Severity")
    xs = np.linspace(ai.min(), ai.max(), 100)
    ax.plot(xs, slope * xs + intercept, "k-", lw=2)
    ax.set_xlabel(f"{ai_feat} (L−R)/(L+R)\n← More Rightward | More Leftward →",
                  fontweight="bold")
    ax.set_ylabel("PQ-BC Severity", fontweight="bold")
    ax.set_title(f"{ai_feat}: r={r:.3f}, p={p:.4f}, n={valid.sum()}", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_asymmetry_tercile(
    X_harm: np.ndarray,
    y: np.ndarray,
    bilateral_pairs: list,
    valid_cols: list,
    ai_feat: str = "pallidum_AI",
    save_path=None,
):
    """Tercile bar plot: mean severity per tercile of AI feature.

    Args same as plot_asymmetry_scatter. save_path: Optional path.
    """
    from .univariate import compute_asymmetry_features

    asym = compute_asymmetry_features(X_harm, valid_cols, bilateral_pairs)
    if ai_feat not in asym:
        raise KeyError(f"Feature '{ai_feat}' not available.")

    ai = asym[ai_feat]
    valid = np.isfinite(ai) & np.isfinite(y)
    ai, y_v = ai[valid], y[valid]
    # Wrap in pd.Series so the result has the .cat accessor
    terciles = pd.qcut(pd.Series(ai), q=3, labels=["Low AI\n(More Rightward)",
                                                     "Medium AI\n(Near-Symmetric)",
                                                     "High AI\n(More Leftward)"])

    group_mean = [y_v[terciles == t].mean() for t in terciles.cat.categories]
    group_sem = [y_v[terciles == t].std() / np.sqrt((terciles == t).sum())
                 for t in terciles.cat.categories]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(terciles.cat.categories, group_mean, yerr=group_sem,
                  capsize=5, color=["#4393c3", "#f7f7f7", "#d6604d"],
                  edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Mean PQ-BC Severity ± SEM", fontweight="bold")
    ax.set_title("Children with More Symmetric Pallidum Have Higher Psychosis Severity",
                 fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_group_comparison_dual(
    X_ctrl: np.ndarray,
    y_ctrl: np.ndarray,
    X_high: np.ndarray,
    y_high: np.ndarray,
    bilateral_pairs: list,
    valid_cols: list,
    ai_feat: str = "pallidum_AI",
    save_path=None,
):
    """Dual panel: (A) violin/box of AI in controls vs high-severity; (B) scatter within high.

    Args:
        X_ctrl, y_ctrl: Control group harmonized features and targets.
        X_high, y_high: High-severity group harmonized features and targets.
        bilateral_pairs, valid_cols, ai_feat: Feature selection args.
        save_path: Optional path to save figure.
    """
    from .univariate import compute_asymmetry_features
    from scipy.stats import ttest_ind, pearsonr

    asym_ctrl = compute_asymmetry_features(X_ctrl, valid_cols, bilateral_pairs)
    asym_high = compute_asymmetry_features(X_high, valid_cols, bilateral_pairs)
    ai_c = asym_ctrl.get(ai_feat, np.array([]))
    ai_h = asym_high.get(ai_feat, np.array([]))

    t, p = ttest_ind(ai_c, ai_h) if len(ai_c) > 5 and len(ai_h) > 5 else (np.nan, np.nan)
    r, rp = pearsonr(ai_h, y_high) if len(ai_h) > 5 else (np.nan, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: violin comparison
    data_plot = [ai_c, ai_h]
    vp = ax1.violinplot(data_plot, positions=[0, 1], showmedians=True)
    vp["bodies"][0].set_facecolor("#4393c3")
    vp["bodies"][1].set_facecolor("#d6604d")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"Controls\n(n={len(ai_c)})", f"High Severity\n(n={len(ai_h)})"])
    ax1.set_ylabel(f"{ai_feat}", fontweight="bold")
    ax1.set_title(f"(A) Group Comparison\nt={t:.2f}, p={p:.4f}", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: scatter within high-severity
    ax2.scatter(ai_h, y_high, alpha=0.4, s=20, color="#d6604d", edgecolors="none")
    if len(ai_h) > 5:
        from scipy.stats import linregress
        sl, ic, *_ = linregress(ai_h, y_high)
        xs = np.linspace(ai_h.min(), ai_h.max(), 100)
        ax2.plot(xs, sl * xs + ic, "k-", lw=2)
    ax2.set_xlabel(f"{ai_feat}", fontweight="bold")
    ax2.set_ylabel("PQ-BC Severity", fontweight="bold")
    ax2.set_title(f"(B) Within High-Severity\nr={r:.3f}, p={rp:.4f}", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_brain_asymmetry_schematic(
    means_ctrl: dict,
    means_high: dict,
    save_path=None,
):
    """Simplified coronal brain schematic comparing L/R structure sizes between groups.

    Args:
        means_ctrl: dict with keys 'L' and 'R' for control group mean volumes.
        means_high: dict with keys 'L' and 'R' for high-severity group mean volumes.
        save_path: Optional path to save figure.
    """
    def _draw_brain(ax, mean_L, mean_R, title, color_L, color_R):
        """Draw a single coronal schematic panel."""
        ax.set_aspect("equal")
        # Scale boxes relative to a reference size
        ref = (mean_L + mean_R) / 2
        scale = 0.8 / max(ref, 1e-6)
        h_L = mean_L * scale
        h_R = mean_R * scale

        # Left pallidum (displayed on right side of image — neurological convention)
        ax.add_patch(plt.Rectangle((0.05, 0.4 - h_L / 2), 0.3, h_L,
                                    color=color_L, alpha=0.7))
        # Right pallidum
        ax.add_patch(plt.Rectangle((0.65, 0.4 - h_R / 2), 0.3, h_R,
                                    color=color_R, alpha=0.7))

        ax.text(0.2, 0.1, "L", ha="center", fontsize=12, fontweight="bold", color=color_L)
        ax.text(0.8, 0.1, "R", ha="center", fontsize=12, fontweight="bold", color=color_R)
        ai = (mean_L - mean_R) / (mean_L + mean_R) if (mean_L + mean_R) > 0 else 0
        ax.set_title(f"{title}\nAI={ai:.3f}", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.8)
        ax.axis("off")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    _draw_brain(ax1, means_ctrl["L"], means_ctrl["R"],
                "Controls", "#4393c3", "#4393c3")
    _draw_brain(ax2, means_high["L"], means_high["R"],
                "High Severity", "#d6604d", "#d6604d")
    fig.suptitle("Pallidum Asymmetry: Controls vs High Severity",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_longitudinal_trajectories(
    wide_df: pd.DataFrame,
    target_col: str,
    save_path=None,
    baseline_name: str = "baseline",
    followup_name: str = "year2",
    n_sample: int = 100,
):
    """Spaghetti plot of individual PQ-BC trajectories from baseline to year 2.

    Args:
        wide_df: Wide-format longitudinal dataframe with baseline/year2 target columns.
        target_col: Base target column name (e.g. 'pps_y_ss_severity_score').
        save_path: Optional path to save figure.
        baseline_name: Suffix for baseline column.
        followup_name: Suffix for follow-up column.
        n_sample: Max subjects to plot as individual lines (for readability).
    """
    bl_col = f"{target_col}_{baseline_name}"
    fu_col = f"{target_col}_{followup_name}"

    valid = wide_df[[bl_col, fu_col]].dropna()
    if len(valid) == 0:
        print("No paired subjects with both timepoints.")
        return

    sample = valid.sample(min(n_sample, len(valid)), random_state=42)
    bl = sample[bl_col].values
    fu = sample[fu_col].values

    fig, ax = plt.subplots(figsize=(7, 5))
    for b, f in zip(bl, fu):
        color = "#d6604d" if f > b else "#4393c3"
        ax.plot([0, 1], [b, f], color=color, alpha=0.3, lw=0.8)

    # Means
    ax.plot([0, 1], [valid[bl_col].mean(), valid[fu_col].mean()],
            "k-o", lw=3, ms=8, label="Group mean")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Year 2"])
    ax.set_ylabel("PQ-BC Severity Score", fontweight="bold")
    ax.set_title(f"Longitudinal Trajectories (n={len(valid)}, shown={len(sample)})",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ─── Phase 2: notebook-extracted plot functions ────────────────────────────

def plot_sibling_structure(full_df, env, save_path=None):
    """3-panel: family size dist, sibling/singleton pie, CV fold leakage."""
    import pandas as pd, numpy as np, matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if "rel_family_id" not in full_df.columns:
        print("rel_family_id not found — skipping."); return

    fam = pd.to_numeric(full_df["rel_family_id"], errors="coerce")
    fam_counts = fam.dropna().value_counts()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel A
    ax = axes[0]; sizes = fam_counts.value_counts().sort_index()
    ax.bar(sizes.index, sizes.values, color="steelblue", edgecolor="black", alpha=0.8)
    for x, y in zip(sizes.index, sizes.values): ax.text(x, y+10, str(y), ha="center", fontsize=9)
    ax.set_xticks(sizes.index)
    ax.set_xlabel("Family size", fontweight="bold"); ax.set_ylabel("# families", fontweight="bold")
    ax.set_title("A. Family Size Distribution", fontweight="bold")

    # Panel B
    ax = axes[1]
    n_sib = fam_counts[fam_counts > 1].sum(); n_sin = fam_counts[fam_counts == 1].sum(); n_mis = fam.isna().sum()
    lbls = [f"Siblings\n(n={n_sib})", f"Singletons\n(n={n_sin})"]
    vals = [n_sib, n_sin]
    if n_mis > 0: lbls.append(f"Missing\n(n={n_mis})"); vals.append(n_mis)
    ax.pie(vals, labels=lbls, colors=["#E53935","#2196F3","#BDBDBD"][:len(vals)],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize":9})
    ax.set_title("B. Sibling Status", fontweight="bold")

    # Panel C
    ax = axes[2]
    fgs = pd.to_numeric(full_df["rel_family_id"], errors="coerce").values
    mis = np.isnan(fgs)
    if mis.any(): mx = np.nanmax(fgs) if (~mis).any() else 0; fgs[mis] = np.arange(mx+1, mx+1+mis.sum())
    tc = env.configs.regression["targets"][0]["column"]
    yd = full_df[tc].fillna(0).values
    yb = pd.qcut(yd, q=min(5,max(2,len(yd)//20)), labels=False, duplicates="drop")
    fa_std = np.zeros(len(yd), dtype=int)
    for fi,(_, ti) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(full_df,yb)):
        fa_std[ti] = fi
    fa_grp = np.zeros(len(yd), dtype=int)
    for fi,(_, ti) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(full_df,yb,fgs)):
        fa_grp[ti] = fi
    fid = pd.to_numeric(full_df["rel_family_id"], errors="coerce")
    multi = fid[fid.duplicated(keep=False)].dropna()
    ss = sg = 0
    for f in multi.unique():
        idx = fid[fid==f].index
        if len(set(fa_std[idx]))>1: ss+=1
        if len(set(fa_grp[idx]))>1: sg+=1
    nm = int(multi.nunique())
    bars = ax.bar(["Standard\nStratifiedKFold","Family-Aware\nGroupKFold"],
                  [ss,sg], color=["#E53935","#4CAF50"], edgecolor="black", alpha=0.8)
    for b,v in zip(bars,[ss,sg]): ax.text(b.get_x()+b.get_width()/2, v+2, f"{v} ({v/max(nm,1)*100:.0f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Families split across folds", fontweight="bold")
    ax.set_title(f"C. Sibling Leakage ({nm} multi-subject families)", fontweight="bold")

    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_demographic_effects(results_data, use_binned, pop_label, save_path=None):
    """Age/sex bar charts for demographic effects."""
    import numpy as np, matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(results_data)*0.5+1)))
    y_pos = np.arange(len(results_data))
    names = [r["target"] for r in results_data]

    ax = axes[0]
    rs = [r.get("age_r",0) for r in results_data]
    cs = ["#d62728" if r.get("age_p",1)<0.05 else "lightgray" for r in results_data]
    ax.barh(y_pos, rs, color=cs, alpha=0.8, edgecolor="black", lw=0.5)
    ax.axvline(0, color="black", lw=1); ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Pearson r with Age", fontweight="bold"); ax.set_title("(A) Age Effect", fontweight="bold", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    diffs = [r.get("female_mean",0)-r.get("male_mean",0) for r in results_data]
    cs2 = ["#d62728" if r.get("sex_p",1)<0.05 else "lightgray" for r in results_data]
    ax.barh(y_pos, diffs, color=cs2, alpha=0.8, edgecolor="black", lw=0.5)
    ax.axvline(0, color="black", lw=1); ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Female − Male (mean diff)", fontweight="bold"); ax.set_title("(B) Sex Difference", fontweight="bold", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle(pop_label, fontsize=10, style="italic", y=1.02)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)

    sig_a = sum(1 for r in results_data if r.get("age_p",1)<0.05)
    sig_s = sum(1 for r in results_data if r.get("sex_p",1)<0.05)
    print(f"\nKEY TAKEAWAYS: Age significant {sig_a}/{len(results_data)} | Sex {sig_s}/{len(results_data)}")
    if not use_binned: print("  (effects residualized before regression)")


def plot_sample_distribution(y_all, y, bin_edges, target_name, save_path=None):
    """Full distribution + custom bins bar chart."""
    import numpy as np, matplotlib.pyplot as plt
    if bin_edges is None or len(bin_edges)<2: return
    lo,hi = bin_edges[0],bin_edges[-1]; nb = len(bin_edges)-1
    bi = np.clip(np.digitize(y,bin_edges)-1, 0, nb-1)
    bc = np.bincount(bi, minlength=nb)
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    ax = axes[0]
    ax.hist(y_all,bins=30,color="lightgray",alpha=0.7,edgecolor="black",label="Excluded")
    ax.hist(y,bins=30,color="steelblue",alpha=0.7,edgecolor="black",label="Included")
    ax.axvline(lo,color="red",ls="--",lw=2,label=f"Min={lo}"); ax.axvline(hi,color="red",ls="--",lw=2,label=f"Max={hi}")
    ax.set_xlabel("Target Value",fontweight="bold"); ax.set_ylabel("Frequency",fontweight="bold")
    ax.set_title(f"{target_name} — Full Distribution",fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    bw = [bin_edges[i+1]-bin_edges[i] for i in range(nb)]; bctr = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(nb)]
    mn = bc[bc>0].min() if (bc>0).any() else 0
    ax.bar(bctr,bc,width=bw,color=["red" if c==mn else "orange" for c in bc],alpha=0.7,edgecolor="black")
    if mn>0: ax.axhline(mn,color="red",ls="--",lw=2,label=f"Min={mn}")
    ax.set_xlabel("Bin Center",fontweight="bold"); ax.set_ylabel("Count",fontweight="bold")
    ax.set_title(f"{target_name} — Custom Bins (n={len(y):,})",fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_cv_scatter_and_summary(all_y_true, all_y_pred, coefficients, feature_names,
                                model_name, target_name, save_path=None):
    """Brain-behavior scatter + 4-panel CV summary."""
    import numpy as np, matplotlib.pyplot as plt, pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from scipy import stats as sp_stats
    from sklearn.metrics import r2_score

    rv, pv = pearsonr(all_y_true, all_y_pred); rho,_ = spearmanr(all_y_true, all_y_pred); r2 = r2_score(all_y_true, all_y_pred)
    pstr = "p < 0.001" if pv<0.001 else f"p = {pv:.4f}"
    z = np.polyfit(all_y_true, all_y_pred, 1); pl = np.poly1d(z); xs = np.sort(all_y_true)
    res = all_y_pred - pl(all_y_true); dof = len(all_y_true)-2
    se = np.sqrt(np.sum(res**2)/dof); tcrit = sp_stats.t.ppf(0.975, dof)
    ci = tcrit*se*np.sqrt(1/len(all_y_true)+(xs-all_y_true.mean())**2/np.sum((all_y_true-all_y_true.mean())**2))

    fig1,ax = plt.subplots(figsize=(7,7))
    ax.scatter(all_y_true,all_y_pred,alpha=0.4,s=30,color="steelblue",edgecolors="navy",lw=0.5)
    ax.plot(xs,pl(xs),"r-",lw=2.5,zorder=9)
    ax.fill_between(xs,pl(xs)-ci,pl(xs)+ci,color="red",alpha=0.12)
    ax.text(0.05,0.97,f"r = {rv:.3f} ({pstr})\nR\u00b2 = {r2:.3f}\n\u03c1 = {rho:.3f}\nn = {len(all_y_true)}",
            transform=ax.transAxes,fontsize=11,va="top",bbox=dict(boxstyle="round",facecolor="white",alpha=0.9),family="monospace")
    ax.set_xlabel("Observed (residualized)",fontsize=13,fontweight="bold"); ax.set_ylabel("Predicted",fontsize=13,fontweight="bold")
    ax.set_title(f"{model_name.upper()} \u2014 {target_name}",fontsize=14,fontweight="bold"); ax.grid(alpha=0.3,ls="--")
    plt.tight_layout()
    sp1 = str(save_path).replace(".png","_scatter.png") if save_path else None
    if sp1: fig1.savefig(sp1, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig1)

    fig2,axes = plt.subplots(2,2,figsize=(14,12)); residuals = all_y_true-all_y_pred
    ax=axes[0,0]; ax.scatter(all_y_true,all_y_pred,alpha=0.35,s=20,color="steelblue",edgecolors="navy",lw=0.3)
    ax.plot(xs,pl(xs),"r-",lw=2); ax.text(0.05,0.95,f"r={rv:.3f}\nR\u00b2={r2:.3f}",transform=ax.transAxes,fontsize=10,va="top",bbox=dict(boxstyle="round",facecolor="white",alpha=0.8))
    ax.set_xlabel("Observed",fontsize=11); ax.set_ylabel("Predicted",fontsize=11); ax.set_title("(A) Brain-Behavior Prediction",fontsize=12,fontweight="bold"); ax.grid(alpha=0.25)
    ax=axes[0,1]; ax.scatter(all_y_pred,residuals,alpha=0.35,s=20,color="coral",edgecolors="darkred",lw=0.3)
    ax.axhline(0,color="black",ls="--",lw=1.5); ax.set_xlabel("Predicted",fontsize=11); ax.set_ylabel("Residuals",fontsize=11)
    ax.set_title("(B) Residual Analysis",fontsize=12,fontweight="bold"); ax.grid(alpha=0.25)
    ax=axes[1,0]
    if coefficients is not None and feature_names is not None:
        cd = pd.DataFrame({"feature":feature_names,"coef":coefficients}).sort_values("coef")
        yp = np.arange(len(cd)); cc = ["#d62728" if c>0 else "#1f77b4" for c in cd["coef"]]
        ax.barh(yp,cd["coef"].values,color=cc,alpha=0.75,edgecolor="black",lw=0.3)
        ax.set_yticks(yp); ax.set_yticklabels(cd["feature"].values,fontsize=8); ax.axvline(0,color="black",lw=1)
        ax.set_xlabel("Coefficient (\u03b2)",fontsize=11); ax.set_title("(C) SVR Coefficients",fontsize=12,fontweight="bold")
    else: ax.text(0.5,0.5,"No coefficients",ha="center",va="center",transform=ax.transAxes)
    ax.grid(alpha=0.25)
    ax=axes[1,1]; ax.hist(residuals,bins=30,color="coral",edgecolor="black",alpha=0.75)
    ax.axhline(0,color="black",ls="--",lw=1.5); ax.set_xlabel("Residuals",fontsize=11); ax.set_ylabel("Count",fontsize=11)
    ax.set_title("(D) Residual Distribution",fontsize=12,fontweight="bold"); ax.grid(alpha=0.25)
    plt.suptitle(f"{model_name.upper()} CV Summary \u2014 {target_name}",fontsize=14,fontweight="bold"); plt.tight_layout()
    if save_path: fig2.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig2)


def plot_coefficient_forest(coef_df, target_name, save_path=None):
    """Forest plot: SVR coefficients mean ± SD across folds."""
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig,ax = plt.subplots(figsize=(8, max(5, len(coef_df)*0.4)))
    yp = np.arange(len(coef_df)); cc = ["#d62728" if c>0 else "#1f77b4" for c in coef_df["Coefficient"]]
    std_col = coef_df["Std"].values if "Std" in coef_df.columns else np.zeros(len(coef_df))
    ax.barh(yp, coef_df["Coefficient"].values, xerr=std_col,
            color=cc, alpha=0.75, edgecolor="black", lw=0.4, error_kw={"lw":1.2,"ecolor":"black","capsize":3})
    ax.axvline(0,color="black",lw=1.2); ax.set_yticks(yp); ax.set_yticklabels(coef_df["Feature"].values,fontsize=10)
    ax.set_xlabel("SVR Coefficient (mean \u00b1 SD across folds)",fontsize=12,fontweight="bold")
    ax.set_title(f"SVR Coefficients \u2014 {target_name}\n(5-fold CV averaged)",fontsize=13,fontweight="bold")
    ax.grid(axis="x",alpha=0.3,ls="--")
    ax.legend(handles=[Patch(facecolor="#d62728",alpha=0.75,label="Positive (\u2191 severity)"),
                       Patch(facecolor="#1f77b4",alpha=0.75,label="Negative (\u2193 severity)")],
              loc="lower right",framealpha=0.9,fontsize=9)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_permutation_results(null_rs, real_r, p_emp, feat_importances,
                              n_samples, target_name, save_path=None):
    """Null distribution + permutation feature importance bars."""
    import numpy as np, matplotlib.pyplot as plt, pandas as pd
    np_panels = 2 if feat_importances else 1
    fig,axes = plt.subplots(1, np_panels, figsize=(7*np_panels, 5))
    if np_panels==1: axes=[axes]
    ax=axes[0]
    ax.hist(null_rs, bins=50, alpha=0.7, color="steelblue", edgecolor="white", density=True)
    ax.axvline(real_r, color="red", ls="--", lw=2, label=f"Observed r = {real_r:.3f}")
    pstr = f"p = {p_emp:.4f}" if p_emp>=0.001 else "p < 0.001"
    ax.set_xlabel("Pearson r (permuted)",fontweight="bold"); ax.set_ylabel("Density",fontweight="bold")
    ax.set_title(f"Permutation Test: {target_name}\n(n={n_samples}, {pstr})",fontweight="bold")
    ax.legend(fontsize=10); ax.text(0.95,0.95,pstr,transform=ax.transAxes,ha="right",va="top",fontsize=10,bbox=dict(boxstyle="round,pad=0.3",facecolor="wheat",alpha=0.5))
    if feat_importances and np_panels>1:
        ax=axes[1]; fi_df = pd.DataFrame(feat_importances).sort_values("mean_drop",ascending=True)
        short=[f.replace("smri_vol_scs_","").replace("dmri_dtimd_fiberat_","CST ") for f in fi_df["feature"]]
        cc=["#d62728" if d>0 else "#1f77b4" for d in fi_df["mean_drop"]]
        yp=np.arange(len(fi_df))
        ax.barh(yp,fi_df["mean_drop"].values,xerr=fi_df["std_drop"].values,
                color=cc,alpha=0.75,edgecolor="black",lw=0.4,error_kw={"ecolor":"black","capsize":3})
        ax.axvline(0,color="black",lw=1); ax.set_yticks(yp); ax.set_yticklabels(short,fontsize=9)
        ax.set_xlabel("\u0394r (drop when permuted)",fontweight="bold"); ax.set_title("Permutation Importance",fontweight="bold"); ax.grid(axis="x",alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_univariate_comparison(vol_asym_df, raw_results, ai_results, tot_results,
                                network, n, save_path=None):
    """AI vs Total paired dots + 3-panel correlation bars."""
    import numpy as np, matplotlib.pyplot as plt
    # Figure 1: paired
    fig1,ax = plt.subplots(figsize=(10,6)); structs=vol_asym_df["structure"].values; yp=np.arange(len(structs))
    ax.scatter(vol_asym_df["r_AI"].values,yp,color="#d62728",s=80,edgecolors="black",lw=0.5,zorder=5,label="AI")
    ax.scatter(vol_asym_df["r_total"].values,yp,color="#1f77b4",s=80,edgecolors="black",lw=0.5,zorder=5,label="Total")
    for i in range(len(structs)):
        ax.plot([vol_asym_df["r_AI"].values[i],vol_asym_df["r_total"].values[i]],[yp[i],yp[i]],color="gray",lw=0.8,alpha=0.5)
    for i,(_,row) in enumerate(vol_asym_df.iterrows()):
        if row["p_AI"]<0.05: ax.annotate("*",(row["r_AI"],yp[i]-0.12),fontsize=14,fontweight="bold",color="#d62728",ha="center",va="bottom")
    ax.axvline(0,color="black",lw=1,alpha=0.5); ax.set_yticks(yp); ax.set_yticklabels(structs,fontsize=11)
    ax.set_xlabel("Pearson r with PQ-BC severity",fontsize=12,fontweight="bold")
    ax.set_title(f"AI vs Total Volume \u2014 {network} (n={n})",fontsize=13,fontweight="bold")
    ax.legend(loc="lower right",framealpha=0.9,fontsize=10); ax.grid(axis="x",alpha=0.3,ls="--")
    plt.tight_layout(); sp1=str(save_path).replace(".png","_paired.png") if save_path else None
    if sp1: fig1.savefig(sp1,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig1)
    # Figure 2: 3-panel
    fig2,axes=plt.subplots(1,3,figsize=(16,5))
    for ai,(df_i,ttl,cl) in enumerate([(raw_results,"Raw Features","#2ca02c"),(ai_results,"AI","#d62728"),(tot_results,"Total","#1f77b4")]):
        ax=axes[ai]; ds=df_i.sort_values("r",ascending=True).reset_index(drop=True); yp2=np.arange(len(ds))
        pc="p_fdr_bh" if "p_fdr_bh" in ds.columns else "p"
        bc=[cl if p<0.05 else "#cccccc" for p in ds[pc]]
        ax.barh(yp2,ds["r"].values,color=bc,alpha=0.8,edgecolor="black",lw=0.4)
        short=[f.replace("smri_vol_scs_","").replace("dmri_dtimd_fiberat_","CST ").replace("_AI","").replace("_total","") for f in ds["feature"]]
        ax.set_yticks(yp2); ax.set_yticklabels(short,fontsize=9); ax.axvline(0,color="black",lw=1)
        ax.set_xlabel("Pearson r",fontweight="bold"); ax.set_title(ttl,fontweight="bold",fontsize=12); ax.grid(axis="x",alpha=0.3)
    plt.tight_layout()
    if save_path: fig2.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig2)


def plot_group_cohen_d(results_rep, high_n, low_n, target_name="", save_path=None):
    """Cohen's d bar chart: high vs control severity group comparison."""
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(10,5))
    names=[r["feature"].replace("_AI","") for r in results_rep]; dvals=[r["cohens_d"] for r in results_rep]
    bc=["crimson" if r.get("p_fdr",1)<0.05 else ("salmon" if r.get("p",1)<0.05 else "steelblue") for r in results_rep]
    ax.bar(names,dvals,color=bc,alpha=0.8,edgecolor="black")
    ax.set_ylabel("Cohen's d (High \u2212 Low)", fontweight="bold")
    ax.set_title(f"Asymmetry Group Differences: High (n={high_n}) vs Control (n={low_n})", fontweight="bold",fontsize=13)
    ax.axhline(0,color="black",lw=0.5)
    ax.legend(handles=[plt.Rectangle((0,0),1,1,fc="crimson",ec="black",label="FDR sig"),
                       plt.Rectangle((0,0),1,1,fc="salmon",ec="black",label="p<0.05"),
                       plt.Rectangle((0,0),1,1,fc="steelblue",ec="black",label="n.s.")],fontsize=9)
    plt.xticks(rotation=30,ha="right"); plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_raw_feature_corr(results, target_name="", save_path=None):
    """Horizontal bar chart of raw L/R feature correlations."""
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(10,6))
    names=[r["feature"].replace("smri_vol_scs_","").replace("dmri_dtimd_fiberat_","CST_MD_") for r in results]
    rvals=[r["r"] for r in results]
    bc=["crimson" if r.get("sig_fdr") else ("salmon" if r["p"]<0.05 else "steelblue") for r in results]
    ax.barh(range(len(names)),rvals,color=bc,edgecolor="black",alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=9)
    ax.axvline(0,color="black",lw=1); ax.set_xlabel("Pearson r",fontweight="bold")
    ax.set_title(f"Raw L/R Feature Correlations \u2014 {target_name}\n(CV test-set features)",fontweight="bold",fontsize=12)
    ax.legend(handles=[plt.Rectangle((0,0),1,1,fc="crimson",ec="black",label="FDR sig"),
                       plt.Rectangle((0,0),1,1,fc="salmon",ec="black",label="p<0.05"),
                       plt.Rectangle((0,0),1,1,fc="steelblue",ec="black",label="n.s.")],fontsize=9)
    plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_sex_perm_boot(sex_summary, null_rs_dict, boot_rs_dict, save_path=None):
    """2x2: null distributions and bootstrap CIs per sex."""
    import numpy as np, matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(14,10))
    for si,sex in enumerate(["male","female"]):
        s=sex_summary.get(sex,{}); nr=null_rs_dict.get(sex,[]); br=boot_rs_dict.get(sex,[])
        rr=s.get("r_fixed",s.get("r",0)); pe=s.get("p_emp",1.0); ci=s.get("ci",[0,0]); n=s.get("n",0)
        pstr=f"p={pe:.3f}" if pe>=0.001 else "p<0.001"
        ax=axes[si,0]
        if len(nr): ax.hist(nr,bins=50,alpha=0.7,color="steelblue",edgecolor="white",density=True); ax.axvline(rr,color="red",ls="--",lw=2,label=f"r={rr:.3f}")
        ax.set_xlabel("Pearson r (permuted)"); ax.set_ylabel("Density"); ax.set_title(f"{sex.capitalize()} Null (n={n})",fontweight="bold")
        ax.legend(); ax.text(0.95,0.95,pstr,transform=ax.transAxes,ha="right",va="top",fontsize=10,bbox=dict(boxstyle="round,pad=0.3",facecolor="wheat",alpha=0.5))
        ax=axes[si,1]
        if len(br): ax.hist(br,bins=60,color="coral",edgecolor="white",alpha=0.75,density=True); ax.axvline(s.get("r",0),color="red",ls="--",lw=2,label=f"r={s.get('r',0):.3f}")
        for cv in ci: ax.axvline(cv,color="gray",ls=":",lw=1.5)
        ax.set_xlabel("Pearson r (bootstrap)"); ax.set_ylabel("Density"); ax.set_title(f"{sex.capitalize()} Bootstrap 95% CI: [{ci[0]:.3f},{ci[1]:.3f}]",fontweight="bold")
        ax.legend(fontsize=8)
    plt.suptitle("Sex-Stratified Permutation + Bootstrap CIs",fontsize=14,fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_sex_feat_importance(all_feat_imp, all_ai_results, target_name="", save_path=None):
    """Per-sex feature importance and univariate AI correlations."""
    import numpy as np, matplotlib.pyplot as plt, pandas as pd
    sexes=list(all_feat_imp.keys()); fig,axes=plt.subplots(len(sexes),2,figsize=(14,5*len(sexes)))
    if len(sexes)==1: axes=axes.reshape(1,-1)
    for si,sex in enumerate(sexes):
        fd=pd.DataFrame(all_feat_imp[sex]).sort_values("mean_drop",ascending=True)
        short=[f.replace("smri_vol_scs_","").replace("dmri_dtimd_fiberat_","CST ") for f in fd["feature"]]
        cc=["#d62728" if d>0 else "#1f77b4" for d in fd["mean_drop"]]; yp=np.arange(len(fd))
        ax=axes[si,0]
        ax.barh(yp,fd["mean_drop"].values,xerr=fd["std_drop"].values,color=cc,alpha=0.75,edgecolor="black",lw=0.4,error_kw={"ecolor":"black","capsize":3})
        ax.axvline(0,color="black",lw=1); ax.set_yticks(yp); ax.set_yticklabels(short,fontsize=9)
        ax.set_xlabel("\u0394r (permutation importance)",fontweight="bold"); ax.set_title(f"{sex.capitalize()} \u2014 Feature Importance",fontweight="bold"); ax.grid(axis="x",alpha=0.3)
        ar=all_ai_results.get(sex,[]); ax=axes[si,1]
        if ar:
            from statsmodels.stats.multitest import multipletests
            ad=pd.DataFrame(ar).sort_values("r",ascending=True); yp2=np.arange(len(ad))
            _, p_fdr_arr, _, _ = multipletests(ad["p"].values, method="fdr_bh")
            cc2=["#d62728" if p<0.05 else "#cccccc" for p in p_fdr_arr]
            ax.barh(yp2,ad["r"].values,color=cc2,alpha=0.8,edgecolor="black",lw=0.4)
            ax.axvline(0,color="black",lw=1); ax.set_yticks(yp2); ax.set_yticklabels([f.replace("_AI","") for f in ad["feature"]],fontsize=9)
            ax.set_xlabel("Pearson r (AI \u2194 severity)",fontweight="bold"); ax.set_title(f"{sex.capitalize()} \u2014 Univariate AI",fontweight="bold"); ax.grid(axis="x",alpha=0.3)
    plt.suptitle(f"Sex-Stratified Analysis \u2014 {target_name}",fontsize=13,fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_sex_diagnostic(df_diag, y_diag, pal_l_h, pal_r_h, save_path=None):
    """3 figures: distributions, hemispheric scatter, correlation bars."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    SEX="sex_mapped"; clrs={"male":"#4878CF","female":"#E24A33"}
    # Figure 1: distributions
    fig1,axes1=plt.subplots(1,2,figsize=(12,4.5))
    for sex in ["male","female"]:
        m=df_diag[SEX]==sex
        axes1[0].hist(y_diag[m.values],bins=20,alpha=0.5,color=clrs[sex],label=f"{sex.capitalize()} (n={m.sum()})",edgecolor="white")
        if "pallidum_AI" in df_diag.columns:
            axes1[1].hist(df_diag.loc[m,"pallidum_AI"],bins=25,alpha=0.5,color=clrs[sex],label=f"{sex.capitalize()}",edgecolor="white")
    for ax,xl,ttl in zip(axes1,["PQ-BC Severity","Pallidum AI"],["Severity by Sex","Pallidum AI by Sex"]):
        ax.set_xlabel(xl); ax.set_ylabel("Count"); ax.set_title(ttl); ax.legend()
    fig1.suptitle("Identical Distributions Across Sexes",fontsize=13,fontweight="bold",y=1.02)
    plt.tight_layout(); sp1=str(save_path).replace(".png","_dist.png") if save_path else None
    if sp1: fig1.savefig(sp1,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig1)
    # Figure 2: hemispheric scatter
    feats=[("pallidum_AI","Pallidum AI","A. Pallidum AI \u00d7 Severity"),(pal_l_h,"Left Pallidum (mm\u00b3)","B. Left Pallidum"),(pal_r_h,"Right Pallidum (mm\u00b3)","C. Right Pallidum")]
    fig2,axes2=plt.subplots(1,3,figsize=(16,5))
    for ax,(feat,xl,ttl) in zip(axes2,feats):
        if feat not in df_diag.columns: continue
        for sex in ["male","female"]:
            m=df_diag[SEX]==sex; x=df_diag.loc[m,feat]; y=y_diag[m.values]
            rv,pv=pearsonr(x,y); sig="***" if pv<0.001 else ("**" if pv<0.01 else ("*" if pv<0.05 else "n.s."))
            ax.scatter(x,y,alpha=0.35,s=20,color=clrs[sex],label=f"{sex.capitalize()}: r={rv:+.3f} ({sig})")
            z=np.polyfit(x,y,1); xl2=np.linspace(x.min(),x.max(),100); ax.plot(xl2,np.polyval(z,xl2),color=clrs[sex],lw=2)
        ax.set_xlabel(xl); ax.set_ylabel("PQ-BC Severity"); ax.set_title(ttl); ax.legend(fontsize=9)
    fig2.suptitle("Hemispheric Decomposition",fontsize=13,fontweight="bold",y=1.02)
    plt.tight_layout(); sp2=str(save_path).replace(".png","_hemi.png") if save_path else None
    if sp2: fig2.savefig(sp2,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig2)
    # Figure 3: bar chart
    fig3,ax3=plt.subplots(figsize=(8,5))
    feats_b=["pallidum_AI",pal_l_h,pal_r_h]; lbls=["Pallidum AI\n\u00d7 Severity","L Pallidum\n\u00d7 Severity","R Pallidum\n\u00d7 Severity"]
    mr,fr,mp,fp=[],[],[],[]
    for feat in feats_b:
        if feat not in df_diag.columns: mr.append(0);fr.append(0);mp.append(1);fp.append(1);continue
        for sex,(lr,lp) in [("male",(mr,mp)),("female",(fr,fp))]:
            m=df_diag[SEX]==sex; rv,pv=pearsonr(df_diag.loc[m,feat],y_diag[m.values]); lr.append(rv);lp.append(pv)
    x3=np.arange(len(lbls)); w=0.35
    ax3.bar(x3-w/2,mr,w,label="Male",color=clrs["male"],alpha=0.8); ax3.bar(x3+w/2,fr,w,label="Female",color=clrs["female"],alpha=0.8)
    for i,(pm,pf) in enumerate(zip(mp,fp)):
        for j,(pv,off,rv) in enumerate([(pm,-w/2,mr[i]),(pf,w/2,fr[i])]):
            st="***" if pv<0.001 else ("**" if pv<0.01 else ("*" if pv<0.05 else ""))
            if st: ax3.text(x3[i]+off,rv+(0.01 if rv>=0 else -0.02),st,ha="center",va="bottom",fontsize=11,fontweight="bold")
    ax3.axhline(0,color="black",lw=0.8); ax3.set_ylabel("Pearson r"); ax3.set_xticks(x3); ax3.set_xticklabels(lbls)
    ax3.set_title("Sex-Stratified Correlations",fontsize=12,fontweight="bold"); ax3.legend(); plt.tight_layout()
    if save_path: fig3.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig3)


def plot_longitudinal_overview(long_df, wide_df, tp_with_data, tp_col, target_col, id_col, save_path=None):
    """Histograms per timepoint + spaghetti + change score distribution."""
    import numpy as np, matplotlib.pyplot as plt
    from core.regression.longitudinal import compute_change_scores
    nh=len(tp_with_data); fig,axes=plt.subplots(2,max(nh,2),figsize=(5*max(nh,2),8))
    for i,(tpn,tpv,_) in enumerate(tp_with_data):
        ax=axes[0,i]; td=long_df[long_df[tp_col]==tpv][target_col].dropna()
        ax.hist(td,bins=30,alpha=0.7,color="steelblue",edgecolor="black")
        ax.set_title(f"{tpn} (n={len(td)})",fontweight="bold"); ax.set_xlabel("PQ-BC"); ax.set_ylabel("Count")
        ax.axvline(td.mean(),color="red",ls="--",lw=1.5,label=f"mean={td.mean():.1f}"); ax.axvline(td.median(),color="orange",ls=":",lw=1.5,label=f"med={td.median():.0f}"); ax.legend(fontsize=8)
    for i in range(nh,max(nh,2)): axes[0,i].set_visible(False)
    ax_t=axes[1,0]; tpnum={tpv:i for i,(tpn,tpv,_) in enumerate(tp_with_data)}; tplbl=[tpn for tpn,_,_ in tp_with_data]
    if len(tp_with_data)>=2:
        sc=long_df.groupby(id_col)[tp_col].nunique(); mi=sc[sc>=2].index
        sids=np.random.RandomState(42).choice(mi,size=min(100,len(mi)),replace=False)
        for sid in sids:
            s=long_df[long_df[id_col]==sid].sort_values("tp_idx")
            x=[tpnum[tp] for tp in s[tp_col] if tp in tpnum]; y=[s[s[tp_col]==tp][target_col].values[0] for tp in s[tp_col] if tp in tpnum]
            ax_t.plot(x,y,alpha=0.15,color="steelblue",lw=0.8)
        my=[long_df[long_df[tp_col]==tpv][target_col].dropna().mean() for _,tpv,_ in tp_with_data]
        ax_t.plot(range(len(tp_with_data)),my,"ro-",lw=2.5,ms=8,zorder=10,label="Mean")
        ax_t.set_xticks(range(len(tp_with_data))); ax_t.set_xticklabels(tplbl)
        ax_t.set_ylabel("PQ-BC Severity",fontweight="bold"); ax_t.set_title("Individual Trajectories (n=100 sample)",fontweight="bold")
        ax_t.legend(fontsize=9); ax_t.grid(alpha=0.2)
    ax_d=axes[1,1]
    if f"{target_col}_year2" in wide_df.columns:
        cdf=compute_change_scores(wide_df,target_col,"baseline","year2"); dl=cdf["delta_target"]
        ax_d.hist(dl,bins=30,alpha=0.7,color="steelblue",edgecolor="black"); ax_d.axvline(0,color="black",lw=1); ax_d.axvline(dl.mean(),color="red",ls="--",lw=1.5)
        ax_d.set_xlabel("Delta PQ-BC (Y2\u2212BL)",fontweight="bold"); ax_d.set_ylabel("Count"); ax_d.set_title(f"Change Scores (n={len(dl)})",fontweight="bold")
        nw=(dl>0).sum(); nb2=(dl<0).sum(); ns=(dl==0).sum()
        ax_d.legend([f"mean={dl.mean():.1f}",f"Worsened: {nw} ({100*nw/len(dl):.0f}%)",f"Improved: {nb2} ({100*nb2/len(dl):.0f}%)",f"Stable: {ns} ({100*ns/len(dl):.0f}%)"],fontsize=8)
    else: ax_d.text(0.5,0.5,"No year 2 data",ha="center",va="center",transform=ax_d.transAxes)
    for i in range(2,max(nh,2)): axes[1,i].set_visible(False)
    plt.suptitle("Longitudinal PQ-BC Overview",fontsize=13,fontweight="bold",y=1.02)
    plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_stability_heatmaps(stability_raw, stability_ai, save_path=None):
    """Side-by-side heatmaps for raw and AI feature stability across timepoints."""
    import numpy as np, matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for ai,(lbl,sdf) in enumerate([("Raw L/R",stability_raw),("Asymmetry Index",stability_ai)]):
        fd=sdf[~sdf["feature"].str.startswith("best") & sdf["r"].notna()].copy()
        if not len(fd): axes[ai].text(0.5,0.5,"No data",ha="center",va="center"); axes[ai].set_title(lbl); continue
        def sn(f): return f.replace("smri_vol_scs_","").replace("dmri_dtimd_fiberat_","CST ").replace("dmri_dtifa_fiberat_","CST-FA ").replace("_AI","")
        tps=fd["timepoint"].unique(); feats=fd[fd["timepoint"]==tps[0]]["feature"].values; sf=[sn(f) for f in feats]
        R=np.full((len(feats),len(tps)),np.nan); P=np.full((len(feats),len(tps)),np.nan); ntp={}
        for j,tp in enumerate(tps):
            td=fd[fd["timepoint"]==tp]; ntp[tp]=int(td["n"].iloc[0]) if len(td)>0 else 0
            for i,ft in enumerate(feats):
                row=td[td["feature"]==ft]
                if len(row): R[i,j]=row["r"].values[0]; P[i,j]=row["p"].values[0]
        ax=axes[ai]; vmax=max(0.15,np.nanmax(np.abs(R))); im=ax.imshow(R,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
        ax.set_xticks(range(len(tps))); ax.set_xticklabels([f"{tp}\n(n={ntp[tp]})" for tp in tps],fontsize=9)
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(sf,fontsize=9)
        for i in range(len(feats)):
            for j in range(len(tps)):
                if not np.isnan(R[i,j]):
                    st="***" if P[i,j]<0.001 else ("**" if P[i,j]<0.01 else ("*" if P[i,j]<0.05 else ""))
                    ax.text(j,i,f"{R[i,j]:.3f}{st}",ha="center",va="center",fontsize=8,color="white" if abs(R[i,j])>vmax*0.6 else "black")
        ax.set_title(lbl,fontweight="bold",fontsize=12); plt.colorbar(im,ax=ax,shrink=0.6,label="Pearson r")
    plt.suptitle("Feature-PQ-BC Correlations Across Timepoints: Raw vs Asymmetry",fontweight="bold",fontsize=13,y=1.02)
    plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_attrition_results(y_comp, y_drop, d, p, all_d, all_labels, n_complete, n_dropout, save_path=None):
    """Completers vs dropouts: PQ-BC histogram + effect size bars."""
    import matplotlib.pyplot as plt
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
    ax1.hist(y_comp,bins=30,alpha=0.6,color="steelblue",label=f"Completers (n={n_complete})",density=True)
    ax1.hist(y_drop,bins=30,alpha=0.6,color="salmon",label=f"Dropouts (n={n_dropout})",density=True)
    ax1.set_xlabel("PQ-BC Severity (Baseline)"); ax1.set_ylabel("Density")
    ax1.set_title(f"Baseline PQ-BC: d={d:+.3f}, p={p:.4f}",fontweight="bold"); ax1.legend(fontsize=9)
    bc2=["crimson" if abs(d)>0.5 else ("salmon" if abs(d)>0.2 else "steelblue") for d in all_d]
    ax2.bar(range(len(all_d)),all_d,color=bc2,edgecolor="black",alpha=0.8); ax2.axhline(0,color="black",lw=0.5)
    ax2.set_xticks(range(len(all_labels))); ax2.set_xticklabels(all_labels,rotation=45,ha="right",fontsize=8)
    ax2.set_ylabel("Cohen's d (Completers \u2212 Dropouts)",fontweight="bold")
    ax2.set_title(f"Attrition Effects ({n_complete} completers, {n_dropout} dropouts)",fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_developmental_change(results_dev, asym_bl, asym_y2, ai_names, n_valid, save_path=None):
    """Baseline vs Year 2 AI paired bars + Cohen's d."""
    import numpy as np, matplotlib.pyplot as plt
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
    short=[n.replace("smri_vol_scs_","").replace("dmri_dtimd_fiberat_","CST ").replace("_AI","") for n in ai_names]
    blm=[asym_bl[n].mean() for n in ai_names]; y2m=[asym_y2[n].mean() for n in ai_names]
    pv=[r["p"] for r in results_dev]; yp=np.arange(len(ai_names))
    ax1.barh(yp-0.15,blm,height=0.3,color="steelblue",alpha=0.8,label="Baseline",edgecolor="black")
    ax1.barh(yp+0.15,y2m,height=0.3,color="coral",alpha=0.8,label="Year 2",edgecolor="black")
    for i,p in enumerate(pv):
        if p<0.05: ax1.text(max(blm[i],y2m[i])+0.001,i,"*" if p>=0.01 else ("**" if p>=0.001 else "***"),va="center",fontsize=12,fontweight="bold")
    ax1.set_yticks(yp); ax1.set_yticklabels(short,fontsize=9); ax1.set_xlabel("AI",fontweight="bold")
    ax1.set_title(f"AI: Baseline vs Year 2 (n={n_valid})",fontweight="bold"); ax1.axvline(0,color="gray",ls="--",lw=0.8); ax1.legend(fontsize=9)
    dv=[r["d"] for r in results_dev]; bc2=["crimson" if r["p"]<0.05 else "steelblue" for r in results_dev]
    si=np.argsort(dv)
    ax2.barh(range(len(ai_names)),[dv[i] for i in si],color=[bc2[i] for i in si],alpha=0.8,edgecolor="black")
    ax2.set_yticks(range(len(ai_names))); ax2.set_yticklabels([short[i] for i in si],fontsize=9)
    ax2.set_xlabel("Cohen's d (Y2\u2212BL)",fontweight="bold"); ax2.set_title("Effect Size: Dev. Change in AI",fontweight="bold")
    ax2.axvline(0,color="black",lw=1); ax2.grid(axis="x",alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_icc_bars(all_icc_df, save_path=None):
    """ICC(3,1) bar charts with 95% CIs for AI and raw features."""
    import numpy as np, matplotlib.pyplot as plt
    types=all_icc_df["type"].unique(); fig,axes=plt.subplots(1,len(types),figsize=(7*len(types),max(4,len(all_icc_df)//max(len(types),1)*0.35+2)))
    if len(types)==1: axes=[axes]
    for ax,t in zip(axes,types):
        sub=all_icc_df[all_icc_df["type"]==t].sort_values("icc",ascending=True)
        short=[f.replace("smri_vol_scs_","").replace("dmri_dtimd_fiberat_","CST ").replace("_AI","") for f in sub["feature"]]
        iccs=sub["icc"].values; err=np.clip(np.array([iccs-sub["ci_lo"].values,sub["ci_hi"].values-iccs]),0,None)
        cc=["#4CAF50" if i>0.75 else ("#FFC107" if i>0.5 else "#F44336") for i in iccs]; yp=np.arange(len(sub))
        ax.barh(yp,iccs,xerr=err,color=cc,alpha=0.8,edgecolor="black",lw=0.4,error_kw={"capsize":4,"ecolor":"black"})
        ax.axvline(0.75,color="green",ls="--",lw=1.5,label="Good (0.75)"); ax.axvline(0.5,color="orange",ls=":",lw=1.5,label="Moderate (0.50)")
        ax.set_yticks(yp); ax.set_yticklabels(short,fontsize=9); ax.set_xlim(0,1.0)
        ax.set_xlabel("ICC(3,1)",fontweight="bold"); ax.set_title(f"{t} \u2014 Reliability",fontweight="bold"); ax.legend(fontsize=8); ax.grid(axis="x",alpha=0.3)
    plt.suptitle("ICC(3,1): Baseline vs Year 2",fontsize=13,fontweight="bold"); plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_offset_trajectories(groups_data, asym_bl_v, asym_y2_v, bl_diff, p_bl,
                              delta_ctrl, delta_high, save_path=None):
    """Pallidum AI trajectories: same slope, different intercept by severity group."""
    import numpy as np, matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,7))
    ctrl_m=high_m=None
    for lbl,mask,color in groups_data:
        if mask.sum()<10: continue
        bm=asym_bl_v["pallidum_AI"][mask].mean(); ym=asym_y2_v["pallidum_AI"][mask].mean()
        bs=asym_bl_v["pallidum_AI"][mask].std()/np.sqrt(mask.sum()); ys=asym_y2_v["pallidum_AI"][mask].std()/np.sqrt(mask.sum())
        ax.errorbar([0,1],[bm,ym],yerr=[bs*1.96,ys*1.96],marker="o",ms=10,lw=3,capsize=6,capthick=2,color=color,label=f"{lbl} (n={mask.sum()})",zorder=5)
        ax.text(0.05,bm,f"{bm:+.4f}",va="center",fontsize=9,color=color,fontweight="bold")
        ax.text(0.95,ym,f"{ym:+.4f}",va="center",fontsize=9,color=color,fontweight="bold",ha="right")
        if "Control" in lbl: ctrl_m=mask
        if "High" in lbl: high_m=mask
    if ctrl_m is not None and high_m is not None:
        cm=asym_bl_v["pallidum_AI"][ctrl_m].mean(); hm=asym_bl_v["pallidum_AI"][high_m].mean()
        ax.annotate("",xy=(-0.12,hm),xytext=(-0.12,cm),arrowprops=dict(arrowstyle="<->",color="black",lw=1.5))
        ax.text(-0.18,(cm+hm)/2,f"Gap={bl_diff:.4f}\np={p_bl:.3f}",ha="center",va="center",fontsize=9,rotation=90,bbox=dict(boxstyle="round,pad=0.3",facecolor="lightyellow",edgecolor="gray"))
    ax.set_xticks([0,1]); ax.set_xticklabels(["Baseline (~age 10)","Year 2 (~age 12)"],fontsize=12,fontweight="bold")
    ax.set_ylabel("Pallidum AI  (L\u2212R)/(L+R)",fontsize=13,fontweight="bold")
    ax.set_title(f"Pallidum AI Developmental Trajectory\nHigh-severity\n(Controls \u0394={delta_ctrl:.4f}, High \u0394={delta_high:.4f})",fontweight="bold",fontsize=12)
    ax.legend(fontsize=10,loc="upper left"); ax.grid(alpha=0.3,ls="--")
    plt.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_lateralization_bars(names, ai_rs, ai_ps, tot_rs, best_name, best_ai, y,
                              target_name, network_name, name_map=None, save_path=None):
    """Bar chart: AI vs Total correlations + best-AI scatter (NB07 cell 9 viz)."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import linregress, pearsonr
    if name_map is None:
        name_map = {'caudate':'Caudate','putamen':'Putamen','pallidum':'Pallidum',
                    'vedc':'VEDC/VTA','aa':'Accumbens','accumbens':'Accumbens',
                    'tp':'Thalamus','scs_MD':'SCS MD'}
    display_names = [name_map.get(n, n) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    x_pos = np.arange(len(names)); width = 0.35
    ax.bar(x_pos - width/2, ai_rs, width, label="Asymmetry Index (L-R)/(L+R)", color="#d62728", alpha=0.7)
    ax.bar(x_pos + width/2, tot_rs, width, label="Total Volume", color="#1f77b4", alpha=0.7)
    for i, (r_val, p_val) in enumerate(zip(ai_rs, ai_ps)):
        star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        if star:
            y_off = r_val + 0.008 * np.sign(r_val)
            ax.text(i - width/2, y_off, star, ha='center',
                    va='bottom' if r_val > 0 else 'top', fontsize=10, fontweight='bold', color='#d62728')
    ax.set_xticks(x_pos); ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(f"Pearson r with {target_name}", fontweight="bold")
    ax.set_title("(A) Asymmetry vs Total Volume (ComBat)", fontweight="bold", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5); ax.legend(loc='lower left'); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    best_r, best_p = pearsonr(best_ai, y)
    best_display = name_map.get(best_name, best_name)
    ax.scatter(best_ai, y, alpha=0.25, s=15, color='#555555', edgecolors='none', rasterized=True)
    slope, intercept, *_ = linregress(best_ai, y)
    x_line = np.linspace(best_ai.min(), best_ai.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='#d62728', linewidth=2.5, label=f'r = {best_r:.3f} (p = {best_p:.4f})')
    n_pts = len(best_ai); x_mean = best_ai.mean(); ss_x = np.sum((best_ai - x_mean)**2)
    resid = y - (slope * best_ai + intercept); mse_v = np.sum(resid**2) / (n_pts - 2)
    se_fit = np.sqrt(mse_v * (1/n_pts + (x_line - x_mean)**2 / ss_x))
    ax.fill_between(x_line, y_line - 1.96*se_fit, y_line + 1.96*se_fit, alpha=0.2, color='#d62728')
    ax.set_xlabel(f"{best_display} Asymmetry Index  (L\u2212R)/(L+R)", fontweight="bold")
    ax.set_ylabel("PQ-BC Severity Score", fontweight="bold")
    ax.set_title(f"(B) {best_display} Asymmetry vs Psychosis Severity", fontweight="bold", fontsize=12)
    ax.legend(fontsize=11, loc='upper left'); ax.grid(alpha=0.3)
    plt.suptitle(f"Lateralization Decomposition: {network_name.upper()} \u2192 {target_name} (ComBat-harmonised)",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_bootstrap_ci(boot_pearson, boot_spearman, real_r, ci_pearson,
                      real_rho, ci_spearman, n_boot, network_name, target_name, save_path=None):
    """Bootstrap distribution plots for Pearson r and Spearman rho (NB07 cell 12 viz)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, label, ci, real_val in [
        (axes[0], boot_pearson, "Pearson r", ci_pearson, real_r),
        (axes[1], boot_spearman, "Spearman rho", ci_spearman, real_rho),
    ]:
        ax.hist(vals, bins=60, color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.3)
        ax.axvline(real_val, color="red", linewidth=2, linestyle="--", label=f"{label} = {real_val:.3f}")
        ax.axvline(ci[0], color="orange", linewidth=1.5, linestyle=":", label=f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
        ax.axvline(ci[1], color="orange", linewidth=1.5, linestyle=":")
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(label, fontweight="bold"); ax.set_ylabel("Count", fontweight="bold")
        ax.set_title(f"Bootstrap {label}", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(f"Bootstrap CIs ({n_boot:,} resamples): {network_name.upper()} \u2192 {target_name}", fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_family_history(group_df, fhx_asym, fh_pos_mask, fh_neg_mask,
                        int_df, ai_pqbc, y_pqbc, fh_pqbc,
                        n_fhx_pos, n_fhx_neg, save_path=None):
    """Two figures: FH group comparison + interaction analysis (NB07 cell 18 viz)."""
    import numpy as np, matplotlib.pyplot as plt
    ai_names = list(group_df["feature"])
    # Figure 1: group differences
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_g = group_df.sort_values("cohen_d").reset_index(drop=True)
    y_pos = np.arange(len(plot_g))
    colors_g = ["#d62728" if p < 0.05 else "#aaaaaa" for p in plot_g.get("p_fdr", plot_g["p"])]
    ax1.barh(y_pos, plot_g["cohen_d"].values, color=colors_g, alpha=0.75, edgecolor="black", lw=0.4)
    ax1.axvline(0, color="black", lw=1); ax1.set_yticks(y_pos); ax1.set_yticklabels(plot_g["feature"].values, fontsize=10)
    ax1.set_xlabel("Cohen's d (FH+ minus FH-)", fontsize=11, fontweight="bold")
    ax1.set_title(f"FH+ vs FH- Asymmetry\n(FH+={n_fhx_pos}, FH-={n_fhx_neg})", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    top3 = group_df.head(3)["feature"].tolist()
    positions = []; violins_data = []; tick_labels = []
    for i, feat in enumerate(top3):
        ai_vals = fhx_asym[feat]
        violins_data.append(ai_vals[fh_neg_mask]); violins_data.append(ai_vals[fh_pos_mask])
        positions.extend([i*2.5, i*2.5+1]); tick_labels.extend([f"{feat}\nFH-", f"{feat}\nFH+"])
    parts = ax2.violinplot(violins_data, positions=positions, showmeans=True, showmedians=False)
    for i, pc in enumerate(parts['bodies']): pc.set_facecolor('#1f77b4' if i%2==0 else '#d62728'); pc.set_alpha(0.6)
    ax2.set_xticks(positions); ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_ylabel("Asymmetry Index", fontsize=11); ax2.set_title("Top 3 AI by Group", fontsize=12, fontweight="bold"); ax2.grid(alpha=0.25)
    plt.tight_layout()
    if save_path: fig1.savefig(str(save_path).replace(".png", "_groups.png"), dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig1)
    # Figure 2: interaction
    top_feat = int_df.iloc[0]["feature"]; ai_v = ai_pqbc[top_feat]
    fh_pos_pqbc = fh_pqbc == 1; fh_neg_pqbc = fh_pqbc == 0
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(ai_v[fh_neg_pqbc], y_pqbc[fh_neg_pqbc], alpha=0.25, s=15, color="#1f77b4", label=f"FH- (n={fh_neg_pqbc.sum()})")
    ax1.scatter(ai_v[fh_pos_pqbc], y_pqbc[fh_pos_pqbc], alpha=0.5, s=25, color="#d62728", label=f"FH+ (n={fh_pos_pqbc.sum()})")
    for mask, color in [(fh_neg_pqbc, "#1f77b4"), (fh_pos_pqbc, "#d62728")]:
        if mask.sum() > 10:
            z = np.polyfit(ai_v[mask], y_pqbc[mask], 1)
            xln = np.linspace(ai_v[mask].min(), ai_v[mask].max(), 100)
            ax1.plot(xln, np.poly1d(z)(xln), color=color, lw=2.5)
    top_row = int_df.iloc[0]
    ax1.set_xlabel(f"{top_feat}", fontsize=12, fontweight="bold"); ax1.set_ylabel("PQ-BC Severity", fontsize=12, fontweight="bold")
    ax1.set_title(f"FH x {top_feat} Interaction\np_int={top_row['p_interaction']:.4f}", fontsize=12, fontweight="bold"); ax1.legend(framealpha=0.9); ax1.grid(alpha=0.25)
    ax2_df = int_df.sort_values("beta_interaction").reset_index(drop=True)
    y_pos2 = np.arange(len(ax2_df))
    ci2 = ["#d62728" if p < 0.05 else "#aaaaaa" for p in ax2_df.get("p_int_fdr", ax2_df["p_interaction"])]
    ax2.barh(y_pos2, ax2_df["beta_interaction"].values, color=ci2, alpha=0.75, edgecolor="black", lw=0.4)
    ax2.axvline(0, color="black", lw=1); ax2.set_yticks(y_pos2); ax2.set_yticklabels(ax2_df["feature"].values, fontsize=10)
    ax2.set_xlabel("Interaction β (AI x FH)", fontsize=11, fontweight="bold"); ax2.set_title("AI x FH Interactions", fontsize=12, fontweight="bold"); ax2.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path: fig2.savefig(str(save_path).replace(".png", "_interaction.png"), dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig2)


def plot_sex_hemi_y2(cohort_results, valid_pairs, present_cols, pal_name,
                     pal_l, pal_r, target_col, save_path=None):
    """Sex-stratified hemispheric AI scatter at Year 2 across two cohort definitions."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    colors = {"male": "steelblue", "female": "salmon"}
    n_cohorts = len(cohort_results)
    fig, axes = plt.subplots(n_cohorts, 3, figsize=(17, 5 * n_cohorts))
    if n_cohorts == 1: axes = axes[None, :]
    for row_idx, (label, data) in enumerate(cohort_results.items()):
        y2_sev = data["df"]; y_col = data["y"]; asym = data["asym"]
        y2_sev = y2_sev.reset_index(drop=True)
        sex_col2 = data.get("sex_col", "sex_mapped")
        for col_idx, (feat, xlabel) in enumerate([
            ("pallidum_AI", "Pallidum AI"), (pal_l, "L Pallidum"), (pal_r, "R Pallidum"),
        ]):
            ax = axes[row_idx, col_idx]
            for sex, color in colors.items():
                mask = y2_sev[sex_col2] == sex
                if feat in asym:
                    x = asym[feat][mask.values]
                else:
                    x = y2_sev.loc[mask, feat].values if feat in y2_sev.columns else np.zeros(mask.sum())
                y_vals = y_col[mask.values] if hasattr(y_col, '__len__') else y_col
                if len(x) > 5:
                    r, p = pearsonr(x, y_vals)
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
                    ax.scatter(x, y_vals, alpha=0.35, s=20, color=color, label=f"{sex.capitalize()}: r={r:+.3f} ({sig})")
                    z = np.polyfit(x, y_vals, 1)
                    xln = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xln, np.poly1d(z)(xln), color=color, linewidth=2)
            ax.set_xlabel(xlabel, fontweight="bold"); ax.set_ylabel("PQ-BC Severity", fontweight="bold")
            ax.set_title(f"{label[:25]}\n{xlabel}", fontweight="bold", fontsize=10); ax.legend(fontsize=8)
    plt.suptitle("Sex-Stratified Hemispheric Decomposition at Year 2", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_y2_svr_scatter(all_true_p, all_pred_p, r_val, p_val, label, save_path=None):
    """Simple scatter plot for prospective/concurrent SVR results."""
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_true_p, all_pred_p, alpha=0.35, s=20, color="steelblue", edgecolors="none", rasterized=True)
    mn, mx = min(all_true_p.min(), all_pred_p.min()), max(all_true_p.max(), all_pred_p.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label=f"r = {r_val:.3f} (p = {p_val:.4f})")
    z = np.polyfit(all_true_p, all_pred_p, 1)
    xln = np.linspace(all_true_p.min(), all_true_p.max(), 100)
    ax.plot(xln, np.poly1d(z)(xln), color='steelblue', lw=2)
    ax.set_xlabel("Observed PQ-BC", fontweight="bold"); ax.set_ylabel("Predicted PQ-BC", fontweight="bold")
    ax.set_title(f"{label}\nr = {r_val:.3f}, p = {p_val:.4f}", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)


def plot_persistent_remitted(ai_names, asym, mask_p, mask_r, n_persistent, n_remitted,
                              bl_df, y2_df, target_col, cutoff, save_path=None):
    """Cohen's d bars + trajectory plot for persistent vs remitted group (NB09 cell 17 viz)."""
    import numpy as np, matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    from core.regression.longitudinal import cohens_d
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Panel 1: Cohen's d bars
    dvals, pvals, names_plot = [], [], []
    for ai in ai_names:
        a_p = asym[ai][mask_p]; a_r = asym[ai][mask_r]
        if len(a_p) > 1 and len(a_r) > 1:
            t, p = ttest_ind(a_p, a_r, equal_var=False)
            d = cohens_d(a_p, a_r)
            dvals.append(d); pvals.append(p); names_plot.append(ai.replace("_AI", ""))
    colors_p = ["#d62728" if p < 0.05 else ("#E8A0A0" if p < 0.1 else "steelblue") for p in pvals]
    ax = axes[0]
    y_pos = np.arange(len(names_plot))
    ax.barh(y_pos, dvals, color=colors_p, alpha=0.75, edgecolor="black", lw=0.5)
    ax.axvline(0, color="black", lw=1); ax.set_yticks(y_pos); ax.set_yticklabels(names_plot, fontsize=10)
    ax.set_xlabel("Cohen's d (Persistent − Remitted)", fontweight="bold", fontsize=11)
    ax.set_title(f"BL Asymmetry: Persistent (n={n_persistent}) vs\nRemitted (n={n_remitted})", fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(handles=[plt.Rectangle((0,0),1,1,fc="#d62728",ec="black",label="p<0.05 uncorr."),
                       plt.Rectangle((0,0),1,1,fc="steelblue",ec="black",label="n.s.")], fontsize=9, loc="lower right")
    # Panel 2: Trajectory (BL→Y2 for top feature)
    if names_plot:
        top_ai = ai_names[np.argmax([abs(d) for d in dvals])]
        ai_bl_p = asym[top_ai][mask_p]; ai_bl_r = asym[top_ai][mask_r]
        ax2 = axes[1]
        for label_g, ai_bl_g, color_g in [("Persistent", ai_bl_p, "#d62728"), ("Remitted", ai_bl_r, "steelblue")]:
            ax2.scatter([0]*len(ai_bl_g), ai_bl_g, alpha=0.3, s=15, color=color_g)
            ax2.scatter([1]*len(ai_bl_g), ai_bl_g, alpha=0.3, s=15, color=color_g)
            for v in ai_bl_g: ax2.plot([0, 1], [v, v], color=color_g, alpha=0.12, lw=0.8)
            ax2.plot([0, 1], [ai_bl_g.mean(), ai_bl_g.mean()], color=color_g, lw=3, label=f"{label_g} (n={len(ai_bl_g)})")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Baseline", "Year 2 (proxy)"])
        ax2.set_ylabel(f"{top_ai.replace('_AI','')} AI", fontweight="bold")
        ax2.set_title(f"Symptom Trajectory Group:\n{top_ai.replace('_AI','')}", fontweight="bold")
        ax2.legend(fontsize=10); ax2.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)



def plot_sibling_structure(full_df, env, save_path=None) -> None:
    """Visualize family/sibling structure in the dataset.

    Panel A: Distribution of family sizes (# subjects per family).
    Panel B: Singleton vs multi-sibling subject breakdown.
    """
    family_col = "rel_family_id"
    if family_col not in full_df.columns:
        print(f"[plot_sibling_structure] Column '{family_col}' not found — skipping.")
        return

    df = full_df.dropna(subset=[family_col]).copy()
    family_sizes = df.groupby(family_col).size()

    n_total = len(df)
    n_families = len(family_sizes)
    n_singletons = (family_sizes == 1).sum()
    n_multi = (family_sizes > 1).sum()
    n_in_multi = family_sizes[family_sizes > 1].sum()
    max_size = int(family_sizes.max())

    print(f"Family structure (n={n_total:,}):")
    print(f"  Unique families   : {n_families:,}")
    print(f"  Singletons        : {n_singletons:,} ({n_singletons/n_families*100:.1f}% of families)")
    print(f"  Multi-sib families: {n_multi:,}  — {n_in_multi:,} subjects ({n_in_multi/n_total*100:.1f}% of sample)")
    print(f"  Max family size   : {max_size}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel A — family size histogram
    ax = axes[0]
    counts = family_sizes.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Family size (# subjects)", fontweight="bold")
    ax.set_ylabel("# Families", fontweight="bold")
    ax.set_title(f"Family Size Distribution\n(N={n_total:,}, {n_families:,} families)", fontweight="bold")
    ax.set_xticks(sorted(counts.index))
    ax.grid(axis="y", alpha=0.3)

    # Panel B — breakdown
    ax = axes[1]
    labels = ["Singleton\nfamilies", "Subjects in\nmulti-sib families", "Multi-sib\nfamilies"]
    values = [int(n_singletons), int(n_in_multi), int(n_multi)]
    colors = ["#4878CF", "#E24A33", "#6ACC65"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Singleton vs Multi-Sibling Breakdown", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Dataset Family/Sibling Structure (Family-Aware CV)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
