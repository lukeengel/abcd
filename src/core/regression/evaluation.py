"""Evaluation metrics for regression."""

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from scipy.stats import pearsonr, spearmanr


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

    # Pearson correlation
    if len(y_true) > 1:
        r, p = pearsonr(y_true, y_pred)
        metrics["pearson_r"] = float(r)
        metrics["pearson_p"] = float(p)

        # Spearman correlation (robust to outliers)
        rho, p_rho = spearmanr(y_true, y_pred)
        metrics["spearman_r"] = float(rho)
        metrics["spearman_p"] = float(p_rho)
    else:
        metrics["pearson_r"] = 0.0
        metrics["pearson_p"] = 1.0
        metrics["spearman_r"] = 0.0
        metrics["spearman_p"] = 1.0

    return metrics


def aggregate_cv_results(folds: list[dict]) -> dict:
    """Aggregate regression results across CV folds."""
    # Overall metrics (all folds concatenated)
    all_y_true = np.concatenate([fold["y_test"] for fold in folds])
    all_y_pred = np.concatenate([fold["y_pred"] for fold in folds])
    overall_metrics = compute_regression_metrics(all_y_true, all_y_pred)

    # Per-fold statistics
    per_fold_metrics = {}
    metric_names = ["r2", "mae", "mse", "rmse", "pearson_r", "spearman_r"]

    for metric_name in metric_names:
        fold_values = [fold["metrics"][metric_name] for fold in folds]
        per_fold_metrics[f"{metric_name}_mean"] = float(np.mean(fold_values))
        per_fold_metrics[f"{metric_name}_std"] = float(np.std(fold_values))
        per_fold_metrics[f"{metric_name}_min"] = float(np.min(fold_values))
        per_fold_metrics[f"{metric_name}_max"] = float(np.max(fold_values))

    return {
        "overall": overall_metrics,
        "per_fold": per_fold_metrics,
        "n_folds": len(folds),
        "n_samples": len(all_y_true),
    }
