"""Evaluation metrics and cross-validation utilities."""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    fbeta_score,
    matthews_corrcoef,
)


def get_cv_splitter(config: dict, seed: int) -> StratifiedKFold:
    """Create stratified K-fold splitter for outer CV from config."""
    cv_cfg = config.get("cv", {})
    return StratifiedKFold(
        n_splits=cv_cfg.get("n_outer_splits", 5),
        shuffle=cv_cfg.get("shuffle", True),
        random_state=seed,
    )


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Compute comprehensive classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Add ROC-AUC using decision scores or probabilities
    if y_score is not None:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            # For binary: use positive class probability or decision function
            if y_score.ndim == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_score, multi_class="ovr", average="weighted"
            )

    return metrics


def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
    default_threshold: float = 0.5,
    metric: str = "balanced_accuracy",
    beta: float = 0.5,
) -> tuple[float, float]:
    """Select threshold that maximizes a chosen metric on validation scores."""
    if thresholds is None or len(thresholds) == 0:
        y_pred = (scores >= default_threshold).astype(int)
        return default_threshold, _compute_metric(y_true, y_pred, metric, beta)

    best_threshold = default_threshold
    best_score = -np.inf

    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        score = _compute_metric(y_true, y_pred, metric, beta)
        if score > best_score:
            best_score = score
            best_threshold = thr

    return best_threshold, best_score


def _compute_metric(y_true, y_pred, metric: str, beta: float) -> float:
    metric = (metric or "balanced_accuracy").lower()
    if metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if metric == "fbeta":
        return fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
    if metric == "mcc":
        return matthews_corrcoef(y_true, y_pred)
    # Default to balanced accuracy
    return balanced_accuracy_score(y_true, y_pred)


def aggregate_cv_predictions(fold_results: list) -> dict:
    """Combine predictions from all CV folds into overall metrics.

    Each subject appears in exactly one test fold, so concatenating
    all fold predictions gives complete out-of-fold predictions.

    Args:
        fold_results: List of dicts with keys: y_test, y_pred, y_score, metrics

    Returns:
        Dict with 'overall' metrics and 'per_fold' statistics
    """
    # Concatenate all test predictions (each subject appears once)
    all_y_true = np.concatenate([fold["y_test"] for fold in fold_results])
    all_y_pred = np.concatenate([fold["y_pred"] for fold in fold_results])

    # Concatenate scores if available
    if "y_score" in fold_results[0] and fold_results[0]["y_score"] is not None:
        all_y_score = np.concatenate([fold["y_score"] for fold in fold_results])
    else:
        all_y_score = None

    # Compute overall metrics on all concatenated predictions
    overall_metrics = compute_metrics(all_y_true, all_y_pred, all_y_score)

    # Compute per-fold statistics (mean ± std)
    per_fold_metrics = [fold["metrics"] for fold in fold_results]
    fold_stats = {}
    for metric_name in per_fold_metrics[0].keys():
        values = [fm[metric_name] for fm in per_fold_metrics]
        fold_stats[f"{metric_name}_mean"] = np.mean(values)
        fold_stats[f"{metric_name}_std"] = np.std(values)

    return {
        "overall": overall_metrics,
        "per_fold": fold_stats,
        "n_folds": len(fold_results),
        "n_samples": len(all_y_true),
    }


def bootstrap_test_metrics(model, X_test, y_test, n_iterations=1000, seed=42):
    """Compute bootstrap confidence intervals for test metrics (optimized).

    Args:
        model: Trained model or list of models (for ensemble)
        X_test: Test features
        y_test: Test labels
        n_iterations: Number of bootstrap iterations (default 1000)
        seed: Random seed

    Returns:
        Dict with mean and 95% CI for each metric
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_test)

    # Check if model is an ensemble (list of models)
    is_ensemble = isinstance(model, list)

    # Pre-compute all predictions once (saves time)
    if is_ensemble:
        # Ensemble: average predictions across all models
        all_preds = np.array([m.predict(X_test) for m in model])
        y_pred_all = (all_preds.mean(axis=0) >= 0.5).astype(int)

        # Get scores from ensemble
        if hasattr(model[0], "decision_function"):
            all_scores = np.array([m.decision_function(X_test) for m in model])
            y_score_all = all_scores.mean(axis=0)
        elif hasattr(model[0], "predict_proba"):
            all_probas = np.array([m.predict_proba(X_test)[:, 1] for m in model])
            y_score_all = all_probas.mean(axis=0)
        else:
            y_score_all = None
    else:
        # Single model
        y_pred_all = model.predict(X_test)

        # Get probability scores (works for both SVM and Random Forest)
        if hasattr(model, "decision_function"):
            y_score_all = model.decision_function(X_test)
        elif hasattr(model, "predict_proba"):
            y_score_all = model.predict_proba(X_test)[
                :, 1
            ]  # Positive class probability
        else:
            y_score_all = None

    metrics_bootstrap = {
        "accuracy": np.zeros(n_iterations),
        "balanced_accuracy": np.zeros(n_iterations),
        "roc_auc": np.zeros(n_iterations) if y_score_all is not None else None,
    }

    # Vectorized bootstrap (much faster)
    for i in range(n_iterations):
        # Bootstrap sample indices
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_test[indices]
        y_pred_boot = y_pred_all[indices]

        # Compute metrics
        metrics_bootstrap["accuracy"][i] = accuracy_score(y_boot, y_pred_boot)
        metrics_bootstrap["balanced_accuracy"][i] = balanced_accuracy_score(
            y_boot, y_pred_boot
        )
        if y_score_all is not None:
            metrics_bootstrap["roc_auc"][i] = roc_auc_score(
                y_boot, y_score_all[indices]
            )

    # Compute mean and 95% CI
    results = {}
    for metric_name, values in metrics_bootstrap.items():
        if values is not None:
            results[f"{metric_name}_mean"] = np.mean(values)
            results[f"{metric_name}_ci_lower"] = np.percentile(values, 2.5)
            results[f"{metric_name}_ci_upper"] = np.percentile(values, 97.5)

    return results
