"""Evaluation metrics and cross-validation utilities."""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


def get_cv_splitter(config: dict, seed: int) -> StratifiedKFold:
    """Create stratified K-fold splitter from config."""
    cv_cfg = config.get("cv", {})
    return StratifiedKFold(
        n_splits=cv_cfg.get("n_splits", 5),
        shuffle=cv_cfg.get("shuffle", True),
        random_state=seed,
    )


def run_cross_validation(model, X, y, cv_splitter, scoring: dict) -> dict:
    """Run cross-validation and return scores."""
    return cross_validate(
        model,
        X,
        y,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
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


def aggregate_cv_scores(cv_results: dict) -> dict:
    """Aggregate cross-validation scores into mean ± std."""
    aggregated = {}
    for key, values in cv_results.items():
        if key.startswith("test_") or key.startswith("train_"):
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
    return aggregated
