"""Shared hyperparameter tuning module for SVM and Random Forest."""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.base import clone
from tqdm import tqdm
from itertools import product
from typing import Dict, List, Any, Tuple
from joblib import Parallel, delayed


def tune_hyperparameters(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    base_model,
    param_grid: Dict[str, List[Any]],
    seed: int,
    n_folds: int = 5,
    use_random_sampling: bool = True,
    n_iterations: int = 20,
    scoring: str = "roc_auc",
    model_type: str = "rf",
    n_jobs: int = 1,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Tune hyperparameters using cross-validation.

    Args:
        X_dev: Development set features (PCA-transformed)
        y_dev: Development set labels
        base_model: Base model instance (will be cloned with new params)
        param_grid: Dictionary of hyperparameter lists to search
        seed: Random seed
        n_folds: Number of CV folds for tuning
        use_random_sampling: Whether to use 1:1 balanced sampling
        n_iterations: Number of sampling iterations if use_random_sampling=True
        scoring: Metric to optimize ('roc_auc' or 'balanced_accuracy')
        model_type: 'rf' for Random Forest or 'svm' for SVM
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        best_params: Best hyperparameters found
        tuning_results: List of results for each parameter combination
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING ({model_type.upper()})")
    print(f"{'='*70}")
    print("Parameter grid:")
    for param_name, values in param_grid.items():
        print(f"  {param_name}: {values}")
    print(f"\nTotal combinations to test: {len(param_combinations)}")
    print(f"CV folds: {n_folds}")
    print(f"Parallel jobs: {n_jobs if n_jobs != -1 else 'all CPUs'}")
    print(
        f"Random sampling: {use_random_sampling} (iterations: {n_iterations if use_random_sampling else 1})"
    )
    print(f"Optimization metric: {scoring}")
    print(f"{'='*70}\n")

    # Setup CV splitter
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    tuning_results = []
    best_score = -np.inf
    best_params = None

    # Pre-compute CV splits
    cv_splits = list(cv.split(X_dev, y_dev))

    # Test each parameter combination
    for param_tuple in tqdm(param_combinations, desc="Grid search"):
        params = dict(zip(param_names, param_tuple))

        # Evaluate this parameter combination with CV (parallel across folds)
        fold_scores = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_fold)(
                X_dev,
                y_dev,
                train_idx,
                val_idx,
                base_model,
                params,
                use_random_sampling,
                n_iterations,
                seed,
                fold_idx,
                scoring,
                model_type,
            )
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits)
        )

        # Average score across folds
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        result = {
            "params": params,
            "mean_score": mean_score,
            "std_score": std_score,
            "fold_scores": fold_scores,
        }
        tuning_results.append(result)

        # Track best parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Sort results by score
    tuning_results = sorted(tuning_results, key=lambda x: x["mean_score"], reverse=True)

    # Print top 5 results
    print(f"\n{'='*70}")
    print("TUNING RESULTS (Top 5):")
    print(f"{'='*70}")
    for i, result in enumerate(tuning_results[:5], 1):
        print(
            f"\n{i}. {scoring}: {result['mean_score']:.4f} ± {result['std_score']:.4f}"
        )
        print("   Parameters:")
        for k, v in result["params"].items():
            print(f"     {k}: {v}")

    print(f"\n{'='*70}")
    print(f"BEST PARAMETERS (optimizing {scoring}):")
    print(f"{'='*70}")
    print(f"{scoring}: {best_score:.4f}")
    print("Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*70}\n")

    return best_params, tuning_results


def _evaluate_fold(
    X_dev,
    y_dev,
    train_idx,
    val_idx,
    base_model,
    params,
    use_random_sampling,
    n_iterations,
    seed,
    fold_idx,
    scoring,
    model_type,
):
    """Evaluate a single CV fold for a given parameter set."""
    X_train_fold = X_dev[train_idx]
    y_train_fold = y_dev[train_idx]
    X_val_fold = X_dev[val_idx]
    y_val_fold = y_dev[val_idx]

    if use_random_sampling:
        # Train ensemble of models on balanced subsets
        rng = np.random.RandomState(seed + fold_idx)
        all_scores = []

        for _ in range(n_iterations):
            # Balance classes
            X_balanced, y_balanced = _balance_classes(X_train_fold, y_train_fold, rng)

            # Clone model with new parameters
            model = clone(base_model)
            model.set_params(**params)
            model.fit(X_balanced, y_balanced)

            # Predict on validation
            if model_type == "svm" and hasattr(model, "decision_function"):
                y_score_val = model.decision_function(X_val_fold)
            else:
                y_score_val = model.predict_proba(X_val_fold)[:, 1]

            all_scores.append(y_score_val)

        # Ensemble prediction
        y_score_val_ensemble = np.mean(all_scores, axis=0)
    else:
        # Train single model
        model = clone(base_model)
        model.set_params(**params)
        model.fit(X_train_fold, y_train_fold)

        if model_type == "svm" and hasattr(model, "decision_function"):
            y_score_val_ensemble = model.decision_function(X_val_fold)
        else:
            y_score_val_ensemble = model.predict_proba(X_val_fold)[:, 1]

    # Compute score
    if scoring == "roc_auc":
        score = roc_auc_score(y_val_fold, y_score_val_ensemble)
    elif scoring == "balanced_accuracy":
        if model_type == "svm":
            y_pred_val = (y_score_val_ensemble >= 0).astype(int)
        else:
            y_pred_val = (y_score_val_ensemble >= 0.5).astype(int)
        score = balanced_accuracy_score(y_val_fold, y_pred_val)
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")

    return score


def _balance_classes(X, y, rng):
    """Balance classes using 1:1 sampling (all minority + matched majority)."""
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)

    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    n_minority = len(minority_indices)

    # Sample majority class to match minority size
    sampled_majority = rng.choice(
        majority_indices,
        size=n_minority,
        replace=False,
    )

    # Combine indices
    balanced_indices = np.concatenate([minority_indices, sampled_majority])
    rng.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]
