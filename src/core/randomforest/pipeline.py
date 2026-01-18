"""Random Forest classification pipeline with nested CV (no fixed holdout test set)."""

import logging
import pickle

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply, harmonizationLearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..downsampling import get_balanced_indices
from ..svm.evaluation import (
    aggregate_cv_predictions,
    compute_confusion_matrix,
    compute_metrics,
    find_best_threshold,
    get_cv_splitter,
)
from ..svm.pipeline import filter_task_data, load_full_dataset
from ..svm.preprocessing import apply_pca_to_fold, fit_pca_on_dev
from ..svm.visualization import plot_confusion_matrix
from .models import create_baseline

logger = logging.getLogger(__name__)


def fit_platt_scaler(scores: np.ndarray, targets: np.ndarray) -> LogisticRegression | None:
    """Fit Platt scaling on out-of-fold scores; return None if calibration is not feasible."""
    if scores is None or len(scores) == 0 or len(np.unique(targets)) < 2:
        return None

    scaler = LogisticRegression(solver="lbfgs", max_iter=1000)
    scaler.fit(scores.reshape(-1, 1), targets)
    return scaler


def apply_platt_scaler(scaler: LogisticRegression | None, raw_scores: np.ndarray) -> np.ndarray:
    """Apply fitted Platt scaler to raw probability scores."""
    if scaler is None:
        return raw_scores
    calibrated = scaler.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    return calibrated


def extract_rf_harmonization_data(df: pd.DataFrame, env) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract imaging features and covariates for RF harmonization."""
    from ..tsne.embeddings import get_imaging_columns

    rf_config = env.configs.randomforest
    harm_config = env.configs.harmonize

    imaging_cols = get_imaging_columns(df, rf_config["imaging_prefixes"])
    X = df[imaging_cols].values

    site_col = harm_config["site_column"]
    covariate_cols = [site_col] + harm_config.get("covariates", [])
    covars = df[covariate_cols].copy()
    covars = covars.rename(columns={site_col: "SITE"})

    return X, covars


def fit_raw_features_on_dev(dev_df: pd.DataFrame, env, seed: int) -> dict:
    """Fit harmonization + scaling pipeline on dev set (NO PCA for RF)."""
    harm_config = env.configs.harmonize

    X, covars = extract_rf_harmonization_data(dev_df, env)

    # Remove zero-variance features
    feature_vars = np.var(X, axis=0)
    valid_features = feature_vars > 1e-10
    X = X[:, valid_features]

    # Harmonization
    eb = harm_config.get("empirical_bayes", True)
    smooth_terms = harm_config.get("smooth_terms", [])
    combat_model, X_harm = harmonizationLearn(X, covars, eb=eb, smooth_terms=smooth_terms)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_harm)

    return {
        "combat_model": combat_model,
        "scaler": scaler,
        "valid_features": valid_features,
        "n_features": X_scaled.shape[1],
    }


def apply_raw_features_to_fold(fold_df: pd.DataFrame, fitted_pipeline: dict, env) -> np.ndarray:
    """Apply harmonization + scaling to fold (NO PCA for RF)."""
    # Extract imaging features and covariates
    X_fold, fold_covars = extract_rf_harmonization_data(fold_df, env)

    # Apply feature filtering
    X_fold = X_fold[:, fitted_pipeline["valid_features"]]

    # Apply harmonization (fitted model already has reference site info)
    X_harm = harmonizationApply(X_fold, fold_covars, fitted_pipeline["combat_model"])

    # Apply scaling
    X_scaled = fitted_pipeline["scaler"].transform(X_harm)

    return X_scaled


def get_threshold_candidates(config: dict) -> list[float] | None:
    """Return thresholds to evaluate for RF decision tuning."""
    eval_cfg = config.get("evaluation", {})
    search_cfg = eval_cfg.get("threshold_search", {})

    if search_cfg and not search_cfg.get("enabled", True):
        return None

    thresholds = None
    if isinstance(search_cfg, dict):
        thresholds = search_cfg.get("thresholds")

    if thresholds is None:
        thresholds = np.linspace(0.02, 0.30, 15).tolist()

    return [float(t) for t in thresholds]


def tune_hyperparameters_with_gridsearch(X_train, y_train, rf_config, seed):
    """Tune RF hyperparameters and derive a validation-informed threshold."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        balanced_accuracy_score,
        roc_auc_score,
        fbeta_score,
        precision_score,
        average_precision_score,
    )
    from sklearn.model_selection import ParameterGrid

    tuning_config = rf_config.get("tuning", {})
    param_grid = tuning_config.get("param_grid", {})
    scoring_metric = tuning_config.get("scoring", "balanced_accuracy")
    cv_folds = tuning_config.get("cv_folds", 3)

    decision_threshold = rf_config.get("evaluation", {}).get("decision_threshold", 0.5)
    threshold_candidates = get_threshold_candidates(rf_config)
    model_cfg = rf_config["model"]
    param_combinations = list(ParameterGrid(param_grid))
    inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    logger.info(f"    Tuning with nested CV ({cv_folds} inner folds)...")
    logger.info(f"    Scoring metric: {scoring_metric}")
    if threshold_candidates:
        logger.info(f"    Threshold search grid: {threshold_candidates}")
    logger.info(f"    Testing {len(param_combinations)} parameter combinations")
    logger.info(
        f"    Progress: Training {len(param_combinations)} × {cv_folds} = "
        f"{len(param_combinations) * cv_folds} models...\n"
    )

    scoring_name = tuning_config.get("scoring", "balanced_accuracy").lower()
    beta = float(tuning_config.get("fbeta_beta", 0.5))

    best_result = None

    for idx, params in enumerate(param_combinations, 1):
        bal_scores = []
        roc_scores = []
        pr_auc_scores = []
        precision_scores = []
        fbeta_scores = []
        val_scores = []
        val_targets = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train), 1):
            full_params = {
                "class_weight": model_cfg.get("class_weight", "balanced_subsample"),
                "bootstrap": model_cfg.get("bootstrap", True),
                "oob_score": False,
                "n_jobs": model_cfg.get("n_jobs", -1),
                "random_state": seed + inner_fold,
                "verbose": 0,
            }
            if "min_samples_split" not in param_grid:
                full_params["min_samples_split"] = model_cfg.get("min_samples_split", 2)
            full_params.update(params)

            model = RandomForestClassifier(**full_params)
            model.fit(X_train[inner_train_idx], y_train[inner_train_idx])

            y_val = y_train[inner_val_idx]
            val_proba = model.predict_proba(X_train[inner_val_idx])[:, 1]
            val_pred = (val_proba >= decision_threshold).astype(int)

            bal_scores.append(balanced_accuracy_score(y_val, val_pred))
            roc_scores.append(roc_auc_score(y_val, val_proba))
            pr_auc_scores.append(average_precision_score(y_val, val_proba))
            precision_scores.append(precision_score(y_val, val_pred, zero_division=0))
            fbeta_scores.append(fbeta_score(y_val, val_pred, beta=beta, zero_division=0))

            val_scores.append(val_proba)
            val_targets.append(y_val)

        bal_acc_mean = float(np.mean(bal_scores))
        bal_acc_std = float(np.std(bal_scores))
        roc_mean = float(np.mean(roc_scores))
        float(np.std(roc_scores))
        pr_auc_mean = float(np.mean(pr_auc_scores))
        pr_auc_std = float(np.std(pr_auc_scores))
        precision_mean = float(np.mean(precision_scores))
        fbeta_mean = float(np.mean(fbeta_scores))

        params_str = str(params).replace("'", "")
        print(
            f"    [{idx}/{len(param_combinations)}] {params_str:<70} "
            f"bal_acc={bal_acc_mean:.3f}±{bal_acc_std:.3f}  "
            f"pr_auc={pr_auc_mean:.3f}±{pr_auc_std:.3f}"
        )

        aggregate_targets = np.concatenate(val_targets)
        aggregate_scores = np.concatenate(val_scores)
        thresholds_for_metric = threshold_candidates or [decision_threshold]

        if scoring_name == "roc_auc":
            metric_score = roc_mean
            selected_metric_threshold = thresholds_for_metric[0]
        elif scoring_name == "average_precision":
            metric_score = pr_auc_mean
            selected_metric_threshold = thresholds_for_metric[0]
        else:
            best_metric = -np.inf
            selected_metric_threshold = thresholds_for_metric[0]
            for thr in thresholds_for_metric:
                y_pred_thr = (aggregate_scores >= thr).astype(int)
                if scoring_name == "precision":
                    metric_val = precision_score(aggregate_targets, y_pred_thr, zero_division=0)
                elif scoring_name == "fbeta":
                    metric_val = fbeta_score(aggregate_targets, y_pred_thr, beta=beta, zero_division=0)
                else:
                    metric_val = balanced_accuracy_score(aggregate_targets, y_pred_thr)
                if metric_val > best_metric:
                    best_metric = metric_val
                    selected_metric_threshold = thr
            metric_score = best_metric

        if best_result is None or metric_score > best_result["score"]:
            best_result = {
                "params": params,
                "score": metric_score,
                "roc_mean": roc_mean,
                "bal_acc_mean": bal_acc_mean,
                "precision_mean": precision_mean,
                "fbeta_mean": fbeta_mean,
                "val_targets": aggregate_targets,
                "val_scores": aggregate_scores,
                "metric_threshold": selected_metric_threshold,
            }

    best_params = best_result["params"]
    best_metric_score = best_result["score"]

    # Use threshold metric from config if specified, otherwise use tuning metric if appropriate
    threshold_metric_config = rf_config.get("evaluation", {}).get("threshold_metric")
    if threshold_metric_config:
        metric_for_threshold = threshold_metric_config.lower()
    elif scoring_name in {"precision", "fbeta", "balanced_accuracy"}:
        metric_for_threshold = scoring_name
    else:
        metric_for_threshold = "balanced_accuracy"

    calibrator = fit_platt_scaler(best_result["val_scores"], best_result["val_targets"])

    calibration_scores = best_result["val_scores"]
    if calibrator is not None:
        calibration_scores = apply_platt_scaler(calibrator, calibration_scores)

    calibrated_threshold_metric = None
    if threshold_candidates:
        best_threshold, calibrated_threshold_metric = find_best_threshold(
            best_result["val_targets"],
            calibration_scores,
            threshold_candidates,
            default_threshold=decision_threshold,
            metric=metric_for_threshold,
            beta=beta,
        )
    else:
        best_threshold = decision_threshold

    logger.info(f"\n    ✓ BEST (by {scoring_metric}): {best_params} → {scoring_metric}={best_metric_score:.3f}")
    if threshold_candidates:
        logger.info(
            f"    ✓ Selected threshold {best_threshold:.3f} "
            f"(validation {metric_for_threshold}={calibrated_threshold_metric:.3f})"
        )

    logger.info("    Training final model with best params on full training data...")
    full_params = {
        "class_weight": model_cfg.get("class_weight", "balanced_subsample"),
        "bootstrap": model_cfg.get("bootstrap", True),
        "oob_score": False,
        "n_jobs": model_cfg.get("n_jobs", -1),
        "random_state": seed,
        "verbose": 0,
    }
    if "min_samples_split" not in param_grid:
        full_params["min_samples_split"] = model_cfg.get("min_samples_split", 2)
    full_params.update(best_params)

    best_model = RandomForestClassifier(**full_params)
    best_model.fit(X_train, y_train)

    logger.info("    ✓ Final model trained (Platt scaling from inner folds)")

    return best_params, best_model, calibrator, best_threshold


def run_single_fold(
    env,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fold_idx: int,
    seed: int,
) -> dict:
    """Process a single outer CV fold for Random Forest.

    Steps:
    1. Optionally fit PCA on train data, OR use raw features (harmonized + scaled)
    2. Optionally tune hyperparameters using GridSearchCV (does its own inner CV)
    3. Train final models on ALL training data with best/default hyperparameters
    4. Evaluate on test (20%)
    """
    rf_config = env.configs.randomforest
    use_pca = rf_config.get("use_pca", False)
    threshold_candidates = get_threshold_candidates(rf_config)

    # Check if downsampling is enabled
    downsample_config = rf_config.get("downsampling", {})
    use_downsampling = downsample_config.get("enabled", False)
    n_iterations = downsample_config.get("n_iterations", 100)

    logger.info(f"  Train: {len(y_train)} (class: {np.bincount(y_train)}) | Test: {len(y_test)}")

    # Feature extraction: PCA or raw features
    if use_pca:
        # Use PCA features (compressed)
        fitted_pipeline = fit_pca_on_dev(train_df, env, seed + fold_idx)
        X_train, _ = apply_pca_to_fold(train_df, train_df, fitted_pipeline, env)
        X_test, _ = apply_pca_to_fold(test_df, test_df, fitted_pipeline, env)
        print(f"  Using PCA features: {X_train.shape[1]} components")
    else:
        # Use raw features (harmonized + scaled)
        fitted_pipeline = fit_raw_features_on_dev(train_df, env, seed + fold_idx)
        X_train = apply_raw_features_to_fold(train_df, fitted_pipeline, env)
        X_test = apply_raw_features_to_fold(test_df, fitted_pipeline, env)
        print(f"  Using raw features: {X_train.shape[1]} features (harmonized + scaled)")

    # Train baseline with threshold optimization
    decision_threshold = rf_config.get("evaluation", {}).get("decision_threshold", 0.5)
    inner_cv_splits = rf_config.get("tuning", {}).get("cv_folds", 3)

    baseline_threshold = decision_threshold
    if threshold_candidates:
        baseline_val_scores = []
        baseline_val_targets = []
        baseline_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=seed + fold_idx)
        for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(baseline_cv.split(X_train, y_train), 1):
            temp_model = create_baseline(rf_config, seed + fold_idx + inner_idx)
            temp_model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
            baseline_val_scores.append(temp_model.predict_proba(X_train[inner_val_idx])[:, 1])
            baseline_val_targets.append(y_train[inner_val_idx])

        threshold_metric = rf_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
        baseline_threshold, _ = find_best_threshold(
            np.concatenate(baseline_val_targets),
            np.concatenate(baseline_val_scores),
            threshold_candidates,
            default_threshold=decision_threshold,
            metric=threshold_metric,
        )

    baseline_model = create_baseline(rf_config, seed + fold_idx)
    baseline_model.fit(X_train, y_train)

    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_pred = (baseline_proba >= baseline_threshold).astype(int)
    baseline_score = baseline_proba
    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_score)

    # Downsampling with averaging
    if use_downsampling:
        import sys
        import os
        from tqdm.auto import tqdm as tqdm_auto
        from sklearn.model_selection import train_test_split

        print(f"\n  Downsampling: {n_iterations} iterations (1:1 balanced)")
        imbalance = max(np.bincount(y_train)) / min(np.bincount(y_train))
        print(f"  Imbalance: {imbalance:.1f}:1 → 1:1")

        # Split training into train/val for threshold optimization
        inner_val_ratio = rf_config.get("cv", {}).get("inner_val_ratio", 0.25)
        X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
            X_train,
            y_train,
            test_size=inner_val_ratio,
            stratify=y_train,
            random_state=seed + fold_idx,
        )

        all_val_probas = []
        all_test_probas = []
        all_iter_params = []  # Track all params to see distribution
        best_iter_roc_auc = -1
        best_iter_params = {}
        tuning_enabled = rf_config.get("tuning", {}).get("enabled", False)

        pbar = tqdm_auto(range(n_iterations), desc="  Training")
        for iter_idx in pbar:
            # Downsample from inner training set only
            balanced_idx = get_balanced_indices(y_inner_train, seed=seed + fold_idx + 1000 + iter_idx)
            X_train_balanced = X_inner_train[balanced_idx]
            y_train_balanced = y_inner_train[balanced_idx]

            # Train model (suppress tuning output, keep stderr for tqdm)
            if tuning_enabled:
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                try:
                    (
                        iter_best_params,
                        iter_model,
                        _,
                        _,
                    ) = tune_hyperparameters_with_gridsearch(
                        X_train_balanced,
                        y_train_balanced,
                        rf_config,
                        seed + fold_idx + iter_idx,
                    )
                finally:
                    sys.stdout.close()
                    sys.stdout = old_stdout
            else:
                from sklearn.ensemble import RandomForestClassifier

                model_cfg = rf_config["model"].copy()
                iter_model = RandomForestClassifier(
                    n_estimators=model_cfg.get("n_estimators", 100),
                    max_depth=model_cfg.get("max_depth", None),
                    min_samples_split=model_cfg.get("min_samples_split", 2),
                    min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
                    max_features=model_cfg.get("max_features", "sqrt"),
                    class_weight=model_cfg.get("class_weight", "balanced_subsample"),
                    bootstrap=model_cfg.get("bootstrap", True),
                    oob_score=model_cfg.get("oob_score", False),
                    n_jobs=model_cfg.get("n_jobs", -1),
                    random_state=seed + fold_idx + iter_idx,
                    verbose=0,
                )
                iter_model.fit(X_train_balanced, y_train_balanced)
                iter_best_params = model_cfg

            # Predict on validation and test sets
            iter_val_proba = iter_model.predict_proba(X_inner_val)[:, 1]
            iter_test_proba = iter_model.predict_proba(X_test)[:, 1]

            all_val_probas.append(iter_val_proba)
            all_test_probas.append(iter_test_proba)

            # Compute validation metrics for monitoring
            from sklearn.metrics import (
                roc_auc_score,
                balanced_accuracy_score,
                precision_score,
                average_precision_score,
            )

            iter_val_pred = (iter_val_proba >= 0.5).astype(int)
            iter_bal_acc = balanced_accuracy_score(y_inner_val, iter_val_pred)
            iter_ppv = precision_score(y_inner_val, iter_val_pred, zero_division=0)
            iter_roc_auc = roc_auc_score(y_inner_val, iter_val_proba)
            iter_pr_auc = average_precision_score(y_inner_val, iter_val_proba)

            # Track all params and best performing iteration
            all_iter_params.append(iter_best_params)
            if iter_roc_auc > best_iter_roc_auc:
                best_iter_roc_auc = iter_roc_auc
                best_iter_params = iter_best_params

            pbar.set_postfix(
                {
                    "bal_acc": f"{iter_bal_acc:.3f}",
                    "ppv": f"{iter_ppv:.3f}",
                    "roc_auc": f"{iter_roc_auc:.3f}",
                    "pr_auc": f"{iter_pr_auc:.3f}",
                }
            )

        pbar.close()

        # Average predictions
        rf_val_proba = np.mean(all_val_probas, axis=0)
        rf_raw_proba = np.mean(all_test_probas, axis=0)
        rf_model = iter_model  # Keep last model for reference

        # Use params from best performing iteration
        best_params = best_iter_params

        # Print hyperparameter distribution
        if tuning_enabled and len(all_iter_params) > 0:
            from collections import Counter

            print(f"\n  Hyperparameter distribution across {n_iterations} iterations:")

            # Count each unique parameter value
            for param_name in [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
            ]:
                if param_name in all_iter_params[0]:
                    values = [p.get(param_name) for p in all_iter_params if param_name in p]
                    if values:
                        counts = Counter(values)
                        total = len(values)
                        print(f"    {param_name}:")
                        for value, count in counts.most_common():
                            pct = 100 * count / total
                            print(f"      {value}: {count}/{total} ({pct:.1f}%)")

        rf_calibrator = None

        # Optimize threshold on VALIDATION SET (not test set!)
        if threshold_candidates:
            threshold_metric = rf_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
            rf_threshold, best_val_score = find_best_threshold(
                y_inner_val,
                rf_val_proba,
                threshold_candidates,
                default_threshold=0.5,
                metric=threshold_metric,
                beta=float(rf_config.get("tuning", {}).get("fbeta_beta", 0.5)),
            )
            print(
                f"  ✓ Averaged {n_iterations} models | Threshold (val {threshold_metric}={best_val_score:.3f}): {rf_threshold:.3f}"
            )
        else:
            rf_threshold = 0.5
            print(f"  ✓ Averaged {n_iterations} models")

    # Hyperparameter tuning with GridSearchCV (if enabled)
    elif rf_config.get("tuning", {}).get("enabled", False):
        best_params = {}
        rf_calibrator = None
        (
            best_params,
            rf_model,
            rf_calibrator,
            rf_threshold,
        ) = tune_hyperparameters_with_gridsearch(X_train, y_train, rf_config, seed + fold_idx)
    else:
        # Use default hyperparameters from config
        from sklearn.ensemble import RandomForestClassifier

        model_cfg = rf_config["model"].copy()

        rf_model = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            max_features=model_cfg.get("max_features", "sqrt"),
            class_weight=model_cfg.get("class_weight", "balanced_subsample"),
            bootstrap=model_cfg.get("bootstrap", True),
            oob_score=model_cfg.get("oob_score", False),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=seed + fold_idx,
            verbose=0,
        )
        rf_model.fit(X_train, y_train)
        rf_threshold = decision_threshold

        # Build calibration from stratified CV on training fold
        cv_folds = rf_config.get("tuning", {}).get("cv_folds", 3)
        fallback_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + fold_idx)
        oof_scores = []
        oof_targets = []
        for inner_idx, (tr_idx, va_idx) in enumerate(fallback_cv.split(X_train, y_train), 1):
            temp_model = RandomForestClassifier(**model_cfg, random_state=seed + fold_idx + inner_idx)
            temp_model.fit(X_train[tr_idx], y_train[tr_idx])
            oof_scores.append(temp_model.predict_proba(X_train[va_idx])[:, 1])
            oof_targets.append(y_train[va_idx])

        if oof_scores:
            oof_scores_concat = np.concatenate(oof_scores)
            oof_targets_concat = np.concatenate(oof_targets)
            rf_calibrator = fit_platt_scaler(oof_scores_concat, oof_targets_concat)
            if rf_calibrator is not None and threshold_candidates:
                calibrated_oof = apply_platt_scaler(rf_calibrator, oof_scores_concat)
                threshold_metric = rf_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
                rf_threshold, _ = find_best_threshold(
                    oof_targets_concat,
                    calibrated_oof,
                    threshold_candidates,
                    default_threshold=decision_threshold,
                    metric=threshold_metric,
                    beta=float(rf_config.get("tuning", {}).get("fbeta_beta", 0.5)),
                )

    # Evaluate on test set with tuned threshold
    logger.info(
        f"\n  Evaluating on test set ({len(y_test)} subjects: {np.bincount(y_test)[1]} clinical, {np.bincount(y_test)[0]} control)..."
    )

    # RF predictions with Platt calibration (or already computed for downsampling)
    if not use_downsampling:
        rf_raw_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_proba = apply_platt_scaler(rf_calibrator, rf_raw_proba)
    rf_pred = (rf_proba >= rf_threshold).astype(int)
    rf_score = rf_proba
    rf_metrics = compute_metrics(y_test, rf_pred, rf_score)

    # Print test set performance
    from sklearn.metrics import precision_score, average_precision_score

    baseline_ppv = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_pr_auc = average_precision_score(y_test, baseline_score)
    rf_ppv = precision_score(y_test, rf_pred, zero_division=0)
    rf_pr_auc = average_precision_score(y_test, rf_score)

    # Format hyperparameters for display
    def format_params(params):
        parts = []
        if "n_estimators" in params:
            parts.append(f"n_est={params['n_estimators']}")
        if "max_depth" in params:
            depth = params["max_depth"]
            parts.append(f"depth={depth if depth is not None else 'None'}")
        if "min_samples_split" in params:
            parts.append(f"min_split={params['min_samples_split']}")
        if "min_samples_leaf" in params:
            parts.append(f"min_leaf={params['min_samples_leaf']}")
        return ", ".join(parts) if parts else "default"

    logger.info("\n  ✓ Test Set Results:")
    logger.info(f"    Hyperparams: {format_params(best_params)}")
    logger.info(
        f"    Baseline (thr={baseline_threshold:.3f}): bal_acc={baseline_metrics['balanced_accuracy']:.3f}  ppv={baseline_ppv:.3f}  roc_auc={baseline_metrics.get('roc_auc', 0):.3f}  pr_auc={baseline_pr_auc:.3f}  ({baseline_pred.sum()}/{len(baseline_pred)} pred+)"
    )
    logger.info(
        f"    RF (thr={rf_threshold:.3f}):      bal_acc={rf_metrics['balanced_accuracy']:.3f}  ppv={rf_ppv:.3f}  roc_auc={rf_metrics.get('roc_auc', 0):.3f}  pr_auc={rf_pr_auc:.3f}  ({rf_pred.sum()}/{len(rf_pred)} pred+)"
    )

    return {
        "baseline": {
            "model": baseline_model,
            "metrics": baseline_metrics,
            "y_pred": baseline_pred,
            "y_score": baseline_score,
            "y_test": y_test,
            "threshold": baseline_threshold,
        },
        "rf": {
            "model": rf_model,
            "calibrator": rf_calibrator,
            "metrics": rf_metrics,
            "y_pred": rf_pred,
            "y_score": rf_score,
            "y_test": y_test,
            "X_test": X_test,  # Raw or PCA features depending on config
            "pipeline": fitted_pipeline,
            "best_params": best_params,  # Store tuned hyperparameters
            "threshold": rf_threshold,
        },
    }


def run_task_with_nested_cv(
    env,
    full_df: pd.DataFrame,
    task_config: dict,
    use_wandb: bool = False,
    sweep_mode: bool = False,
):
    """Run single task with nested CV across all folds."""
    task_name = task_config["name"]
    seed = env.configs.run["seed"]
    rf_config = env.configs.randomforest
    group_col = env.configs.data["columns"]["mapping"]["research_group"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Task: {task_name}")
    logger.info(f"{'='*60}")

    # Filter data for this task
    df_filtered, y = filter_task_data(full_df, task_config, group_col)
    logger.info(f"Total samples: {len(y)} | Class balance: {np.bincount(y)}")
    logger.info(f"Imbalance ratio: 1:{np.bincount(y)[0]/np.bincount(y)[1]:.1f}")

    # Create outer CV splitter (5-fold)
    outer_cv = get_cv_splitter(rf_config, seed)

    # Storage for fold results
    baseline_folds = []
    rf_folds = []

    # Outer CV loop - each fold gets 20% test
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(outer_cv.split(df_filtered, y), total=outer_cv.n_splits, desc="CV Folds")
    ):
        print(f"\nFold {fold_idx + 1}/{outer_cv.n_splits}")

        # Split into train (80%) and test (20%)
        train_df = df_filtered.iloc[train_idx].reset_index(drop=True)
        test_df = df_filtered.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Run this fold (GridSearchCV will handle inner CV for tuning)
        fold_result = run_single_fold(env, train_df, test_df, y_train, y_test, fold_idx, seed)

        baseline_folds.append(fold_result["baseline"])
        rf_folds.append(fold_result["rf"])

    # Aggregate results
    baseline_agg = aggregate_cv_predictions(baseline_folds)
    rf_agg = aggregate_cv_predictions(rf_folds)

    # Global threshold sweep on aggregated predictions (post-hoc)
    threshold_candidates = get_threshold_candidates(rf_config)
    global_threshold = None
    global_threshold_metric = None
    if threshold_candidates:
        all_targets = np.concatenate([fold["y_test"] for fold in rf_folds])
        all_scores = np.concatenate([fold["y_score"] for fold in rf_folds])
        threshold_metric = rf_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
        global_threshold, global_threshold_metric = find_best_threshold(
            all_targets,
            all_scores,
            threshold_candidates,
            default_threshold=rf_config.get("evaluation", {}).get("decision_threshold", 0.5),
            metric=threshold_metric,
            beta=float(rf_config.get("tuning", {}).get("fbeta_beta", 0.5)),
        )
        rf_agg["global_threshold"] = global_threshold
        rf_agg["global_threshold_metric"] = global_threshold_metric

    # Setup output directory
    data_dir = env.repo_root / "outputs" / env.configs.run["run_name"] / env.configs.run["run_id"] / f"seed_{seed}"
    rf_dir = data_dir / "randomforest" / task_name
    rf_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = rf_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {task_name}")
    logger.info(f"{'='*60}")
    logger.info("Baseline (Logistic Regression):")
    logger.info(f"  Overall Balanced Accuracy: {baseline_agg['overall']['balanced_accuracy']:.3f}")
    logger.info(f"  Overall ROC-AUC: {baseline_agg['overall'].get('roc_auc', 0):.3f}")
    logger.info("\nRandom Forest:")
    logger.info(f"  Overall Balanced Accuracy: {rf_agg['overall']['balanced_accuracy']:.3f}")
    logger.info(f"  Overall ROC-AUC: {rf_agg['overall'].get('roc_auc', 0):.3f}")
    logger.info("\nPer-Fold Stats (Mean ± Std):")
    logger.info(
        f"  Balanced Accuracy: {rf_agg['per_fold']['balanced_accuracy_mean']:.3f} ± {rf_agg['per_fold']['balanced_accuracy_std']:.3f}"
    )
    logger.info(f"  ROC-AUC: {rf_agg['per_fold']['roc_auc_mean']:.3f} ± {rf_agg['per_fold']['roc_auc_std']:.3f}")
    if global_threshold is not None:
        threshold_metric_name = rf_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
        logger.info(
            f"  Global threshold ({threshold_metric_name}): {global_threshold:.3f} → {global_threshold_metric:.3f}"
        )
    logger.info(f"{'='*60}\n")

    if not sweep_mode:
        # Visualizations
        all_y_true = np.concatenate([f["y_test"] for f in rf_folds])
        all_y_pred_baseline = np.concatenate([f["y_pred"] for f in baseline_folds])
        all_y_pred_rf = np.concatenate([f["y_pred"] for f in rf_folds])

        cm_baseline = compute_confusion_matrix(all_y_true, all_y_pred_baseline)
        cm_rf = compute_confusion_matrix(all_y_true, all_y_pred_rf)

        plot_confusion_matrix(
            cm_baseline,
            ["Negative", "Positive"],
            f"Baseline - {task_name}",
            plots_dir / f"cm_baseline_{task_name}.png",
        )
        plot_confusion_matrix(
            cm_rf,
            ["Negative", "Positive"],
            f"Random Forest - {task_name}",
            plots_dir / f"cm_rf_{task_name}.png",
        )

        # Save results
        results = {
            "baseline": baseline_agg,
            "rf": rf_agg,
            "baseline_folds": baseline_folds,
            "rf_folds": rf_folds,
            "global_threshold": global_threshold,
        }
        with open(rf_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {rf_dir}\n")

    if use_wandb:
        import wandb

        wandb.log(
            {
                f"{task_name}/overall_balanced_accuracy": rf_agg["overall"]["balanced_accuracy"],
                f"{task_name}/overall_roc_auc": rf_agg["overall"].get("roc_auc", 0),
                f"{task_name}/per_fold_balanced_accuracy_mean": rf_agg["per_fold"]["balanced_accuracy_mean"],
                f"{task_name}/per_fold_balanced_accuracy_std": rf_agg["per_fold"]["balanced_accuracy_std"],
            }
        )

    return {
        "baseline": baseline_agg,
        "rf": rf_agg,
        "global_threshold": global_threshold,
    }


def run_randomforest_pipeline(env, use_wandb: bool = False, sweep_mode: bool = False):
    """Run complete Random Forest pipeline with nested CV."""
    logger.info("=" * 60)
    logger.info("Random Forest Pipeline with Nested Cross-Validation")
    logger.info("=" * 60)
    logger.info("Loading full dataset for 5-fold CV...")

    full_df = load_full_dataset(env)
    logger.info(f"Total samples: {len(full_df)}")

    tasks = env.configs.randomforest.get("tasks", [])
    all_results = {}

    for task_config in tasks:
        task_results = run_task_with_nested_cv(env, full_df, task_config, use_wandb, sweep_mode)
        all_results[task_config["name"]] = task_results

    logger.info("\n" + "=" * 60)
    logger.info("Random Forest Pipeline Complete")
    logger.info("=" * 60)

    return all_results
