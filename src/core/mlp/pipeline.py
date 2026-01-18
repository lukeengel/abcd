"""MLP classification pipeline with nested CV (no fixed holdout test set)."""

import logging
import pickle

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply, harmonizationLearn
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
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
from ..svm.feature_mapping import enrich_brain_regions
from ..svm.interpretation import (
    get_feature_importance_permutation,
    map_pca_to_brain_regions,
)
from ..svm.preprocessing import apply_pca_to_fold, fit_pca_on_dev
from ..svm.visualization import plot_confusion_matrix, plot_feature_importance
from .models import create_baseline

logger = logging.getLogger(__name__)


def load_full_dataset(env) -> pd.DataFrame:
    """Load all data for nested CV (train+val+test combined)."""
    run_cfg = env.configs.run
    data_dir = (
        env.repo_root / "outputs" / run_cfg["run_name"] / run_cfg["run_id"] / f"seed_{run_cfg['seed']}" / "datasets"
    )

    # Load all splits and combine
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return full_df


def filter_task_data(df: pd.DataFrame, task_config: dict, group_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter data for specific classification task."""
    pos_classes = task_config.get("positive_classes") or [task_config.get("positive_class")]
    neg_class = task_config["negative_class"]

    mask = df[group_col].isin(pos_classes + [neg_class])
    df_filtered = df[mask].copy()
    y = np.where(df_filtered[group_col].isin(pos_classes), 1, 0)

    return df_filtered, y


def extract_mlp_harmonization_data(df: pd.DataFrame, env):
    """Extract imaging features and covariates for harmonization (same as RF)."""
    mlp_config = env.configs.mlp
    harm_config = env.configs.harmonize

    from ..tsne.embeddings import get_imaging_columns

    imaging_cols = get_imaging_columns(df, mlp_config["imaging_prefixes"])
    X = df[imaging_cols].values

    site_col = harm_config["site_column"]
    covariate_cols = [site_col] + harm_config.get("covariates", [])
    covars = df[covariate_cols].copy()
    covars = covars.rename(columns={site_col: "SITE"})

    return X, covars


def fit_raw_features_on_train(train_df: pd.DataFrame, env, seed: int) -> dict:
    """Fit harmonization + scaling pipeline on train set (same as RF/SVM)."""
    harm_config = env.configs.harmonize

    X, covars = extract_mlp_harmonization_data(train_df, env)

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
    """Apply harmonization + scaling to fold (same as RF/SVM)."""
    # Extract imaging features and covariates
    X_fold, fold_covars = extract_mlp_harmonization_data(fold_df, env)

    # Apply feature filtering
    X_fold = X_fold[:, fitted_pipeline["valid_features"]]

    # Apply harmonization
    X_harm = harmonizationApply(X_fold, fold_covars, fitted_pipeline["combat_model"])

    # Apply scaling
    X_scaled = fitted_pipeline["scaler"].transform(X_harm)

    return X_scaled


def get_threshold_candidates(config: dict) -> list[float] | None:
    """Return list of thresholds to evaluate for decision optimization."""
    eval_cfg = config.get("evaluation", {})
    search_cfg = eval_cfg.get("threshold_search", {})

    if search_cfg and not search_cfg.get("enabled", True):
        return None

    thresholds = None
    if isinstance(search_cfg, dict):
        thresholds = search_cfg.get("thresholds")

    if thresholds is None:
        # Default search grid focuses on low probabilities for rare positives
        thresholds = np.linspace(0.02, 0.30, 15).tolist()

    return [float(t) for t in thresholds]


def tune_hyperparameters_with_gridsearch(X_train, y_train, mlp_config, seed):
    """Tune MLP hyperparameters and derive a validation-informed threshold."""
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_score,
        roc_auc_score,
        fbeta_score,
        average_precision_score,
    )
    from sklearn.model_selection import ParameterGrid

    tuning_config = mlp_config.get("tuning", {})
    param_grid = tuning_config.get("param_grid", {})
    scoring_metric = tuning_config.get("scoring", "balanced_accuracy")
    scoring_name = scoring_metric.lower()
    beta = float(tuning_config.get("fbeta_beta", 0.5))
    cv_folds = tuning_config.get("cv_folds", 3)

    decision_threshold = mlp_config.get("evaluation", {}).get("decision_threshold", 0.5)
    threshold_candidates = get_threshold_candidates(mlp_config)
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

    best_result = None
    for idx, params in enumerate(param_combinations, 1):
        bal_scores = []
        roc_scores = []
        pr_auc_scores = []
        precision_scores = []
        fbeta_scores = []
        val_scores_threshold = []
        val_targets = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train), 1):
            # Build model params
            model_params = params.copy()

            # Convert hidden_layer_sizes from list to tuple (YAML parses as list)
            if "hidden_layer_sizes" in model_params and isinstance(model_params["hidden_layer_sizes"], list):
                model_params["hidden_layer_sizes"] = tuple(model_params["hidden_layer_sizes"])

            # Add fixed parameters from model config (not in param_grid)
            model_params["solver"] = mlp_config["model"].get("solver", "adam")
            model_params["max_iter"] = mlp_config["model"].get("max_iter", 1000)
            model_params["early_stopping"] = mlp_config["model"].get("early_stopping", True)
            model_params["validation_fraction"] = mlp_config["model"].get("validation_fraction", 0.1)
            model_params["n_iter_no_change"] = mlp_config["model"].get("n_iter_no_change", 10)
            model_params["learning_rate_init"] = mlp_config["model"].get("learning_rate_init", 0.001)
            model_params["tol"] = mlp_config["model"].get("tol", 0.0001)
            model_params["beta_1"] = mlp_config["model"].get("beta_1", 0.9)
            model_params["beta_2"] = mlp_config["model"].get("beta_2", 0.999)
            model_params["random_state"] = seed + inner_fold

            model = MLPClassifier(**model_params)
            model.fit(X_train[inner_train_idx], y_train[inner_train_idx])

            y_val = y_train[inner_val_idx]
            scores_for_threshold = model.predict_proba(X_train[inner_val_idx])[:, 1]
            scores_for_roc = scores_for_threshold

            y_val_pred = (scores_for_threshold >= decision_threshold).astype(int)

            bal_scores.append(balanced_accuracy_score(y_val, y_val_pred))
            roc_scores.append(roc_auc_score(y_val, scores_for_roc))
            pr_auc_scores.append(average_precision_score(y_val, scores_for_threshold))
            precision_scores.append(precision_score(y_val, y_val_pred, zero_division=0))
            fbeta_scores.append(fbeta_score(y_val, y_val_pred, beta=beta, zero_division=0))

            val_scores_threshold.append(scores_for_threshold)
            val_targets.append(y_val)

        bal_acc_mean = float(np.mean(bal_scores))
        bal_acc_std = float(np.std(bal_scores))
        roc_mean = float(np.mean(roc_scores))
        roc_std = float(np.std(roc_scores))
        pr_auc_mean = float(np.mean(pr_auc_scores))
        pr_auc_std = float(np.std(pr_auc_scores))
        prec_mean = float(np.mean(precision_scores))
        prec_std = float(np.std(precision_scores))
        fbeta_mean = float(np.mean(fbeta_scores))

        params_str = str(params).replace("'", "")
        logger.info(
            f"    [{idx}/{len(param_combinations)}] {params_str:<50} "
            f"bal_acc={bal_acc_mean:.3f}±{bal_acc_std:.3f}  "
            f"roc_auc={roc_mean:.3f}±{roc_std:.3f}  "
            f"pr_auc={pr_auc_mean:.3f}±{pr_auc_std:.3f}  "
            f"ppv={prec_mean:.3f}±{prec_std:.3f}"
        )

        aggregate_targets = np.concatenate(val_targets)
        aggregate_threshold_scores = np.concatenate(val_scores_threshold)
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
                y_pred_thr = (aggregate_threshold_scores >= thr).astype(int)
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
                "precision_mean": prec_mean,
                "fbeta_mean": fbeta_mean,
                "val_targets": aggregate_targets,
                "val_scores_threshold": aggregate_threshold_scores,
                "metric_threshold": selected_metric_threshold,
            }

    best_params = best_result["params"]
    best_metric_score = best_result["score"]

    # Use threshold metric from config if specified,
    # otherwise use tuning metric if appropriate
    threshold_metric_config = mlp_config.get("evaluation", {}).get("threshold_metric")
    if threshold_metric_config:
        metric_for_threshold = threshold_metric_config.lower()
    elif scoring_name in {"precision", "fbeta", "balanced_accuracy"}:
        metric_for_threshold = scoring_name
    else:
        metric_for_threshold = "balanced_accuracy"

    threshold_list = threshold_candidates if threshold_candidates else None

    # MLP has native predict_proba - no Platt scaling needed
    calibration_scores = best_result["val_scores_threshold"]

    calibrated_threshold_metric = None
    if threshold_list:
        best_threshold, calibrated_threshold_metric = find_best_threshold(
            best_result["val_targets"],
            calibration_scores,
            threshold_list,
            default_threshold=decision_threshold,
            metric=metric_for_threshold,
            beta=beta,
        )
    else:
        best_threshold = decision_threshold

    logger.info(f"\n    ✓ BEST (by {scoring_metric}): {best_params} " f"→ {scoring_metric}={best_metric_score:.3f}")
    if threshold_candidates:
        logger.info(
            f"    ✓ Selected threshold {best_threshold:.3f} "
            f"(validation {metric_for_threshold}={calibrated_threshold_metric:.3f})"
        )

    logger.info("    Training final model with best params on full training data...")
    # Build final model params
    final_params = best_params.copy()
    # Convert hidden_layer_sizes from list to tuple (YAML parses as list)
    if "hidden_layer_sizes" in final_params and isinstance(final_params["hidden_layer_sizes"], list):
        final_params["hidden_layer_sizes"] = tuple(final_params["hidden_layer_sizes"])

    # Add fixed parameters from model config (not in param_grid)
    final_params["solver"] = mlp_config["model"].get("solver", "adam")
    final_params["max_iter"] = mlp_config["model"].get("max_iter", 1000)
    final_params["early_stopping"] = mlp_config["model"].get("early_stopping", True)
    final_params["validation_fraction"] = mlp_config["model"].get("validation_fraction", 0.1)
    final_params["n_iter_no_change"] = mlp_config["model"].get("n_iter_no_change", 10)
    final_params["learning_rate_init"] = mlp_config["model"].get("learning_rate_init", 0.001)
    final_params["tol"] = mlp_config["model"].get("tol", 0.0001)
    final_params["beta_1"] = mlp_config["model"].get("beta_1", 0.9)
    final_params["beta_2"] = mlp_config["model"].get("beta_2", 0.999)
    final_params["random_state"] = seed

    best_model = MLPClassifier(**final_params)
    best_model.fit(X_train, y_train)

    logger.info("    ✓ Final model trained (native MLP probabilities)")

    return best_params, best_model, best_threshold


def run_single_fold(
    env,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fold_idx: int,
    seed: int,
    use_wandb: bool = False,
) -> dict:
    """Process a single outer CV fold.

    Steps:
    1. Fit PCA on train data (80% of full dataset) if use_pca is enabled
    2. Optionally tune hyperparameters using GridSearchCV (does its own inner CV)
    3. Train final models on ALL training data with best/default hyperparameters
    4. Evaluate on test (20%)

    Args:
        env: Environment with configs
        train_df: Training data (80% of full data)
        test_df: Test data (20% of full data)
        y_train: Labels for train
        y_test: Labels for test
        fold_idx: Fold index (0-4)
        seed: Random seed

    Returns:
        Dict with baseline and mlp results
    """
    mlp_config = env.configs.mlp
    use_pca = mlp_config.get("use_pca", True)
    threshold_candidates = get_threshold_candidates(mlp_config)

    # Check if downsampling is enabled
    downsample_config = mlp_config.get("downsampling", {})
    use_downsampling = downsample_config.get("enabled", False)
    n_iterations = downsample_config.get("n_iterations", 100)

    logger.info(f"  Train: {len(y_train)} (class: {np.bincount(y_train)}) | Test: {len(y_test)}")

    # Apply PCA if enabled, otherwise use raw features with harmonization + scaling
    if use_pca:
        # Fit PCA on train set only (no data leakage)
        fitted_pipeline = fit_pca_on_dev(train_df, env, seed + fold_idx)

        # Apply PCA to both sets
        X_train_pca, _ = apply_pca_to_fold(train_df, train_df, fitted_pipeline, env)
        X_test_pca, _ = apply_pca_to_fold(test_df, test_df, fitted_pipeline, env)
    else:
        # Use raw features with harmonization + scaling (same as RF/SVM)
        fitted_pipeline = fit_raw_features_on_train(train_df, env, seed + fold_idx)
        X_train_pca = apply_raw_features_to_fold(train_df, fitted_pipeline, env)
        X_test_pca = apply_raw_features_to_fold(test_df, fitted_pipeline, env)

    # Train baseline (Logistic Regression with class_weight='balanced')
    decision_threshold = mlp_config.get("evaluation", {}).get("decision_threshold", 0.5)
    inner_cv_splits = mlp_config.get("tuning", {}).get("cv_folds", 3)

    baseline_threshold = decision_threshold
    if threshold_candidates:
        baseline_val_scores = []
        baseline_val_targets = []
        baseline_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=seed + fold_idx)
        for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(baseline_cv.split(X_train_pca, y_train), 1):
            temp_model = create_baseline(mlp_config, seed + fold_idx + inner_idx)
            temp_model.fit(X_train_pca[inner_train_idx], y_train[inner_train_idx])
            baseline_val_scores.append(temp_model.predict_proba(X_train_pca[inner_val_idx])[:, 1])
            baseline_val_targets.append(y_train[inner_val_idx])

        threshold_metric = mlp_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
        # Use MCC for baseline since it handles imbalanced data better
        # (baseline doesn't use downsampling, so validation is imbalanced 22:1)
        baseline_threshold, _ = find_best_threshold(
            np.concatenate(baseline_val_targets),
            np.concatenate(baseline_val_scores),
            threshold_candidates,
            default_threshold=decision_threshold,
            metric="mcc",  # MCC better for imbalanced data
        )

    baseline_model = create_baseline(mlp_config, seed + fold_idx)
    baseline_model.fit(X_train_pca, y_train)

    # Baseline predictions with optimized threshold
    baseline_proba = baseline_model.predict_proba(X_test_pca)[:, 1]
    baseline_pred = (baseline_proba >= baseline_threshold).astype(int)
    baseline_score = baseline_proba
    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_score)

    # Downsampling with averaging
    if use_downsampling:
        import os
        import sys

        from sklearn.model_selection import train_test_split
        from tqdm.auto import tqdm as tqdm_auto

        logger.info(f"\n  Downsampling: {n_iterations} iterations (1:1 balanced)")
        imbalance = max(np.bincount(y_train)) / min(np.bincount(y_train))
        logger.info(f"  Imbalance: {imbalance:.1f}:1 → 1:1")

        # Split training into train/val for threshold optimization
        inner_val_ratio = mlp_config.get("cv", {}).get("inner_val_ratio", 0.25)
        X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
            X_train_pca,
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
        tuning_enabled = mlp_config.get("tuning", {}).get("enabled", False)

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
                    ) = tune_hyperparameters_with_gridsearch(
                        X_train_balanced,
                        y_train_balanced,
                        mlp_config,
                        seed + fold_idx + iter_idx,
                    )
                finally:
                    sys.stdout.close()
                    sys.stdout = old_stdout
            else:
                model_cfg = mlp_config["model"].copy()

                # Convert hidden_layer_sizes from list to tuple (YAML parses as list)
                hidden_layers = model_cfg.get("hidden_layer_sizes", (128, 64))
                if isinstance(hidden_layers, list):
                    hidden_layers = tuple(hidden_layers)

                iter_model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation=model_cfg.get("activation", "relu"),
                    alpha=model_cfg.get("alpha", 0.001),
                    learning_rate=model_cfg.get("learning_rate", "adaptive"),
                    solver=model_cfg.get("solver", "adam"),
                    max_iter=model_cfg.get("max_iter", 1000),
                    early_stopping=model_cfg.get("early_stopping", True),
                    validation_fraction=model_cfg.get("validation_fraction", 0.1),
                    n_iter_no_change=model_cfg.get("n_iter_no_change", 10),
                    learning_rate_init=model_cfg.get("learning_rate_init", 0.001),
                    tol=model_cfg.get("tol", 0.0001),
                    batch_size=model_cfg.get("batch_size", "auto"),
                    beta_1=model_cfg.get("beta_1", 0.9),
                    beta_2=model_cfg.get("beta_2", 0.999),
                    random_state=seed + fold_idx + iter_idx,
                )
                iter_model.fit(X_train_balanced, y_train_balanced)
                iter_best_params = model_cfg

            # Predict on validation set (for threshold optimization)
            iter_val_proba = iter_model.predict_proba(X_inner_val)[:, 1]
            iter_test_proba = iter_model.predict_proba(X_test_pca)[:, 1]

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

            # WandB logging - track loss and metrics per iteration
            if use_wandb:
                import wandb

                global_step = fold_idx * n_iterations + iter_idx
                log_dict = {
                    "val/roc_auc": iter_roc_auc,
                    "val/bal_acc": iter_bal_acc,
                    "val/ppv": iter_ppv,
                    "val/pr_auc": iter_pr_auc,
                    "meta/fold": fold_idx,
                    "meta/iteration": iter_idx,
                }

                # Add loss curve info if available
                if hasattr(iter_model, "loss_curve_") and len(iter_model.loss_curve_) > 0:
                    log_dict["train/final_loss"] = iter_model.loss_curve_[-1]
                    log_dict["train/n_iter"] = iter_model.n_iter_
                    log_dict["train/converged"] = int(iter_model.n_iter_ < model_cfg.get("max_iter", 1000))

                wandb.log(log_dict, step=global_step)

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
        mlp_val_score = np.mean(all_val_probas, axis=0)
        mlp_raw_score = np.mean(all_test_probas, axis=0)
        mlp_model = iter_model  # Keep last model for reference

        # Use params from best performing iteration
        best_params = best_iter_params

        # Print hyperparameter distribution
        if tuning_enabled and len(all_iter_params) > 0:
            from collections import Counter

            logger.info(f"\n  Hyperparameter distribution across {n_iterations} iterations:")

            # count each unique parameter combination
            for param_name in [
                "hidden_layer_sizes",
                "alpha",
                "activation",
                "learning_rate",
            ]:
                if param_name in all_iter_params[0]:
                    values = [p.get(param_name) for p in all_iter_params if param_name in p]
                    if values:
                        # convert lists to tuples for hashability
                        hashable_values = [tuple(v) if isinstance(v, list) else v for v in values]
                        counts = Counter(hashable_values)
                        total = len(values)
                        logger.info(f"    {param_name}:")
                        for value, count in counts.most_common():
                            pct = 100 * count / total
                            logger.info(f"      {value}: {count}/{total} ({pct:.1f}%)")

        # Optimize threshold on VALIDATION SET (not test set!)
        if threshold_candidates:
            threshold_metric = mlp_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
            mlp_threshold, best_val_score = find_best_threshold(
                y_inner_val,
                mlp_val_score,
                threshold_candidates,
                default_threshold=0.5,
                metric=threshold_metric,
                beta=float(mlp_config.get("tuning", {}).get("fbeta_beta", 0.5)),
            )
            logger.info(
                f"  ✓ Averaged {n_iterations} models | Threshold (val {threshold_metric}={best_val_score:.3f}): {mlp_threshold:.3f}"
            )
        else:
            mlp_threshold = 0.5
            logger.info(f"  ✓ Averaged {n_iterations} models")

    # Hyperparameter tuning with GridSearchCV (if enabled)
    elif mlp_config.get("tuning", {}).get("enabled", False):
        best_params = {}
        (
            best_params,
            mlp_model,
            mlp_threshold,
        ) = tune_hyperparameters_with_gridsearch(X_train_pca, y_train, mlp_config, seed + fold_idx)
    else:
        # Use default hyperparameters from config
        model_cfg = mlp_config["model"].copy()

        # Convert hidden_layer_sizes from list to tuple (YAML parses as list)
        hidden_layers = model_cfg.get("hidden_layer_sizes", (128, 64))
        if isinstance(hidden_layers, list):
            hidden_layers = tuple(hidden_layers)

        mlp_model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=model_cfg.get("activation", "relu"),
            alpha=model_cfg.get("alpha", 0.001),
            learning_rate=model_cfg.get("learning_rate", "adaptive"),
            solver=model_cfg.get("solver", "adam"),
            max_iter=model_cfg.get("max_iter", 1000),
            early_stopping=model_cfg.get("early_stopping", True),
            validation_fraction=model_cfg.get("validation_fraction", 0.1),
            n_iter_no_change=model_cfg.get("n_iter_no_change", 10),
            learning_rate_init=model_cfg.get("learning_rate_init", 0.001),
            tol=model_cfg.get("tol", 0.0001),
            batch_size=model_cfg.get("batch_size", "auto"),
            beta_1=model_cfg.get("beta_1", 0.9),
            beta_2=model_cfg.get("beta_2", 0.999),
            random_state=seed + fold_idx,
        )
        mlp_model.fit(X_train_pca, y_train)
        mlp_threshold = mlp_config.get("evaluation", {}).get("decision_threshold", 0.5)

        # Optimize threshold using fallback CV
        if threshold_candidates:
            cv_folds = mlp_config.get("tuning", {}).get("cv_folds", 3)
            fallback_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + fold_idx)
            oof_scores = []
            oof_targets = []
            for inner_idx, (tr_idx, va_idx) in enumerate(fallback_cv.split(X_train_pca, y_train), 1):
                # Use the already converted hidden_layers from above
                temp_model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation=model_cfg.get("activation", "relu"),
                    alpha=model_cfg.get("alpha", 0.001),
                    learning_rate=model_cfg.get("learning_rate", "adaptive"),
                    solver=model_cfg.get("solver", "adam"),
                    max_iter=model_cfg.get("max_iter", 1000),
                    early_stopping=model_cfg.get("early_stopping", True),
                    validation_fraction=model_cfg.get("validation_fraction", 0.1),
                    n_iter_no_change=model_cfg.get("n_iter_no_change", 10),
                    learning_rate_init=model_cfg.get("learning_rate_init", 0.001),
                    tol=model_cfg.get("tol", 0.0001),
                    batch_size=model_cfg.get("batch_size", "auto"),
                    beta_1=model_cfg.get("beta_1", 0.9),
                    beta_2=model_cfg.get("beta_2", 0.999),
                    random_state=seed + fold_idx + inner_idx,
                )
                temp_model.fit(X_train_pca[tr_idx], y_train[tr_idx])
                oof_scores.append(temp_model.predict_proba(X_train_pca[va_idx])[:, 1])
                oof_targets.append(y_train[va_idx])

            if oof_scores:
                oof_scores_concat = np.concatenate(oof_scores)
                oof_targets_concat = np.concatenate(oof_targets)
                threshold_metric = mlp_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
                mlp_threshold, _ = find_best_threshold(
                    oof_targets_concat,
                    oof_scores_concat,
                    threshold_candidates,
                    default_threshold=mlp_threshold,
                    metric=threshold_metric,
                    beta=float(mlp_config.get("tuning", {}).get("fbeta_beta", 0.5)),
                )

    # Evaluate on test set
    logger.info(
        f"\n  Evaluating on test set ({len(y_test)} subjects: {np.bincount(y_test)[1]} clinical, {np.bincount(y_test)[0]} control)..."
    )

    # MLP predictions using native probabilities (or already computed for downsampling)
    if not use_downsampling:
        mlp_raw_score = mlp_model.predict_proba(X_test_pca)[:, 1]

    mlp_score = mlp_raw_score  # No Platt scaling needed for MLP
    mlp_pred = (mlp_score >= mlp_threshold).astype(int)
    mlp_threshold_display = mlp_threshold
    mlp_metrics = compute_metrics(y_test, mlp_pred, mlp_score)

    # Print test set performance
    from sklearn.metrics import precision_score, average_precision_score

    baseline_ppv = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_pr_auc = average_precision_score(y_test, baseline_score)
    mlp_ppv = precision_score(y_test, mlp_pred, zero_division=0)
    mlp_pr_auc = average_precision_score(y_test, mlp_score)

    # Format hyperparameters for display
    def format_params(params):
        if "downsampling" in params:
            return f"downsampling={params['downsampling']}"
        parts = []
        if "hidden_layer_sizes" in params:
            parts.append(f"hidden_layer_sizes={params['hidden_layer_sizes']}")
        if "alpha" in params:
            parts.append(f"alpha={params['alpha']}")
        if "activation" in params:
            parts.append(f"activation={params['activation']}")
        if "learning_rate" in params:
            parts.append(f"learning_rate={params['learning_rate']}")
        return ", ".join(parts) if parts else "default"

    logger.info("\n  ✓ Test Set Results:")
    logger.info(f"    Hyperparams: {format_params(best_params)}")
    logger.info(
        f"    Baseline (thr={baseline_threshold:.3f}): bal_acc={baseline_metrics['balanced_accuracy']:.3f}  ppv={baseline_ppv:.3f}  roc_auc={baseline_metrics.get('roc_auc', 0):.3f}  pr_auc={baseline_pr_auc:.3f}  ({baseline_pred.sum()}/{len(baseline_pred)} pred+)"
    )
    logger.info(
        f"    MLP (thr={mlp_threshold_display:.3f}):      bal_acc={mlp_metrics['balanced_accuracy']:.3f}  ppv={mlp_ppv:.3f}  roc_auc={mlp_metrics.get('roc_auc', 0):.3f}  pr_auc={mlp_pr_auc:.3f}  ({mlp_pred.sum()}/{len(mlp_pred)} pred+)"
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
        "mlp": {
            "model": mlp_model,
            "metrics": mlp_metrics,
            "y_pred": mlp_pred,
            "y_score": mlp_score,
            "y_test": y_test,
            "X_test_pca": X_test_pca,
            "pipeline": fitted_pipeline,
            "best_params": best_params,
            "threshold": mlp_threshold,
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
    mlp_config = env.configs.mlp
    group_col = env.configs.data["columns"]["mapping"]["research_group"]

    # Initialize WandB if enabled
    if use_wandb:
        import wandb

        if not sweep_mode:
            wandb.init(
                project="abcd-psychosis-mlp",
                name=f"{task_name}_seed{seed}",
                config={
                    "task": task_name,
                    "seed": seed,
                    "model": mlp_config["model"],
                    "use_pca": mlp_config.get("use_pca", True),
                    "downsampling": mlp_config.get("downsampling", {}),
                },
            )

    logger.info(f"\n{'='*60}")
    logger.info(f"Task: {task_name}")
    logger.info(f"{'='*60}")

    # Filter data for this task
    df_filtered, y = filter_task_data(full_df, task_config, group_col)
    logger.info(f"Total samples: {len(y)} | Class balance: {np.bincount(y)}")
    logger.info(f"Imbalance ratio: 1:{np.bincount(y)[0]/np.bincount(y)[1]:.1f}")

    # Create outer CV splitter (5-fold)
    outer_cv = get_cv_splitter(mlp_config, seed)

    # Storage for fold results
    baseline_folds = []
    mlp_folds = []

    # Outer CV loop - each fold gets 20% test
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(outer_cv.split(df_filtered, y), total=outer_cv.n_splits, desc="CV Folds")
    ):
        logger.info(f"\nFold {fold_idx + 1}/{outer_cv.n_splits}")

        # Split into train (80%) and test (20%)
        train_df = df_filtered.iloc[train_idx].reset_index(drop=True)
        test_df = df_filtered.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Run this fold (GridSearchCV will handle inner CV for tuning)
        fold_result = run_single_fold(env, train_df, test_df, y_train, y_test, fold_idx, seed, use_wandb)

        baseline_folds.append(fold_result["baseline"])
        mlp_folds.append(fold_result["mlp"])

    # Aggregate results across all folds
    baseline_agg = aggregate_cv_predictions(baseline_folds)
    mlp_agg = aggregate_cv_predictions(mlp_folds)

    # Global threshold sweep on aggregated predictions
    threshold_candidates = get_threshold_candidates(mlp_config)
    global_threshold = None
    global_threshold_metric = None
    if threshold_candidates:
        all_targets = np.concatenate([fold["y_test"] for fold in mlp_folds])
        all_scores = np.concatenate([fold["y_score"] for fold in mlp_folds])
        threshold_metric = mlp_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
        global_threshold, global_threshold_metric = find_best_threshold(
            all_targets,
            all_scores,
            threshold_candidates,
            default_threshold=mlp_config.get("evaluation", {}).get("decision_threshold", 0.0),
            metric=threshold_metric,
            beta=float(mlp_config.get("tuning", {}).get("fbeta_beta", 0.5)),
        )
        mlp_agg["global_threshold"] = global_threshold
        mlp_agg["global_threshold_metric"] = global_threshold_metric

    # Setup output directory
    data_dir = env.repo_root / "outputs" / env.configs.run["run_name"] / env.configs.run["run_id"] / f"seed_{seed}"
    mlp_dir = data_dir / "mlp" / task_name
    mlp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = mlp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {task_name}")
    logger.info(f"{'='*60}")
    logger.info("Baseline (Logistic Regression):")
    logger.info(f"  Overall Balanced Accuracy: {baseline_agg['overall']['balanced_accuracy']:.3f}")
    logger.info(f"  Overall ROC-AUC: {baseline_agg['overall'].get('roc_auc', 0):.3f}")
    logger.info("\nMLP:")
    logger.info(f"  Overall Balanced Accuracy: {mlp_agg['overall']['balanced_accuracy']:.3f}")
    logger.info(f"  Overall ROC-AUC: {mlp_agg['overall'].get('roc_auc', 0):.3f}")
    logger.info("\nPer-Fold Stats (Mean ± Std):")
    logger.info(
        f"  Balanced Accuracy: {mlp_agg['per_fold']['balanced_accuracy_mean']:.3f} ± {mlp_agg['per_fold']['balanced_accuracy_std']:.3f}"
    )
    logger.info(f"  ROC-AUC: {mlp_agg['per_fold']['roc_auc_mean']:.3f} ± {mlp_agg['per_fold']['roc_auc_std']:.3f}")
    if global_threshold is not None:
        threshold_metric_name = mlp_config.get("evaluation", {}).get("threshold_metric", "balanced_accuracy")
        logger.info(
            f"  Global threshold ({threshold_metric_name}): {global_threshold:.3f} → {global_threshold_metric:.3f}"
        )
    logger.info(f"{'='*60}\n")

    if not sweep_mode:
        # Generate visualizations on aggregated predictions
        all_y_true = np.concatenate([f["y_test"] for f in mlp_folds])
        all_y_pred_baseline = np.concatenate([f["y_pred"] for f in baseline_folds])
        all_y_pred_mlp = np.concatenate([f["y_pred"] for f in mlp_folds])

        # Confusion matrices
        cm_baseline = compute_confusion_matrix(all_y_true, all_y_pred_baseline)
        cm_mlp = compute_confusion_matrix(all_y_true, all_y_pred_mlp)

        plot_confusion_matrix(
            cm_baseline,
            ["Negative", "Positive"],
            f"Baseline - {task_name}",
            plots_dir / f"cm_baseline_{task_name}.png",
        )
        plot_confusion_matrix(
            cm_mlp,
            ["Negative", "Positive"],
            f"MLP - {task_name}",
            plots_dir / f"cm_mlp_{task_name}.png",
        )

        # Save all results (feature importance done separately)
        results = {
            "baseline": baseline_agg,
            "mlp": mlp_agg,
            "baseline_folds": baseline_folds,
            "mlp_folds": mlp_folds,
            "global_threshold": global_threshold,
        }
        with open(mlp_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Results saved to {mlp_dir}\n")

    if use_wandb:
        import wandb

        # Log overall metrics
        wandb.log(
            {
                f"{task_name}/overall_balanced_accuracy": mlp_agg["overall"]["balanced_accuracy"],
                f"{task_name}/overall_roc_auc": mlp_agg["overall"].get("roc_auc", 0),
                f"{task_name}/per_fold_balanced_accuracy_mean": mlp_agg["per_fold"]["balanced_accuracy_mean"],
                f"{task_name}/per_fold_balanced_accuracy_std": mlp_agg["per_fold"]["balanced_accuracy_std"],
            }
        )

    return {
        "baseline": baseline_agg,
        "mlp": mlp_agg,
        "global_threshold": global_threshold,
    }


def compute_feature_importance(env, full_df, task_config, mlp_folds, seed):
    """Compute feature importance and brain region mapping separately.

    Args:
        env: Environment with configs
        full_df: Full dataset
        task_config: Task configuration
        mlp_folds: List of MLP fold results
        seed: Random seed

    Returns:
        Dict with importance dataframes and plots saved
    """
    task_name = task_config["name"]
    mlp_config = env.configs.mlp
    group_col = env.configs.data["columns"]["mapping"]["research_group"]

    # Filter data for this task
    df_filtered, _ = filter_task_data(full_df, task_config, group_col)

    # Setup output directory
    data_dir = env.repo_root / "outputs" / env.configs.run["run_name"] / env.configs.run["run_id"] / f"seed_{seed}"
    mlp_dir = data_dir / "mlp" / task_name
    plots_dir = mlp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Use last fold as representative (could also aggregate across folds)
    last_fold = mlp_folds[-1]
    X_test_pca = last_fold["X_test_pca"]
    y_test = last_fold["y_test"]
    model = last_fold["model"]
    pipeline = last_fold["pipeline"]

    # Only compute feature importance if PCA was used
    if pipeline is not None:
        n_components = pipeline["n_components"]
        pca_features = [f"PC{i+1}" for i in range(n_components)]

        logger.info(f"\nComputing feature importance for {task_name}...")

        # Permutation importance
        mlp_importance = get_feature_importance_permutation(model, X_test_pca, y_test, pca_features, seed)

        plot_feature_importance(
            mlp_importance,
            f"MLP Feature Importance - {task_name}",
            plots_dir / f"importance_mlp_{task_name}.png",
            top_n=mlp_config.get("interpretation", {}).get("top_n_features", 20),
        )

        # Map to brain regions
        from ..tsne.embeddings import get_imaging_columns

        all_imaging_cols = get_imaging_columns(df_filtered, mlp_config["imaging_prefixes"])
        valid_features = pipeline["valid_features"]
        imaging_cols = [col for i, col in enumerate(all_imaging_cols) if valid_features[i]]

        brain_regions = map_pca_to_brain_regions(
            mlp_importance,
            pipeline["pca"],
            imaging_cols,
            top_n_components=mlp_config.get("interpretation", {}).get("top_n_features", 20),
            top_n_features=mlp_config.get("interpretation", {}).get("top_n_features", 20),
        )
        brain_regions_enriched = enrich_brain_regions(brain_regions, env)
        brain_regions_enriched.to_csv(mlp_dir / "brain_regions.csv", index=False)

        plot_feature_importance(
            brain_regions_enriched,
            f"Top Brain Regions - {task_name}",
            plots_dir / f"brain_regions_{task_name}.png",
            top_n=20,
        )

        logger.info(f"Feature importance saved to {mlp_dir}\n")

        return {
            "pca_importance": mlp_importance,
            "brain_regions": brain_regions_enriched,
        }
    else:
        logger.info("\nSkipping feature importance (PCA not used)")
        return None


def run_mlp_pipeline(env, use_wandb: bool = False, sweep_mode: bool = False):
    """Run complete MLP pipeline with nested CV (no fixed holdout test set)."""
    logger.info("=" * 60)
    logger.info("MLP Pipeline with Nested Cross-Validation")
    logger.info("=" * 60)
    logger.info("Loading full dataset for 5-fold CV...")

    full_df = load_full_dataset(env)
    logger.info(f"Total samples: {len(full_df)}")

    tasks = env.configs.mlp.get("tasks", [])
    all_results = {}

    for task_config in tasks:
        task_results = run_task_with_nested_cv(env, full_df, task_config, use_wandb, sweep_mode)
        all_results[task_config["name"]] = task_results

    logger.info("\n" + "=" * 60)
    logger.info("MLP Pipeline Complete")
    logger.info("=" * 60)

    return all_results
