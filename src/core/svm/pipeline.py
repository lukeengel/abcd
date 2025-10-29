"""SVM classification pipeline with nested CV (no fixed holdout test set)."""

import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .evaluation import (
    aggregate_cv_predictions,
    compute_confusion_matrix,
    compute_metrics,
    get_cv_splitter,
    find_best_threshold,
)
from .feature_mapping import enrich_brain_regions
from .interpretation import (
    get_feature_importance_permutation,
    map_pca_to_brain_regions,
)
from .models import create_baseline
from .preprocessing import apply_pca_to_fold, fit_pca_on_dev
from .visualization import plot_confusion_matrix, plot_feature_importance


def fit_platt_scaler(
    scores: np.ndarray, targets: np.ndarray
) -> LogisticRegression | None:
    """Fit Platt scaling on out-of-fold scores; return None if not feasible."""
    if scores is None or len(scores) == 0 or len(np.unique(targets)) < 2:
        return None

    scaler = LogisticRegression(solver="lbfgs", max_iter=1000)
    scaler.fit(scores.reshape(-1, 1), targets)
    return scaler


def apply_platt_scaler(
    scaler: LogisticRegression | None, raw_scores: np.ndarray
) -> np.ndarray:
    """Apply fitted Platt scaler to raw decision scores."""
    if scaler is None:
        return raw_scores
    calibrated = scaler.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    return calibrated


def load_full_dataset(env) -> pd.DataFrame:
    """Load all data for nested CV (train+val+test combined)."""
    run_cfg = env.configs.run
    data_dir = (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
        / "datasets"
    )

    # Load all splits and combine
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return full_df


def filter_task_data(
    df: pd.DataFrame, task_config: dict, group_col: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter data for specific classification task."""
    pos_classes = task_config.get("positive_classes") or [
        task_config.get("positive_class")
    ]
    neg_class = task_config["negative_class"]

    mask = df[group_col].isin(pos_classes + [neg_class])
    df_filtered = df[mask].copy()
    y = np.where(df_filtered[group_col].isin(pos_classes), 1, 0)

    return df_filtered, y


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


def tune_hyperparameters_with_gridsearch(X_train, y_train, svm_config, seed):
    """Tune SVM hyperparameters and derive a validation-informed threshold."""
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_score,
        roc_auc_score,
        fbeta_score,
        average_precision_score,
    )
    from sklearn.model_selection import ParameterGrid

    tuning_config = svm_config.get("tuning", {})
    param_grid = tuning_config.get("param_grid", {})
    scoring_metric = tuning_config.get("scoring", "balanced_accuracy")
    scoring_name = scoring_metric.lower()
    beta = float(tuning_config.get("fbeta_beta", 0.5))
    cv_folds = tuning_config.get("cv_folds", 3)

    decision_threshold = svm_config.get("evaluation", {}).get("decision_threshold", 0.5)
    threshold_candidates = get_threshold_candidates(svm_config)
    use_probability = svm_config["model"].get("probability", False)
    param_combinations = list(ParameterGrid(param_grid))
    inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    print(f"    Tuning with nested CV ({cv_folds} inner folds)...")
    print(f"    Scoring metric: {scoring_metric}")
    if threshold_candidates:
        print(f"    Threshold search grid: {threshold_candidates}")
    print(f"    Testing {len(param_combinations)} parameter combinations")
    print(
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

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(X_train, y_train), 1
        ):
            # Build model params - only add class_weight if not in grid search params
            model_params = params.copy()
            if "class_weight" not in model_params:
                model_params["class_weight"] = svm_config["model"].get(
                    "class_weight", "balanced"
                )
            model_params["max_iter"] = svm_config["model"].get("max_iter", -1)
            model_params["probability"] = use_probability
            model_params["random_state"] = seed + inner_fold

            model = SVC(**model_params)
            model.fit(X_train[inner_train_idx], y_train[inner_train_idx])

            y_val = y_train[inner_val_idx]
            if use_probability and hasattr(model, "predict_proba"):
                scores_for_threshold = model.predict_proba(X_train[inner_val_idx])[:, 1]
            else:
                scores_for_threshold = model.decision_function(X_train[inner_val_idx])

            if hasattr(model, "decision_function"):
                scores_for_roc = model.decision_function(X_train[inner_val_idx])
            else:
                scores_for_roc = scores_for_threshold

            if use_probability:
                y_val_pred = (scores_for_threshold >= decision_threshold).astype(int)
            else:
                y_val_pred = (scores_for_threshold >= 0.0).astype(int)

            bal_scores.append(balanced_accuracy_score(y_val, y_val_pred))
            roc_scores.append(roc_auc_score(y_val, scores_for_roc))
            pr_auc_scores.append(average_precision_score(y_val, scores_for_threshold))
            precision_scores.append(precision_score(y_val, y_val_pred, zero_division=0))
            fbeta_scores.append(
                fbeta_score(y_val, y_val_pred, beta=beta, zero_division=0)
            )

            val_scores_threshold.append(scores_for_threshold)
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
            f"    [{idx}/{len(param_combinations)}] {params_str:<50} "
            f"bal_acc={bal_acc_mean:.3f}±{bal_acc_std:.3f}  "
            f"pr_auc={pr_auc_mean:.3f}±{pr_auc_std:.3f}"
        )

        aggregate_targets = np.concatenate(val_targets)
        aggregate_threshold_scores = np.concatenate(val_scores_threshold)
        thresholds_for_metric = threshold_candidates or (
            [decision_threshold] if use_probability else [0.0]
        )

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
                    metric_val = precision_score(
                        aggregate_targets, y_pred_thr, zero_division=0
                    )
                elif scoring_name == "fbeta":
                    metric_val = fbeta_score(
                        aggregate_targets, y_pred_thr, beta=beta, zero_division=0
                    )
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
                "val_scores_threshold": aggregate_threshold_scores,
                "metric_threshold": selected_metric_threshold,
            }

    best_params = best_result["params"]
    best_metric_score = best_result["score"]

    # Use threshold metric from config if specified, otherwise use tuning metric if appropriate
    threshold_metric_config = svm_config.get("evaluation", {}).get("threshold_metric")
    if threshold_metric_config:
        metric_for_threshold = threshold_metric_config.lower()
    elif scoring_name in {"precision", "fbeta", "balanced_accuracy"}:
        metric_for_threshold = scoring_name
    else:
        metric_for_threshold = "balanced_accuracy"

    threshold_list = threshold_candidates if threshold_candidates else None

    calibrator = fit_platt_scaler(
        best_result["val_scores_threshold"], best_result["val_targets"]
    )
    calibration_scores = best_result["val_scores_threshold"]
    if calibrator is not None:
        calibration_scores = apply_platt_scaler(calibrator, calibration_scores)

    calibrated_threshold_metric = None
    if threshold_list:
        best_threshold, calibrated_threshold_metric = find_best_threshold(
            best_result["val_targets"],
            calibration_scores,
            threshold_list,
            default_threshold=decision_threshold if use_probability else 0.0,
            metric=metric_for_threshold,
            beta=beta,
        )
    else:
        best_threshold = decision_threshold if use_probability else 0.0

    print(
        f"\n    ✓ BEST (by {scoring_metric}): {best_params} → {scoring_metric}={best_metric_score:.3f}"
    )
    if threshold_candidates:
        print(
            f"    ✓ Selected threshold {best_threshold:.3f} "
            f"(validation {metric_for_threshold}={calibrated_threshold_metric:.3f})"
        )

    print("    Training final model with best params on full training data...")
    # Build final model params - only add class_weight if not in best_params
    final_params = best_params.copy()
    if "class_weight" not in final_params:
        final_params["class_weight"] = svm_config["model"].get(
            "class_weight", "balanced"
        )
    final_params["max_iter"] = svm_config["model"].get("max_iter", -1)
    final_params["probability"] = use_probability
    final_params["random_state"] = seed

    best_model = SVC(**final_params)
    best_model.fit(X_train, y_train)

    print("    ✓ Final model trained (Platt scaling from inner folds)")

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
    """Process a single outer CV fold.

    Steps:
    1. Fit PCA on train data (80% of full dataset)
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
        Dict with baseline and svm results
    """
    svm_config = env.configs.svm
    use_probability = svm_config["model"].get("probability", False)
    threshold_candidates = get_threshold_candidates(svm_config)

    print(
        f"  Train: {len(y_train)} (class: {np.bincount(y_train)}) | Test: {len(y_test)}"
    )

    # Fit PCA on train set only (no data leakage)
    fitted_pipeline = fit_pca_on_dev(train_df, env, seed + fold_idx)

    # Apply PCA to both sets
    X_train_pca, _ = apply_pca_to_fold(train_df, train_df, fitted_pipeline, env)
    X_test_pca, _ = apply_pca_to_fold(test_df, test_df, fitted_pipeline, env)

    # Train baseline (Logistic Regression with class_weight='balanced')
    decision_threshold = svm_config.get("evaluation", {}).get("decision_threshold", 0.5)
    inner_cv_splits = svm_config.get("tuning", {}).get("cv_folds", 3)

    baseline_threshold = decision_threshold
    if threshold_candidates:
        baseline_val_scores = []
        baseline_val_targets = []
        baseline_cv = StratifiedKFold(
            n_splits=inner_cv_splits, shuffle=True, random_state=seed + fold_idx
        )
        for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(
            baseline_cv.split(X_train_pca, y_train), 1
        ):
            temp_model = create_baseline(svm_config, seed + fold_idx + inner_idx)
            temp_model.fit(X_train_pca[inner_train_idx], y_train[inner_train_idx])
            baseline_val_scores.append(
                temp_model.predict_proba(X_train_pca[inner_val_idx])[:, 1]
            )
            baseline_val_targets.append(y_train[inner_val_idx])

        threshold_metric = svm_config.get("evaluation", {}).get(
            "threshold_metric", "balanced_accuracy"
        )
        baseline_threshold, _ = find_best_threshold(
            np.concatenate(baseline_val_targets),
            np.concatenate(baseline_val_scores),
            threshold_candidates,
            default_threshold=decision_threshold,
            metric=threshold_metric,
        )

    baseline_model = create_baseline(svm_config, seed + fold_idx)
    baseline_model.fit(X_train_pca, y_train)

    # Baseline predictions with optimized threshold
    baseline_proba = baseline_model.predict_proba(X_test_pca)[:, 1]
    baseline_pred = (baseline_proba >= baseline_threshold).astype(int)
    baseline_score = baseline_proba
    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_score)

    # Hyperparameter tuning with GridSearchCV (if enabled)
    best_params = {}
    svm_calibrator = None
    if svm_config.get("tuning", {}).get("enabled", False):
        (
            best_params,
            svm_model,
            svm_calibrator,
            svm_threshold,
        ) = tune_hyperparameters_with_gridsearch(
            X_train_pca, y_train, svm_config, seed + fold_idx
        )
    else:
        # Use default hyperparameters from config
        model_cfg = svm_config["model"].copy()
        svm_model = SVC(
            kernel=model_cfg.get("kernel", "linear"),
            C=model_cfg.get("C", 1.0),
            gamma=model_cfg.get("gamma", "scale"),
            class_weight=model_cfg.get("class_weight", "balanced"),
            max_iter=model_cfg.get("max_iter", -1),
            tol=model_cfg.get("tol", 0.001),
            probability=use_probability,
            random_state=seed + fold_idx,
        )
        svm_model.fit(X_train_pca, y_train)
        svm_threshold = (
            svm_config.get("evaluation", {}).get("decision_threshold", 0.5)
            if use_probability
            else 0.0
        )

        # Build Platt scaler from fallback CV
        cv_folds = svm_config.get("tuning", {}).get("cv_folds", 3)
        fallback_cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=seed + fold_idx
        )
        oof_scores = []
        oof_targets = []
        for inner_idx, (tr_idx, va_idx) in enumerate(
            fallback_cv.split(X_train_pca, y_train), 1
        ):
            temp_model = SVC(
                kernel=model_cfg.get("kernel", "linear"),
                C=model_cfg.get("C", 1.0),
                gamma=model_cfg.get("gamma", "scale"),
                class_weight=model_cfg.get("class_weight", "balanced"),
                max_iter=model_cfg.get("max_iter", -1),
                tol=model_cfg.get("tol", 0.001),
                probability=use_probability,
                random_state=seed + fold_idx + inner_idx,
            )
            temp_model.fit(X_train_pca[tr_idx], y_train[tr_idx])
            if use_probability and hasattr(temp_model, "predict_proba"):
                oof_scores.append(temp_model.predict_proba(X_train_pca[va_idx])[:, 1])
            else:
                oof_scores.append(temp_model.decision_function(X_train_pca[va_idx]))
            oof_targets.append(y_train[va_idx])

        if oof_scores:
            oof_scores_concat = np.concatenate(oof_scores)
            oof_targets_concat = np.concatenate(oof_targets)
            svm_calibrator = fit_platt_scaler(oof_scores_concat, oof_targets_concat)
            if svm_calibrator is not None and threshold_candidates:
                calibrated_oof = apply_platt_scaler(svm_calibrator, oof_scores_concat)
                threshold_metric = svm_config.get("evaluation", {}).get(
                    "threshold_metric", "balanced_accuracy"
                )
                svm_threshold, _ = find_best_threshold(
                    oof_targets_concat,
                    calibrated_oof,
                    threshold_candidates,
                    default_threshold=svm_threshold,
                    metric=threshold_metric,
                    beta=float(svm_config.get("tuning", {}).get("fbeta_beta", 0.5)),
                )

    # Evaluate on test set
    print(
        f"\n  Evaluating on test set ({len(y_test)} subjects: {np.bincount(y_test)[1]} clinical, {np.bincount(y_test)[0]} control)..."
    )

    # SVM predictions using Platt-calibrated scores
    if use_probability and hasattr(svm_model, "predict_proba"):
        svm_raw_score = svm_model.predict_proba(X_test_pca)[:, 1]
    elif hasattr(svm_model, "decision_function"):
        svm_raw_score = svm_model.decision_function(X_test_pca)
    else:
        raise AttributeError("SVM model lacks predict_proba and decision_function")

    svm_score = apply_platt_scaler(svm_calibrator, svm_raw_score)
    svm_pred = (svm_score >= svm_threshold).astype(int)
    svm_threshold_display = svm_threshold
    svm_metrics = compute_metrics(y_test, svm_pred, svm_score)

    # Print test set performance
    print("\n  ✓ Test Set Results:")
    print(
        f"    Baseline (threshold={baseline_threshold:.3f}): bal_acc={baseline_metrics['balanced_accuracy']:.3f}  roc_auc={baseline_metrics.get('roc_auc', 0):.3f}  ({baseline_pred.sum()}/{len(baseline_pred)} predicted positive)"
    )
    print(
        f"    SVM (threshold={svm_threshold_display:.3f}):       bal_acc={svm_metrics['balanced_accuracy']:.3f}  roc_auc={svm_metrics.get('roc_auc', 0):.3f}  ({svm_pred.sum()}/{len(svm_pred)} predicted positive)"
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
        "svm": {
            "model": svm_model,
            "calibrator": svm_calibrator,
            "metrics": svm_metrics,
            "y_pred": svm_pred,
            "y_score": svm_score,
            "y_test": y_test,
            "X_test_pca": X_test_pca,
            "pipeline": fitted_pipeline,
            "best_params": best_params,
            "threshold": svm_threshold,
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
    svm_config = env.configs.svm
    group_col = env.configs.data["columns"]["mapping"]["research_group"]

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    # Filter data for this task
    df_filtered, y = filter_task_data(full_df, task_config, group_col)
    print(f"Total samples: {len(y)} | Class balance: {np.bincount(y)}")
    print(f"Imbalance ratio: 1:{np.bincount(y)[0]/np.bincount(y)[1]:.1f}")

    # Create outer CV splitter (5-fold)
    outer_cv = get_cv_splitter(svm_config, seed)

    # Storage for fold results
    baseline_folds = []
    svm_folds = []

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
        fold_result = run_single_fold(
            env, train_df, test_df, y_train, y_test, fold_idx, seed
        )

        baseline_folds.append(fold_result["baseline"])
        svm_folds.append(fold_result["svm"])

    # Aggregate results across all folds
    baseline_agg = aggregate_cv_predictions(baseline_folds)
    svm_agg = aggregate_cv_predictions(svm_folds)

    # Global threshold sweep on aggregated predictions
    threshold_candidates = get_threshold_candidates(svm_config)
    global_threshold = None
    global_threshold_metric = None
    if threshold_candidates:
        all_targets = np.concatenate([fold["y_test"] for fold in svm_folds])
        all_scores = np.concatenate([fold["y_score"] for fold in svm_folds])
        threshold_metric = svm_config.get("evaluation", {}).get(
            "threshold_metric", "balanced_accuracy"
        )
        global_threshold, global_threshold_metric = find_best_threshold(
            all_targets,
            all_scores,
            threshold_candidates,
            default_threshold=svm_config.get("evaluation", {}).get(
                "decision_threshold", 0.0
            ),
            metric=threshold_metric,
            beta=float(svm_config.get("tuning", {}).get("fbeta_beta", 0.5)),
        )
        svm_agg["global_threshold"] = global_threshold
        svm_agg["global_threshold_metric"] = global_threshold_metric

    # Setup output directory
    data_dir = (
        env.repo_root
        / "outputs"
        / env.configs.run["run_name"]
        / env.configs.run["run_id"]
        / f"seed_{seed}"
    )
    svm_dir = data_dir / "svm" / task_name
    svm_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = svm_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {task_name}")
    print(f"{'='*60}")
    print("Baseline (Logistic Regression):")
    print(
        f"  Overall Balanced Accuracy: {baseline_agg['overall']['balanced_accuracy']:.3f}"
    )
    print(f"  Overall ROC-AUC: {baseline_agg['overall'].get('roc_auc', 0):.3f}")
    print("\nSVM:")
    print(f"  Overall Balanced Accuracy: {svm_agg['overall']['balanced_accuracy']:.3f}")
    print(f"  Overall ROC-AUC: {svm_agg['overall'].get('roc_auc', 0):.3f}")
    print("\nPer-Fold Stats (Mean ± Std):")
    print(
        f"  Balanced Accuracy: {svm_agg['per_fold']['balanced_accuracy_mean']:.3f} ± {svm_agg['per_fold']['balanced_accuracy_std']:.3f}"
    )
    print(
        f"  ROC-AUC: {svm_agg['per_fold']['roc_auc_mean']:.3f} ± {svm_agg['per_fold']['roc_auc_std']:.3f}"
    )
    if global_threshold is not None:
        threshold_metric_name = svm_config.get("evaluation", {}).get(
            "threshold_metric", "balanced_accuracy"
        )
        print(
            f"  Global threshold ({threshold_metric_name}): {global_threshold:.3f} → {global_threshold_metric:.3f}"
        )
    print(f"{'='*60}\n")

    if not sweep_mode:
        # Generate visualizations on aggregated predictions
        all_y_true = np.concatenate([f["y_test"] for f in svm_folds])
        all_y_pred_baseline = np.concatenate([f["y_pred"] for f in baseline_folds])
        all_y_pred_svm = np.concatenate([f["y_pred"] for f in svm_folds])

        # Confusion matrices
        cm_baseline = compute_confusion_matrix(all_y_true, all_y_pred_baseline)
        cm_svm = compute_confusion_matrix(all_y_true, all_y_pred_svm)

        plot_confusion_matrix(
            cm_baseline,
            ["Negative", "Positive"],
            f"Baseline - {task_name}",
            plots_dir / f"cm_baseline_{task_name}.png",
        )
        plot_confusion_matrix(
            cm_svm,
            ["Negative", "Positive"],
            f"SVM - {task_name}",
            plots_dir / f"cm_svm_{task_name}.png",
        )

        # Save all results (feature importance done separately)
        results = {
            "baseline": baseline_agg,
            "svm": svm_agg,
            "baseline_folds": baseline_folds,
            "svm_folds": svm_folds,
            "global_threshold": global_threshold,
        }
        with open(svm_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {svm_dir}\n")

    if use_wandb:
        import wandb

        # Log overall metrics
        wandb.log(
            {
                f"{task_name}/overall_balanced_accuracy": svm_agg["overall"][
                    "balanced_accuracy"
                ],
                f"{task_name}/overall_roc_auc": svm_agg["overall"].get("roc_auc", 0),
                f"{task_name}/per_fold_balanced_accuracy_mean": svm_agg["per_fold"][
                    "balanced_accuracy_mean"
                ],
                f"{task_name}/per_fold_balanced_accuracy_std": svm_agg["per_fold"][
                    "balanced_accuracy_std"
                ],
            }
        )

    return {
        "baseline": baseline_agg,
        "svm": svm_agg,
        "global_threshold": global_threshold,
    }


def compute_feature_importance(env, full_df, task_config, svm_folds, seed):
    """Compute feature importance and brain region mapping separately.

    Args:
        env: Environment with configs
        full_df: Full dataset
        task_config: Task configuration
        svm_folds: List of SVM fold results
        seed: Random seed

    Returns:
        Dict with importance dataframes and plots saved
    """
    task_name = task_config["name"]
    svm_config = env.configs.svm
    group_col = env.configs.data["columns"]["mapping"]["research_group"]

    # Filter data for this task
    df_filtered, _ = filter_task_data(full_df, task_config, group_col)

    # Setup output directory
    data_dir = (
        env.repo_root
        / "outputs"
        / env.configs.run["run_name"]
        / env.configs.run["run_id"]
        / f"seed_{seed}"
    )
    svm_dir = data_dir / "svm" / task_name
    plots_dir = svm_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Use last fold as representative (could also aggregate across folds)
    last_fold = svm_folds[-1]
    X_test_pca = last_fold["X_test_pca"]
    y_test = last_fold["y_test"]
    model = last_fold["model"]
    pipeline = last_fold["pipeline"]

    n_components = pipeline["n_components"]
    pca_features = [f"PC{i+1}" for i in range(n_components)]

    print(f"\nComputing feature importance for {task_name}...")

    # Permutation importance
    svm_importance = get_feature_importance_permutation(
        model, X_test_pca, y_test, pca_features, seed
    )

    plot_feature_importance(
        svm_importance,
        f"SVM Feature Importance - {task_name}",
        plots_dir / f"importance_svm_{task_name}.png",
        top_n=svm_config.get("interpretation", {}).get("top_n_pcs", 10),
    )

    # Map to brain regions
    from ..tsne.embeddings import get_imaging_columns

    all_imaging_cols = get_imaging_columns(df_filtered, svm_config["imaging_prefixes"])
    valid_features = pipeline["valid_features"]
    imaging_cols = [col for i, col in enumerate(all_imaging_cols) if valid_features[i]]

    brain_regions = map_pca_to_brain_regions(
        svm_importance,
        pipeline["pca"],
        imaging_cols,
        top_n_components=svm_config.get("interpretation", {}).get("top_n_pcs", 10),
        top_n_features=svm_config.get("interpretation", {}).get("top_n_features", 20),
    )
    brain_regions_enriched = enrich_brain_regions(brain_regions, env)
    brain_regions_enriched.to_csv(svm_dir / "brain_regions.csv", index=False)

    plot_feature_importance(
        brain_regions_enriched,
        f"Top Brain Regions - {task_name}",
        plots_dir / f"brain_regions_{task_name}.png",
        top_n=20,
    )

    print(f"Feature importance saved to {svm_dir}\n")

    return {
        "pca_importance": svm_importance,
        "brain_regions": brain_regions_enriched,
    }


def run_svm_pipeline(env, use_wandb: bool = False, sweep_mode: bool = False):
    """Run complete SVM pipeline with nested CV (no fixed holdout test set)."""
    print("=" * 60)
    print("SVM Pipeline with Nested Cross-Validation")
    print("=" * 60)
    print("Loading full dataset for 5-fold CV...")

    full_df = load_full_dataset(env)
    print(f"Total samples: {len(full_df)}")

    tasks = env.configs.svm.get("tasks", [])
    all_results = {}

    for task_config in tasks:
        task_results = run_task_with_nested_cv(
            env, full_df, task_config, use_wandb, sweep_mode
        )
        all_results[task_config["name"]] = task_results

    print("\n" + "=" * 60)
    print("SVM Pipeline Complete")
    print("=" * 60)

    return all_results
