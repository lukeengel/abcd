"""Random Forest classification pipeline with nested CV."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..svm.evaluation import compute_confusion_matrix, compute_metrics, get_cv_splitter
from ..svm.preprocessing import apply_pca_to_fold, fit_pca_on_dev
from ..svm.visualization import plot_confusion_matrix
from .models import create_baseline, create_random_forest


def load_development_data(env) -> tuple[pd.DataFrame, Path]:
    """Load and combine train + val splits into 90% development set."""
    run_cfg = env.configs.run
    data_dir = (
        env.repo_root
        / "outputs"
        / run_cfg["run_name"]
        / run_cfg["run_id"]
        / f"seed_{run_cfg['seed']}"
        / "datasets"
    )

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    return dev_df, data_dir


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


def run_nested_cv(
    df: pd.DataFrame,
    y: np.ndarray,
    model,
    env,
    seed: int,
    fitted_pipeline: dict = None,
    use_wandb: bool = False,
    use_random_sampling: bool = False,
) -> dict:
    """Run nested cross-validation with pre-fitted PCA applied to all folds.

    Args:
        df: Development dataframe
        y: Target labels
        model: Base classifier (RandomForest or LogisticRegression)
        env: Environment with configs
        seed: Random seed
        fitted_pipeline: Pre-fitted PCA pipeline
        use_wandb: Whether to log to W&B
        use_random_sampling: If True, apply 1:1 balanced sampling within each CV fold
    """
    cv = get_cv_splitter(env.configs.randomforest, seed)
    rf_config = env.configs.randomforest
    n_iterations = rf_config.get("n_iterations", 20) if use_random_sampling else 1
    # For CV: always use threshold=0.5 for meaningful metrics (probabilities calibrated to balanced data)
    # Custom threshold only applies to final test set (imbalanced data)
    threshold = 0.5
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(cv.split(df, y), total=cv.n_splits, desc="CV folds")
    ):
        train_df_fold = df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = df.iloc[val_idx].reset_index(drop=True)
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Apply pre-fitted PCA pipeline
        X_train_pca, X_val_pca = apply_pca_to_fold(
            train_df_fold, val_df_fold, fitted_pipeline, env
        )
        pipeline = fitted_pipeline

        # Random sampling iterations within this fold
        iteration_results = []
        rng = np.random.RandomState(seed + fold_idx)

        for iter_idx in range(n_iterations):
            if use_random_sampling:
                # Apply 1:1 balanced sampling
                X_train_balanced, y_train_balanced = _balance_classes(
                    X_train_pca, y_train_fold, rng
                )
            else:
                X_train_balanced = X_train_pca
                y_train_balanced = y_train_fold

            # Train model on balanced data
            from sklearn.base import clone

            model_iter = clone(model)
            model_iter.fit(X_train_balanced, y_train_balanced)

            # Evaluate on validation set
            if hasattr(model_iter, "predict_proba"):
                y_score_val = model_iter.predict_proba(X_val_pca)[:, 1]
            else:
                y_score_val = None

            iteration_results.append(
                {
                    "y_score": y_score_val,
                    "model": model_iter,
                }
            )

        # Aggregate predictions across iterations (ensemble averaging)
        if use_random_sampling:
            # Average probability scores across all iterations
            all_scores = np.array([r["y_score"] for r in iteration_results])
            y_score_val = all_scores.mean(axis=0)
            y_pred_val = (y_score_val >= threshold).astype(int)
            # Use the last model for storage (all are trained similarly)
            final_model = iteration_results[-1]["model"]
        else:
            # For non-sampling (baseline), use model's default predict (threshold=0.5)
            y_score_val = iteration_results[0]["y_score"]
            y_pred_val = iteration_results[0]["model"].predict(X_val_pca)
            final_model = iteration_results[0]["model"]

        # Compute metrics on ensemble predictions
        metrics = compute_metrics(y_val_fold, y_pred_val, y_score_val)
        metrics["n_components"] = pipeline["n_components"]
        metrics["variance_explained"] = pipeline["variance_explained"]
        metrics["n_iterations"] = n_iterations

        # Store fold data
        import copy

        fold_results.append(
            {
                "metrics": metrics,
                "model": copy.deepcopy(final_model),
                "X_val_pca": X_val_pca,
                "y_val": y_val_fold,
                "pipeline": pipeline,
            }
        )

    # Aggregate metrics across folds
    aggregated = {}
    for key in fold_results[0]["metrics"].keys():
        values = [fold["metrics"][key] for fold in fold_results]
        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)

    if use_wandb:
        import wandb

        wandb.log({f"cv_{k}": v for k, v in aggregated.items()})

    return {"fold_results": fold_results, "aggregated": aggregated}


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


def run_final_model(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_dev: np.ndarray,
    y_test: np.ndarray,
    model,
    env,
    seed: int,
    task_name: str,
    rf_dir: Path,
    fitted_pipeline: dict = None,
    save_model: bool = True,
    use_random_sampling: bool = False,
) -> dict:
    """Train final model on full dev set, evaluate on test set."""
    # Apply pre-fitted PCA to dev and test
    X_dev_pca, X_test_pca = apply_pca_to_fold(dev_df, test_df, fitted_pipeline, env)
    pipeline = fitted_pipeline

    if use_random_sampling:
        # Train ensemble of models on balanced subsets (SAME AS CV)
        rf_config = env.configs.randomforest
        n_iterations = rf_config.get("n_iterations", 20)
        threshold = rf_config.get("evaluation", {}).get("decision_threshold", 0.5)

        print(
            f"\nTraining final Random Forest on full dev set ({len(y_dev)} samples)..."
        )
        print(
            f"  Using random sampling with {n_iterations} iterations (1:1 balanced subsets)"
        )

        rng = np.random.RandomState(seed)
        all_models = []

        for i in range(n_iterations):
            # Balance classes for this iteration
            X_balanced, y_balanced = _balance_classes(X_dev_pca, y_dev, rng)

            # Train model on balanced subset
            from sklearn.base import clone

            model_iter = clone(model)
            model_iter.fit(X_balanced, y_balanced)
            all_models.append(model_iter)

        # Ensemble predictions: average probabilities across all models (SAME AS CV)
        all_probas = np.array([m.predict_proba(X_test_pca)[:, 1] for m in all_models])
        y_score_test = all_probas.mean(axis=0)
        y_pred_test = (y_score_test >= threshold).astype(int)

        # Save all models for reproducibility
        model = all_models
    else:
        # Train single model on full dev set
        print(f"\nTraining final model on full dev set ({len(y_dev)} samples)...")
        model.fit(X_dev_pca, y_dev)

        # Get probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_score_test = model.predict_proba(X_test_pca)[:, 1]
        else:
            y_score_test = None

        y_pred_test = model.predict(X_test_pca)

    test_metrics = compute_metrics(y_test, y_pred_test, y_score_test)

    # Save model and pipeline
    if save_model:
        with open(rf_dir / f"model_{task_name}.pkl", "wb") as f:
            pickle.dump({"model": model, "pipeline": pipeline}, f)

    return {
        "test_metrics": test_metrics,
        "model": model,
        "pipeline": pipeline,
        "X_test_pca": X_test_pca,
        "y_test": y_test,
        "y_pred": y_pred_test,
    }


def run_task(
    env,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_config: dict,
    use_wandb: bool = False,
):
    """Run single classification task with nested CV and final model."""
    task_name = task_config["name"]
    seed = env.configs.run["seed"]
    rf_config = env.configs.randomforest
    group_col = env.configs.data["columns"]["mapping"]["research_group"]

    # Filter data
    dev_filtered, y_dev = filter_task_data(dev_df, task_config, group_col)
    test_filtered, y_test = filter_task_data(test_df, task_config, group_col)

    # Setup output directory
    data_dir = (
        env.repo_root
        / "outputs"
        / env.configs.run["run_name"]
        / env.configs.run["run_id"]
        / f"seed_{seed}"
    )
    rf_dir = data_dir / "randomforest" / task_name
    rf_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = rf_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Dev: {len(y_dev)} | Test: {len(y_test)}")
    print(f"{'='*60}\n")

    # Fit PCA once on full dev set
    print("Fitting PCA on full dev set...")
    fitted_pipeline = fit_pca_on_dev(dev_filtered, env, seed)
    print(f"✓ PCA components: {fitted_pipeline['n_components']}")
    print(f"✓ Variance explained: {fitted_pipeline['variance_explained']:.2%}")

    # Get decision threshold from config
    threshold = rf_config.get("evaluation", {}).get("decision_threshold", 0.5)
    print(f"✓ Decision threshold: {threshold}")

    # Hyperparameter tuning (optional)
    tuning_config = rf_config.get("tuning", {})
    if tuning_config.get("enabled", False):
        from ..tuning import tune_hyperparameters
        from sklearn.ensemble import RandomForestClassifier

        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING ENABLED")
        print("=" * 60)

        # Apply PCA to dev set for tuning
        X_dev_pca = fitted_pipeline["pca"].transform(
            fitted_pipeline["scaler"].transform(
                dev_filtered[env.configs.pca["imaging_prefixes"]].values
            )
        )

        # Create base model for tuning
        base_model = RandomForestClassifier(
            class_weight=None
            if rf_config.get("use_random_sampling", False)
            else "balanced_subsample",
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=seed,
        )

        # Tune hyperparameters
        best_params, tuning_results = tune_hyperparameters(
            X_dev_pca,
            y_dev,
            base_model,
            param_grid=tuning_config.get("param_grid", {}),
            seed=seed,
            n_folds=tuning_config.get("cv_folds", 5),
            use_random_sampling=rf_config.get("use_random_sampling", False),
            n_iterations=rf_config.get("n_iterations", 20),
            scoring=tuning_config.get("scoring", "roc_auc"),
            model_type="rf",
        )

        # Update model config with best parameters
        rf_config["model"].update(best_params)

        print(f"\n{'='*60}")
        print("APPLYING BEST HYPERPARAMETERS TO FINAL MODELS")
        print(f"{'='*60}")
        print("Best parameters found (will be used for CV and test):")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"{'='*60}\n")

        # Save tuning results
        import pickle

        with open(rf_dir / "tuning_results.pkl", "wb") as f:
            pickle.dump({"best_params": best_params, "all_results": tuning_results}, f)
        print(f"✓ Tuning results saved to: {rf_dir / 'tuning_results.pkl'}\n")

    results = {}

    # Baseline: Logistic Regression (no random sampling)
    print("\nRunning baseline with nested CV...")
    baseline = create_baseline(rf_config, seed)
    baseline_cv = run_nested_cv(
        dev_filtered,
        y_dev,
        baseline,
        env,
        seed,
        fitted_pipeline=fitted_pipeline,
        use_wandb=use_wandb,
        use_random_sampling=False,
    )
    baseline_final = run_final_model(
        dev_filtered,
        test_filtered,
        y_dev,
        y_test,
        baseline,
        env,
        seed,
        f"baseline_{task_name}",
        rf_dir,
        fitted_pipeline,
        use_random_sampling=False,  # Baseline never uses custom threshold
    )
    results["baseline"] = {**baseline_cv, **baseline_final}

    # Random Forest with optional random sampling
    use_random_sampling = rf_config.get("use_random_sampling", False)
    n_iterations = rf_config.get("n_iterations", 20)

    if use_random_sampling:
        print(
            f"\nRunning Random Forest with 1:1 balanced sampling within each CV fold ({n_iterations} iterations per fold)..."
        )
        # Create base RF WITHOUT class_weight (data will be balanced via sampling)
        from sklearn.ensemble import RandomForestClassifier

        model_cfg = rf_config.get("model", {})
        rf = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            max_features=model_cfg.get("max_features", "sqrt"),
            class_weight=None,  # No class weighting - data is manually balanced
            bootstrap=model_cfg.get("bootstrap", True),
            oob_score=model_cfg.get("oob_score", False),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=seed,
        )
        print("\nRandom Forest Configuration (used for CV and test):")
        print(f"  n_estimators: {rf.n_estimators}")
        print(f"  max_depth: {rf.max_depth}")
        print(f"  min_samples_split: {rf.min_samples_split}")
        print(f"  min_samples_leaf: {rf.min_samples_leaf}")
        print(f"  max_features: {rf.max_features}")
        print(f"  class_weight: {rf.class_weight}")
        print(f"  Random sampling iterations: {n_iterations}")
        # Pass use_random_sampling=True to run_nested_cv
        rf_cv = run_nested_cv(
            dev_filtered,
            y_dev,
            rf,
            env,
            seed,
            fitted_pipeline=fitted_pipeline,
            use_wandb=use_wandb,
            use_random_sampling=True,
        )
        # For final model, use NO class weight (threshold handles imbalance)
        rf_final_model = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            max_features=model_cfg.get("max_features", "sqrt"),
            class_weight=None,  # No class weighting - custom threshold handles imbalance
            bootstrap=model_cfg.get("bootstrap", True),
            oob_score=model_cfg.get("oob_score", False),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=seed,
        )
    else:
        print("\nRunning Random Forest with nested CV...")
        rf = create_random_forest(rf_config, seed)
        rf_cv = run_nested_cv(
            dev_filtered,
            y_dev,
            rf,
            env,
            seed,
            fitted_pipeline=fitted_pipeline,
            use_wandb=use_wandb,
            use_random_sampling=False,
        )
        rf_final_model = rf

    # Pass use_random_sampling flag to final model for threshold handling
    rf_final = run_final_model(
        dev_filtered,
        test_filtered,
        y_dev,
        y_test,
        rf_final_model,
        env,
        seed,
        f"rf_{task_name}",
        rf_dir,
        fitted_pipeline,
        use_random_sampling=use_random_sampling,
    )
    results["randomforest"] = {**rf_cv, **rf_final}

    # Visualizations
    rf_final["X_test_pca"]
    y_pred_baseline = baseline_final["y_pred"]
    y_pred_rf = rf_final["y_pred"]

    cm_baseline = compute_confusion_matrix(y_test, y_pred_baseline)
    cm_rf = compute_confusion_matrix(y_test, y_pred_rf)

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
    with open(rf_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    if use_wandb:
        import wandb

        wandb.log(
            {
                f"{task_name}/cv_balanced_accuracy": rf_cv["aggregated"][
                    "balanced_accuracy_mean"
                ],
                f"{task_name}/cv_roc_auc": rf_cv["aggregated"].get("roc_auc_mean", 0),
            }
        )

    print(f"\nResults saved to {rf_dir}\n")
    return results


def run_randomforest_pipeline(env, use_wandb: bool = False):
    """Run complete Random Forest pipeline across all tasks."""
    print("Loading development (train+val) and test data...")
    dev_df, data_dir = load_development_data(env)
    test_df = pd.read_parquet(data_dir / "test.parquet")

    print(f"Dev set: {len(dev_df)} | Test set: {len(test_df)}")

    tasks = env.configs.randomforest.get("tasks", [])
    all_results = {}

    for task_config in tqdm(tasks, desc="Classification tasks", unit="task"):
        task_results = run_task(env, dev_df, test_df, task_config, use_wandb)
        all_results[task_config["name"]] = task_results

    print("\n" + "=" * 60)
    print("Random Forest Pipeline Complete")
    print("=" * 60)

    return all_results
