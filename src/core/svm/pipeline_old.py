"""SVM classification pipeline with nested CV and W&B integration."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .evaluation import compute_confusion_matrix, compute_metrics, get_cv_splitter
from .feature_mapping import enrich_brain_regions
from .interpretation import (
    get_feature_importance_permutation,
    map_pca_to_brain_regions,
)
from .models import create_baseline, create_svm
from .preprocessing import apply_pca_to_fold
from .visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
)


def load_development_data(env) -> pd.DataFrame:
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
        model: Base classifier (SVC or LogisticRegression)
        env: Environment with configs
        seed: Random seed
        fitted_pipeline: Pre-fitted PCA pipeline
        use_wandb: Whether to log to W&B
        use_random_sampling: If True, apply 1:1 balanced sampling within each CV fold
    """
    cv = get_cv_splitter(env.configs.svm, seed)
    svm_config = env.configs.svm
    n_iterations = svm_config.get("n_iterations", 20) if use_random_sampling else 1
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(cv.split(df, y), total=cv.n_splits, desc="CV folds")
    ):
        train_df_fold = df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = df.iloc[val_idx].reset_index(drop=True)
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Apply pre-fitted PCA pipeline (same transformation for all folds)
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
            y_pred_val = model_iter.predict(X_val_pca)
            y_score_val = (
                model_iter.decision_function(X_val_pca)
                if hasattr(model_iter, "decision_function")
                else None
            )

            iteration_results.append(
                {
                    "y_pred": y_pred_val,
                    "y_score": y_score_val,
                    "model": model_iter,
                }
            )

        # Aggregate predictions across iterations (ensemble averaging)
        if use_random_sampling:
            # Average decision scores across all iterations
            all_scores = np.array([r["y_score"] for r in iteration_results])
            y_score_val = all_scores.mean(axis=0)
            y_pred_val = (y_score_val > 0).astype(int)
            # Use the last model for storage (all are trained similarly)
            final_model = iteration_results[-1]["model"]
        else:
            y_pred_val = iteration_results[0]["y_pred"]
            y_score_val = iteration_results[0]["y_score"]
            final_model = iteration_results[0]["model"]

        # Compute metrics on ensemble predictions
        metrics = compute_metrics(y_val_fold, y_pred_val, y_score_val)
        metrics["n_components"] = pipeline["n_components"]
        metrics["variance_explained"] = pipeline["variance_explained"]
        metrics["n_iterations"] = n_iterations

        # Store fold data for feature importance
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
    svm_dir: Path,
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
        svm_config = env.configs.svm
        n_iterations = svm_config.get("n_iterations", 20)

        print(f"\nTraining final SVM on full dev set ({len(y_dev)} samples)...")
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

        # Ensemble predictions: average decision scores across all models (SAME AS CV)
        if hasattr(all_models[0], "decision_function"):
            all_scores = np.array([m.decision_function(X_test_pca) for m in all_models])
            y_score_test = all_scores.mean(axis=0)
        else:
            all_probas = np.array(
                [m.predict_proba(X_test_pca)[:, 1] for m in all_models]
            )
            y_score_test = all_probas.mean(axis=0)

        y_pred_test = (y_score_test >= 0).astype(int)  # SVM uses 0 as decision boundary

        # Save all models for reproducibility
        model = all_models
    else:
        # Train single model on full dev set
        print(f"\nTraining final SVM on full dev set ({len(y_dev)} samples)...")
        model.fit(X_dev_pca, y_dev)

        # Evaluate on test
        y_pred_test = model.predict(X_test_pca)
        y_score_test = (
            model.decision_function(X_test_pca)
            if hasattr(model, "decision_function")
            else None
        )

    test_metrics = compute_metrics(y_test, y_pred_test, y_score_test)

    # Save model and pipeline (skip in sweep mode)
    if save_model:
        with open(svm_dir / f"model_{task_name}.pkl", "wb") as f:
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
    sweep_mode: bool = False,
):
    """Run single classification task with nested CV and final model."""
    task_name = task_config["name"]
    seed = env.configs.run["seed"]
    svm_config = env.configs.svm
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
    svm_dir = data_dir / "svm" / task_name
    svm_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = svm_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Dev: {len(y_dev)} | Test: {len(y_test)}")
    print(f"{'='*60}\n")

    # Hyperparameter tuning (optional)
    tuning_config = svm_config.get("tuning", {})
    if tuning_config.get("enabled", False):
        from ..tuning import tune_hyperparameters
        from sklearn.svm import SVC

        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING ENABLED")
        print("=" * 60)

        # Fit PCA on dev set for tuning
        fitted_pipeline = fit_pca_on_dev(dev_filtered, env, seed)
        X_dev_pca = fitted_pipeline["pca"].transform(
            fitted_pipeline["scaler"].transform(
                dev_filtered[
                    [
                        col
                        for col in dev_filtered.columns
                        if any(
                            col.startswith(p)
                            for p in env.configs.svm["imaging_prefixes"]
                        )
                    ]
                ].values
            )
        )

        # Create base model for tuning
        base_model = SVC(
            class_weight=None
            if svm_config.get("use_random_sampling", False)
            else "balanced",
            max_iter=-1,
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
            use_random_sampling=svm_config.get("use_random_sampling", False),
            n_iterations=svm_config.get("n_iterations", 20),
            scoring=tuning_config.get("scoring", "roc_auc"),
            model_type="svm",
        )

        # Update model config with best parameters
        svm_config["model"].update(best_params)

        print(f"\n{'='*60}")
        print("APPLYING BEST HYPERPARAMETERS TO FINAL MODELS")
        print(f"{'='*60}")
        print("Best parameters found (will be used for CV and test):")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"{'='*60}\n")

        # Save tuning results
        import pickle

        with open(svm_dir / "tuning_results.pkl", "wb") as f:
            pickle.dump({"best_params": best_params, "all_results": tuning_results}, f)
        print(f"✓ Tuning results saved to: {svm_dir / 'tuning_results.pkl'}\n")

    results = {}

    # Baseline: Logistic Regression (no random sampling)
    print("Running baseline with nested CV...")
    baseline = create_baseline(svm_config, seed)
    baseline_cv = run_nested_cv(
        dev_filtered,
        y_dev,
        baseline,
        env,
        seed,
        fitted_pipeline=None,
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
        svm_dir,
        save_model=not sweep_mode,
    )
    results["baseline"] = {**baseline_cv, **baseline_final}

    # SVM with optional random sampling
    use_random_sampling = svm_config.get("use_random_sampling", False)
    n_iterations = svm_config.get("n_iterations", 20)

    if use_random_sampling:
        print(
            f"\nRunning SVM with 1:1 balanced sampling within each CV fold ({n_iterations} iterations per fold)..."
        )
        # Create base SVM WITHOUT class_weight (data will be balanced via sampling)
        model_cfg = svm_config.get("model", {})
        svm = SVC(
            kernel=model_cfg.get("kernel", "linear"),
            C=model_cfg.get("C", 1.0),
            gamma=model_cfg.get("gamma", "scale"),
            class_weight=None,  # No class weighting - data is manually balanced
            max_iter=model_cfg.get("max_iter", 1000000),
            tol=model_cfg.get("tol", 0.001),
            random_state=seed,
        )
        print("\nSVM Configuration (used for CV and test):")
        print(f"  kernel: {svm.kernel}")
        print(f"  C: {svm.C}")
        print(f"  gamma: {svm.gamma}")
        print(f"  class_weight: {svm.class_weight}")
        print(f"  Random sampling iterations: {n_iterations}")
        # Pass use_random_sampling=True to run_nested_cv
        svm_cv = run_nested_cv(
            dev_filtered,
            y_dev,
            svm,
            env,
            seed,
            fitted_pipeline=None,
            use_wandb=use_wandb,
            use_random_sampling=True,
        )
        # For final model, use same base SVM (ensemble handled in run_final_model)
        svm_final_model = svm
    else:
        print("Running SVM with nested CV...")
        svm = create_svm(svm_config, seed)
        svm_cv = run_nested_cv(
            dev_filtered,
            y_dev,
            svm,
            env,
            seed,
            fitted_pipeline=None,
            use_wandb=use_wandb,
            use_random_sampling=False,
        )
        svm_final_model = svm

    svm_final = run_final_model(
        dev_filtered,
        test_filtered,
        y_dev,
        y_test,
        svm_final_model,
        env,
        seed,
        f"svm_{task_name}",
        svm_dir,
        save_model=not sweep_mode,
        use_random_sampling=use_random_sampling,
    )
    results["svm"] = {**svm_cv, **svm_final}

    # Only generate visualizations and save artifacts if NOT in sweep mode
    if not sweep_mode:
        # Visualizations (reuse test data from final models)
        X_test_pca = svm_final["X_test_pca"]
        y_pred_baseline = baseline_final["y_pred"]
        y_pred_svm = svm_final["y_pred"]

        cm_baseline = compute_confusion_matrix(y_test, y_pred_baseline)
        cm_svm = compute_confusion_matrix(y_test, y_pred_svm)

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

        # Feature importance (using permutation for all kernels for consistency)
        n_components = svm_final["pipeline"]["n_components"]
        pca_features = [f"PC{i+1}" for i in range(n_components)]

        svm_importance = get_feature_importance_permutation(
            svm_final["model"], X_test_pca, y_test, pca_features, seed
        )

        plot_feature_importance(
            svm_importance,
            f"SVM Feature Importance - {task_name}",
            plots_dir / f"importance_svm_{task_name}.png",
            top_n=svm_config.get("interpretation", {}).get("top_n_pcs", 10),
        )

        # Map to brain regions
        from ..tsne.embeddings import get_imaging_columns

        all_imaging_cols = get_imaging_columns(
            dev_filtered, svm_config["imaging_prefixes"]
        )
        valid_features = svm_final["pipeline"]["valid_features"]
        imaging_cols = [
            col for i, col in enumerate(all_imaging_cols) if valid_features[i]
        ]
        brain_regions = map_pca_to_brain_regions(
            svm_importance,
            svm_final["pipeline"]["pca"],
            imaging_cols,
            top_n_components=svm_config.get("interpretation", {}).get("top_n_pcs", 10),
            top_n_features=svm_config.get("interpretation", {}).get(
                "top_n_features", 20
            ),
        )
        brain_regions_enriched = enrich_brain_regions(brain_regions, env)
        brain_regions_enriched.to_csv(svm_dir / "brain_regions.csv", index=False)

        plot_feature_importance(
            brain_regions_enriched,
            f"Top Brain Regions - {task_name}",
            plots_dir / f"brain_regions_{task_name}.png",
            top_n=20,
        )

        # Save results
        with open(svm_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)

    if use_wandb:
        import wandb

        # Log CV metrics (for sweep optimization)
        wandb.log(
            {
                f"{task_name}/cv_balanced_accuracy": svm_cv["aggregated"][
                    "balanced_accuracy_mean"
                ],
                f"{task_name}/cv_roc_auc": svm_cv["aggregated"].get("roc_auc_mean", 0),
            }
        )

        # Only generate plots if NOT in sweep mode (sweeps only need metrics)
        if not sweep_mode:
            labels = ["Control", "Case"]
            try:
                # Get test data from final model results
                X_test_pca_wandb = svm_final["X_test_pca"]
                y_pred_svm_wandb = svm_final["y_pred"]

                # Get decision scores and convert to probabilities
                y_scores = svm_final["model"].decision_function(X_test_pca_wandb)
                from scipy.special import expit

                y_probas_pos = expit(y_scores)
                # W&B expects 2D array: [prob_class_0, prob_class_1] for each sample
                y_probas = np.column_stack([1 - y_probas_pos, y_probas_pos])

                # Generate individual plots (skip expensive calibration/learning curves)
                wandb.sklearn.plot_confusion_matrix(y_test, y_pred_svm_wandb, labels)
                wandb.sklearn.plot_roc(y_test, y_probas, labels)
                wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)
                wandb.sklearn.plot_class_proportions(y_dev, y_test, labels)
                wandb.sklearn.plot_summary_metrics(
                    svm_final["model"], X_test_pca_wandb, y_test, y_test
                )
            except Exception as e:
                print(f"W&B sklearn plot warning: {e}")

    print(f"\nResults saved to {svm_dir}\n")
    return results


def run_svm_pipeline(env, use_wandb: bool = False, sweep_mode: bool = False):
    """Run complete SVM pipeline across all tasks."""
    print("Loading development (train+val) and test data...")
    dev_df, data_dir = load_development_data(env)
    test_df = pd.read_parquet(data_dir / "test.parquet")

    print(f"Dev set: {len(dev_df)} | Test set: {len(test_df)}")

    tasks = env.configs.svm.get("tasks", [])
    all_results = {}

    for task_config in tqdm(tasks, desc="Classification tasks", unit="task"):
        task_results = run_task(
            env, dev_df, test_df, task_config, use_wandb, sweep_mode
        )
        all_results[task_config["name"]] = task_results

    print("\n" + "=" * 60)
    print("SVM Pipeline Complete")
    print("=" * 60)

    return all_results
