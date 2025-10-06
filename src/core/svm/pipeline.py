"""SVM classification pipeline with nested CV and W&B integration."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .evaluation import compute_confusion_matrix, compute_metrics, get_cv_splitter
from .feature_mapping import enrich_brain_regions
from .interpretation import (
    get_feature_importance_linear,
    get_feature_importance_permutation,
    map_pca_to_brain_regions,
)
from .models import create_baseline, create_svm
from .preprocessing import preprocess_fold
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
    use_wandb: bool = False,
) -> dict:
    """Run nested cross-validation with full preprocessing per fold."""
    cv = get_cv_splitter(env.configs.svm, seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(cv.split(df, y), total=cv.n_splits, desc="CV folds")
    ):
        train_df_fold = df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = df.iloc[val_idx].reset_index(drop=True)
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # Nested preprocessing
        X_train_pca, X_val_pca, pipeline = preprocess_fold(
            train_df_fold, val_df_fold, env, seed
        )

        # Train model
        model.fit(X_train_pca, y_train_fold)

        # Evaluate
        y_pred_val = model.predict(X_val_pca)
        y_score_val = (
            model.decision_function(X_val_pca)
            if hasattr(model, "decision_function")
            else None
        )

        metrics = compute_metrics(y_val_fold, y_pred_val, y_score_val)
        metrics["n_components"] = pipeline["n_components"]
        metrics["variance_explained"] = pipeline["variance_explained"]

        fold_results.append(metrics)

    # Aggregate across folds
    aggregated = {}
    for key in fold_results[0].keys():
        values = [fold[key] for fold in fold_results]
        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)

    if use_wandb:
        import wandb

        wandb.log({f"cv_{k}": v for k, v in aggregated.items()})

    return {"fold_results": fold_results, "aggregated": aggregated}


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
    save_model: bool = True,
) -> dict:
    """Train final model on full dev set, evaluate on test set."""
    # Preprocess dev + test
    X_dev_pca, X_test_pca, pipeline = preprocess_fold(dev_df, test_df, env, seed)

    # Train final model
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

    results = {}

    # Baseline: Logistic Regression
    print("Running baseline with nested CV...")
    baseline = create_baseline(svm_config, seed)
    baseline_cv = run_nested_cv(dev_filtered, y_dev, baseline, env, seed, use_wandb)
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

    # SVM
    print("Running SVM with nested CV...")
    svm = create_svm(svm_config, seed)
    svm_cv = run_nested_cv(dev_filtered, y_dev, svm, env, seed, use_wandb)
    svm_final = run_final_model(
        dev_filtered,
        test_filtered,
        y_dev,
        y_test,
        svm,
        env,
        seed,
        f"svm_{task_name}",
        svm_dir,
        save_model=not sweep_mode,
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

        # Feature importance
        n_components = svm_final["pipeline"]["n_components"]
        pca_features = [f"PC{i+1}" for i in range(n_components)]

        if svm_config["model"]["kernel"] == "linear":
            svm_importance = get_feature_importance_linear(
                svm_final["model"], pca_features
            )
        else:
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
