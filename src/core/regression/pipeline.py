"""Regression pipeline for predicting CBCL subscales with nested CV."""

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from ..svm.preprocessing import apply_pca_to_fold, fit_pca_on_dev
from ..tsne.embeddings import get_imaging_columns
from .evaluation import aggregate_cv_results, compute_regression_metrics
from .models import MODEL_REGISTRY, create_baseline
from .visualization import (
    create_summary_figure,
    plot_coefficients,
    plot_predictions,
    plot_residuals,
)

logger = logging.getLogger(__name__)


def residualize_target(y, df, covariate_cols):
    """Regress out covariates (age, sex) from target variable."""
    from sklearn.linear_model import LinearRegression
    import pandas as pd

    # Prepare covariates - encode categorical if needed
    X_cov_list = []
    for col in covariate_cols:
        if col in df.columns:
            values = df[col].values
            # If categorical (like sex_mapped), convert to numeric
            if df[col].dtype == "object":
                values = pd.Categorical(df[col]).codes
            X_cov_list.append(values.reshape(-1, 1))

    X_cov = np.hstack(X_cov_list)
    model = LinearRegression()
    model.fit(X_cov, y)
    y_residual = y - model.predict(X_cov)

    return y_residual


def compute_sample_weights(y, bin_edges, method="inverse_freq"):
    """Compute sample weights or downsample indices for imbalanced regression."""
    # Continuous inverse histogram weighting (bucket-free)
    if method == "inverse_histogram":
        from collections import Counter

        value_counts = Counter(y)
        weights = np.array([1.0 / value_counts[val] for val in y])
        weights = weights / weights.mean()
        return weights

    # Binned methods require bin_edges
    n_bins = len(bin_edges) - 1
    bin_indices = np.digitize(y, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    if method == "inverse_freq":
        weights_per_bin = len(y) / (n_bins * np.maximum(bin_counts, 1))
        sample_weights = weights_per_bin[bin_indices]
        return sample_weights

    elif method == "downsample":
        min_bin_count = bin_counts[bin_counts > 0].min()
        keep_indices = []
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            bin_sample_indices = np.where(bin_mask)[0]
            if len(bin_sample_indices) > 0:
                n_to_sample = min(min_bin_count, len(bin_sample_indices))
                sampled = np.random.choice(bin_sample_indices, size=n_to_sample, replace=False)
                keep_indices.extend(sampled)
        return np.array(keep_indices)

    else:
        raise ValueError(f"Unknown weighting method: {method}")


def apply_sample_weighting(y_train, target_name, env, method="inverse_freq"):
    """Apply sample weighting or downsampling (subjects already filtered by bin range)."""
    reg_config = env.configs.regression
    weighting_cfg = reg_config.get("sample_weighting", {})
    custom_bins = weighting_cfg.get("custom_bins", {})

    if target_name not in custom_bins:
        raise ValueError(f"No custom bins defined for target '{target_name}'")

    bin_edges = custom_bins[target_name]
    result = compute_sample_weights(y_train, bin_edges, method)
    return result


def fit_raw_features_on_train(train_df, env, seed):
    """Fit harmonization + scaling on train (for non-PCA mode)."""
    from ..mlp.pipeline import fit_raw_features_on_train as mlp_fit

    return mlp_fit(train_df, env, seed)


def apply_raw_features_to_fold(fold_df, fitted_pipeline, env):
    """Apply harmonization + scaling to fold (for non-PCA mode)."""
    from ..mlp.pipeline import apply_raw_features_to_fold as mlp_apply

    return mlp_apply(fold_df, fitted_pipeline, env)


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


def filter_target_data(df: pd.DataFrame, target_config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter data for specific regression target."""
    target_col = target_config["column"]

    # Remove subjects with missing target values
    mask = df[target_col].notna()
    df_filtered = df[mask].copy()
    y = df_filtered[target_col].values

    # Remove outliers if configured
    return df_filtered, y


def remove_outliers(y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Remove outliers using IQR method. Returns boolean mask of valid indices."""
    q25, q75 = np.percentile(y, [25, 75])
    iqr = q75 - q25
    lower = q25 - threshold * iqr
    upper = q75 + threshold * iqr
    return (y >= lower) & (y <= upper)


def run_single_fold(
    env,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fold_idx: int,
    seed: int,
    target_name: str,
    model_name: str,
) -> dict:
    """Process a single outer CV fold for one model.

    Args:
        env: Environment with configs
        train_df: Training data (80% of full data)
        test_df: Test data (20% of full data)
        y_train: Target values for train
        y_test: Target values for test
        fold_idx: Fold index (0-4)
        seed: Random seed
        target_name: Name of target subscale
        model_name: Name of model to use

    Returns:
        Dict with baseline and model results
    """
    reg_config = env.configs.regression
    use_pca = reg_config.get("use_pca", True)

    logger.info(
        f"  Train: {len(y_train)} (mean={y_train.mean():.2f}, std={y_train.std():.2f}) | "
        f"Test: {len(y_test)} (mean={y_test.mean():.2f}, std={y_test.std():.2f})"
    )

    # Apply PCA or use raw features
    if use_pca:
        fitted_pipeline = fit_pca_on_dev(train_df, env, seed + fold_idx)
        X_train, _ = apply_pca_to_fold(train_df, train_df, fitted_pipeline, env)
        X_test, _ = apply_pca_to_fold(test_df, test_df, fitted_pipeline, env)
    else:
        fitted_pipeline = fit_raw_features_on_train(train_df, env, seed + fold_idx)
        X_train = apply_raw_features_to_fold(train_df, fitted_pipeline, env)
        X_test = apply_raw_features_to_fold(test_df, fitted_pipeline, env)

    # Apply sample weighting if enabled
    weighting_cfg = reg_config.get("sample_weighting", {})
    if weighting_cfg.get("enabled", False):
        method = weighting_cfg.get("method", "inverse_freq")
        result = apply_sample_weighting(y_train, target_name, env, method)

        if method in ["inverse_freq", "inverse_histogram"]:
            sample_weights = result
            X_train_weighted = X_train
            y_train_weighted = y_train
        elif method == "downsample":
            keep_indices = result
            sample_weights = None
            X_train_weighted = X_train[keep_indices]
            y_train_weighted = y_train[keep_indices]
            print(f"  Downsampled to {len(keep_indices)} samples (from {len(y_train)})")
    else:
        sample_weights = None
        X_train_weighted = X_train
        y_train_weighted = y_train

    # Train baseline (Ridge)
    baseline_model = create_baseline(reg_config, seed + fold_idx)
    if sample_weights is not None:
        baseline_model.fit(X_train_weighted, y_train_weighted, sample_weight=sample_weights)
    else:
        baseline_model.fit(X_train_weighted, y_train_weighted)
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = compute_regression_metrics(y_test, baseline_pred)

    # Train target model
    model_fn = MODEL_REGISTRY[model_name]
    target_model = model_fn(reg_config, seed + fold_idx)
    if sample_weights is not None:
        target_model.fit(X_train_weighted, y_train_weighted, sample_weight=sample_weights)
    else:
        target_model.fit(X_train_weighted, y_train_weighted)
    target_pred = target_model.predict(X_test)
    target_metrics = compute_regression_metrics(y_test, target_pred)

    logger.info(f"  Baseline Ridge: R²={baseline_metrics['r2']:.3f}, " f"MAE={baseline_metrics['mae']:.3f}")
    logger.info(f"  {model_name.upper()}: R²={target_metrics['r2']:.3f}, " f"MAE={target_metrics['mae']:.3f}")

    return {
        "baseline": {
            "model": baseline_model,
            "metrics": baseline_metrics,
            "y_pred": baseline_pred,
            "y_test": y_test,
        },
        model_name: {
            "model": target_model,
            "metrics": target_metrics,
            "y_pred": target_pred,
            "y_test": y_test,
            "X_test": X_test,
            "pipeline": fitted_pipeline,
        },
    }


def run_target_with_nested_cv(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    model_name: str,
):
    """Run regression for single target with nested CV."""
    target_name = target_config["name"]
    target_col = target_config["column"]
    seed = env.configs.run["seed"]
    reg_config = env.configs.regression
    use_pca = reg_config.get("use_pca", True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Target: {target_name} ({target_col})")
    logger.info(f"Model: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Filter data for this target
    df_filtered, y = filter_target_data(full_df, target_config)

    # Filter by bin range if sample weighting is enabled
    weighting_cfg = reg_config.get("sample_weighting", {})
    if weighting_cfg.get("enabled", False):
        custom_bins = weighting_cfg.get("custom_bins", {})
        if target_name in custom_bins:
            bin_edges = custom_bins[target_name]
            min_val, max_val = bin_edges[0], bin_edges[-1]
            valid_mask = (y >= min_val) & (y < max_val)
            n_excluded = (~valid_mask).sum()

            if n_excluded > 0:
                excluded_values = y[~valid_mask]
                print(f"Excluding {n_excluded} subjects outside bin range [{min_val}, {max_val})")
                print(f"  Excluded range: [{excluded_values.min():.1f}, {excluded_values.max():.1f}]")
                df_filtered = df_filtered[valid_mask].reset_index(drop=True)
                y = y[valid_mask]

    # Residualize target if configured
    cov_cfg = reg_config.get("covariates", {})
    if cov_cfg.get("residualize", False):
        is_raw_score = target_name.endswith("_raw")
        apply_to_raw_only = cov_cfg.get("apply_to_raw_scores_only", True)

        if not apply_to_raw_only or is_raw_score:
            covariate_cols = cov_cfg.get("columns", ["age", "sex"])
            y_original = y.copy()
            y = residualize_target(y, df_filtered, covariate_cols)
            print(f"Residualized target (removed {', '.join(covariate_cols)} effects)")
            print(f"  Before: mean={y_original.mean():.2f}, std={y_original.std():.2f}")
            print(f"  After:  mean={y.mean():.2f}, std={y.std():.2f}")

    # Remove outliers if configured
    outlier_cfg = reg_config.get("outliers", {})
    if outlier_cfg.get("enabled", True):
        threshold = outlier_cfg.get("threshold", 3.0)
        valid_mask = remove_outliers(y, threshold=threshold)
        n_outliers = (~valid_mask).sum()
        if n_outliers > 0:
            logger.info(f"Removed {n_outliers} outliers ({100*n_outliers/len(y):.1f}%)")
            df_filtered = df_filtered[valid_mask].reset_index(drop=True)
            y = y[valid_mask]

    logger.info(f"Total samples: {len(y)}")
    logger.info(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}, " f"range=[{y.min():.2f}, {y.max():.2f}]")

    # Create outer CV splitter (stratified by binned target)
    # Bin target into quartiles for stratification
    y_binned = pd.qcut(y, q=5, labels=False, duplicates="drop")
    outer_cv = StratifiedKFold(
        n_splits=reg_config.get("cv", {}).get("n_outer_splits", 5),
        shuffle=True,
        random_state=seed,
    )

    # Storage for fold results
    baseline_folds = []
    model_folds = []

    # Outer CV loop
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(
            outer_cv.split(df_filtered, y_binned),
            total=outer_cv.n_splits,
            desc="CV Folds",
        )
    ):
        print(f"\nFold {fold_idx + 1}/{outer_cv.n_splits}")

        # Split data
        train_df = df_filtered.iloc[train_idx].reset_index(drop=True)
        test_df = df_filtered.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Run this fold
        fold_result = run_single_fold(
            env,
            train_df,
            test_df,
            y_train,
            y_test,
            fold_idx,
            seed,
            target_name,
            model_name,
        )

        baseline_folds.append(fold_result["baseline"])
        model_folds.append(fold_result[model_name])

    # Aggregate results across all folds
    baseline_agg = aggregate_cv_results(baseline_folds)
    model_agg = aggregate_cv_results(model_folds)

    # Setup output directory
    data_dir = env.repo_root / "outputs" / env.configs.run["run_name"] / env.configs.run["run_id"] / f"seed_{seed}"
    reg_dir = data_dir / "regression" / target_name / model_name
    reg_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reg_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {target_name} - {model_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info("Baseline (Ridge):")
    logger.info(f"  R²: {baseline_agg['overall']['r2']:.3f}")
    logger.info(f"  MAE: {baseline_agg['overall']['mae']:.3f}")
    logger.info(f"  RMSE: {baseline_agg['overall']['rmse']:.3f}")
    logger.info(f"  Pearson r: {baseline_agg['overall']['pearson_r']:.3f}")

    logger.info(f"\n{model_name.upper()}:")
    logger.info(f"  R²: {model_agg['overall']['r2']:.3f}")
    logger.info(f"  MAE: {model_agg['overall']['mae']:.3f}")
    logger.info(f"  RMSE: {model_agg['overall']['rmse']:.3f}")
    logger.info(f"  Pearson r: {model_agg['overall']['pearson_r']:.3f}")

    logger.info("\nPer-Fold Stats (Mean ± Std):")
    logger.info(f"  R²: {model_agg['per_fold']['r2_mean']:.3f} ± {model_agg['per_fold']['r2_std']:.3f}")
    logger.info(f"  MAE: {model_agg['per_fold']['mae_mean']:.3f} ± {model_agg['per_fold']['mae_std']:.3f}")
    logger.info(f"{'='*60}\n")

    # Generate visualizations
    all_y_true = np.concatenate([f["y_test"] for f in model_folds])
    np.concatenate([f["y_pred"] for f in baseline_folds])
    all_y_pred_model = np.concatenate([f["y_pred"] for f in model_folds])

    # Main prediction plot
    plot_predictions(
        all_y_true,
        all_y_pred_model,
        f"{model_name.upper()} - {target_name}",
        plots_dir / f"predictions_{model_name}_{target_name}.png",
    )

    # Residual analysis
    plot_residuals(
        all_y_true,
        all_y_pred_model,
        f"{model_name.upper()} - {target_name}",
        plots_dir / f"residuals_{model_name}_{target_name}.png",
    )

    # Extract coefficients for linear models
    coefficients = None
    feature_names = None
    if model_name in ["linear", "ridge", "elastic_net"]:
        # Get coefficients from last fold (they should all have same shape)
        # Use last fold as representative since PCA is fitted consistently
        coefficients = model_folds[-1]["model"].coef_

        # Verify all folds have same number of features
        coef_shapes = [f["model"].coef_.shape for f in model_folds]
        if len(set(coef_shapes)) == 1:
            # All same shape, we can average
            all_coefs = np.array([f["model"].coef_ for f in model_folds])
            coefficients = np.mean(all_coefs, axis=0)
        else:
            # Different shapes (shouldn't happen but handle gracefully)
            print(f"  Warning: Coefficient shapes vary across folds: {coef_shapes}")
            print("  Using coefficients from last fold only")

        # Generate feature names
        n_features = len(coefficients)
        if use_pca:
            feature_names = [f"PC{i+1}" for i in range(n_features)]
        else:
            # Get actual feature names from the first fold
            imaging_cols = get_imaging_columns(df_filtered.iloc[:10], reg_config.get("imaging_prefixes", []))
            feature_names = imaging_cols[:n_features]

        # Plot coefficients
        plot_coefficients(
            coefficients,
            feature_names,
            f"{model_name.upper()} Coefficients - {target_name}",
            plots_dir / f"coefficients_{model_name}_{target_name}.png",
            top_n=30,
        )

    # Create comprehensive summary figure
    create_summary_figure(
        all_y_true,
        all_y_pred_model,
        coefficients,
        feature_names,
        f"{model_name.upper()} - {target_name}",
        plots_dir / f"summary_{model_name}_{target_name}.png",
    )

    # Save results
    results = {
        "baseline": baseline_agg,
        model_name: model_agg,
        "baseline_folds": baseline_folds,
        f"{model_name}_folds": model_folds,
    }
    with open(reg_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    logger.info(f"Results saved to {reg_dir}\n")

    return {
        "baseline": baseline_agg,
        model_name: model_agg,
    }


def run_regression_pipeline(env):
    """Run complete regression pipeline for all targets and models."""
    logger.info("=" * 60)
    logger.info("Regression Pipeline: CBCL Subscale Prediction")
    logger.info("=" * 60)
    logger.info("Loading full dataset for nested CV...")

    full_df = load_full_dataset(env)
    logger.info(f"Total samples: {len(full_df)}")

    reg_config = env.configs.regression
    targets = reg_config.get("targets", [])
    models_config = reg_config.get("models", {})

    # Get enabled models
    enabled_models = [name for name, cfg in models_config.items() if cfg.get("enabled", True)]

    logger.info(f"\nTargets: {[t['name'] for t in targets]}")
    logger.info(f"Models: {enabled_models}")

    all_results = {}

    for target_config in targets:
        target_name = target_config["name"]
        all_results[target_name] = {}

        for model_name in enabled_models:
            results = run_target_with_nested_cv(env, full_df, target_config, model_name)
            all_results[target_name][model_name] = results

    logger.info("\n" + "=" * 60)
    logger.info("Regression Pipeline Complete")
    logger.info("=" * 60)

    return all_results
