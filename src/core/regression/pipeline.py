"""Regression pipeline for predicting psychosis severity with nested CV."""

import io
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm

from ..svm.preprocessing import apply_pca_to_fold, fit_pca_on_dev
from ..tsne.embeddings import get_imaging_columns, get_roi_columns_from_config
from .evaluation import aggregate_cv_results, compute_regression_metrics
from .models import MODEL_REGISTRY, create_baseline, model_supports_sample_weight
from .run_tracker import save_run_metadata
from .visualization import (
    create_summary_figure,
    plot_coefficients,
    plot_permutation_importance,
    plot_predictions,
    plot_residuals,
)

logger = logging.getLogger(__name__)


def compare_cv_strategies(
    df: pd.DataFrame,
    y: np.ndarray,
    stratify_key,
    family_groups: np.ndarray | None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Compare fold composition between StratifiedKFold and StratifiedGroupKFold.

    Quantifies how much fold assignments change when family-awareness is added,
    helping distinguish algorithmic fold-recomposition from genuine sibling-leakage
    correction. Use this to understand the r=0.181 → r=0.152 drop.

    Args:
        df: Feature dataframe (used for size reference only).
        y: Target values.
        stratify_key: Stratification array passed to CV splitter.
        family_groups: Family group IDs (None → returns trivial result).
        n_splits: Number of CV folds.
        seed: Random seed.

    Returns:
        Dict with:
            jaccard_per_fold: list of per-fold Jaccard similarities (1.0 = identical).
            mean_jaccard: Mean fold overlap across folds.
            n_siblings_in_sample: Subjects sharing a family group.
            fold_size_diff_per_fold: Absolute test-set size differences per fold.
            interpretation: Human-readable summary string.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    cv_std = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    std_folds = list(cv_std.split(df, stratify_key))

    if family_groups is None:
        return {
            "jaccard_per_fold": [1.0] * n_splits,
            "mean_jaccard": 1.0,
            "n_siblings_in_sample": 0,
            "fold_size_diff_per_fold": [0] * n_splits,
            "mean_fold_size_diff": 0.0,
            "interpretation": "No family groups — CV strategies are identical.",
        }

    cv_fam = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fam_folds = list(cv_fam.split(df, stratify_key, family_groups))

    jaccards, size_diffs = [], []
    for (_, std_test), (_, fam_test) in zip(std_folds, fam_folds):
        std_set, fam_set = set(std_test), set(fam_test)
        intersection = len(std_set & fam_set)
        union = len(std_set | fam_set)
        jaccards.append(intersection / union if union > 0 else 1.0)
        size_diffs.append(abs(len(std_test) - len(fam_test)))

    unique_fam, counts = np.unique(family_groups, return_counts=True)
    n_siblings = int(np.sum(counts[counts > 1]))
    mean_j = float(np.mean(jaccards))

    overlap_label = "mostly algorithmic" if mean_j > 0.80 else "substantial"
    interpretation = (
        f"{mean_j * 100:.1f}% average fold overlap ({overlap_label} change from "
        f"family-awareness). {n_siblings} subjects share a family in this sample."
    )

    return {
        "jaccard_per_fold": jaccards,
        "mean_jaccard": mean_j,
        "n_siblings_in_sample": n_siblings,
        "fold_size_diff_per_fold": size_diffs,
        "mean_fold_size_diff": float(np.mean(size_diffs)),
        "interpretation": interpretation,
    }


def _prepare_covariates(df, covariate_cols):
    """Prepare covariate matrix from dataframe."""
    X_cov_list = []
    for col in covariate_cols:
        if col in df.columns:
            values = df[col].values
            if not pd.api.types.is_numeric_dtype(df[col]):
                values = pd.Categorical(df[col]).codes
            X_cov_list.append(values.reshape(-1, 1))
    return np.hstack(X_cov_list)


def fit_residualize(y, df, covariate_cols):
    """Fit residualization model on training data. Returns fitted model."""
    from sklearn.linear_model import LinearRegression

    X_cov = _prepare_covariates(df, covariate_cols)
    model = LinearRegression()
    model.fit(X_cov, y)
    return model


def apply_residualize(y, df, covariate_cols, model):
    """Apply fitted residualization model to remove covariate effects."""
    X_cov = _prepare_covariates(df, covariate_cols)
    return y - model.predict(X_cov)


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
        rng = np.random.RandomState(42)
        min_bin_count = bin_counts[bin_counts > 0].min()
        keep_indices = []
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            bin_sample_indices = np.where(bin_mask)[0]
            if len(bin_sample_indices) > 0:
                n_to_sample = min(min_bin_count, len(bin_sample_indices))
                sampled = rng.choice(bin_sample_indices, size=n_to_sample, replace=False)
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
        # No custom bins for this target - return None (no weighting)
        return None

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


def filter_target_data(
    df: pd.DataFrame,
    target_config: dict,
    harmonize_config: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter data for specific regression target.

    Drops subjects with:
      - Missing target values
      - NaN in ComBat site or covariate columns (if harmonize_config provided)

    Applying the covariate NaN drop here ensures every downstream pipeline
    (main SVR, lateralization comparison, univariate analysis) operates on
    the exact same subject set.
    """
    target_col = target_config["column"]

    # Remove subjects with missing target values
    mask = df[target_col].notna()
    df_filtered = df[mask].copy()
    y = df_filtered[target_col].values

    # Explicitly exclude non-binary sex subjects (intersex/other) — incompatible
    # with binary ComBat sex covariate (demo_sex_v2 = 3 maps to "intersex_other").
    if "sex_mapped" in df_filtered.columns:
        non_binary = ~df_filtered["sex_mapped"].isin(["male", "female"])
        non_binary = non_binary & df_filtered["sex_mapped"].notna()
        if non_binary.any():
            n_nb = int(non_binary.sum())
            if verbose:
                print(f"  filter_target_data: excluding {n_nb} subject(s) with non-binary sex "
                      f"(sex_mapped={df_filtered.loc[non_binary, 'sex_mapped'].unique().tolist()}) "
                      f"— not compatible with binary ComBat sex covariate")
            df_filtered = df_filtered[~non_binary].reset_index(drop=True)
            y = df_filtered[target_col].values

    # Remove subjects with NaN in ComBat site/covariate columns
    if harmonize_config is not None:
        site_col = harmonize_config.get("site_column", "mri_info_manufacturer")
        cov_cols = harmonize_config.get("covariates", [])
        check_cols = [c for c in [site_col] + cov_cols if c in df_filtered.columns]
        if check_cols:
            nan_mask = df_filtered[check_cols].isna().any(axis=1)
            if nan_mask.any():
                n_drop = int(nan_mask.sum())
                if verbose:
                    print(f"  filter_target_data: dropping {n_drop} subject(s) with NaN in ComBat columns {check_cols}")
                df_filtered = df_filtered[~nan_mask].reset_index(drop=True)
                y = y[~nan_mask.values]

    return df_filtered, y


def remove_outliers(y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Remove outliers using IQR method. Returns boolean mask of valid indices."""
    q25, q75 = np.percentile(y, [25, 75])
    iqr = q75 - q25
    lower = q25 - threshold * iqr
    upper = q75 + threshold * iqr
    return (y >= lower) & (y <= upper)


def get_feature_names(env, df, n_imaging):
    """Resolve feature names based on config (handles raw, asymmetry, ai_total)."""
    reg_config = env.configs.regression
    use_pca = reg_config.get("use_pca", True)
    feature_mode = reg_config.get("feature_mode", "pca" if use_pca else "raw")
    feature_transform = reg_config.get("feature_transform", "raw")

    if use_pca:
        return [f"PC{i+1}" for i in range(n_imaging)]

    if feature_mode == "roi" and feature_transform != "raw":
        from .univariate import extract_bilateral_pairs
        roi_networks = reg_config.get("roi_networks", [])
        bilateral_pairs, _ = extract_bilateral_pairs(env.configs.data, roi_networks)
        if feature_transform == "asymmetry":
            names = [f"{name}_AI" for name, _, _ in bilateral_pairs]
        elif feature_transform == "ai_total":
            names = [f"{name}_AI" for name, _, _ in bilateral_pairs] + \
                    [f"{name}_total" for name, _, _ in bilateral_pairs]
        else:
            names = [f"{name}_AI" for name, _, _ in bilateral_pairs] + \
                    [f"{name}_total" for name, _, _ in bilateral_pairs]
        return sorted(names)[:n_imaging]

    if feature_mode == "roi":
        roi_networks = reg_config.get("roi_networks", [])
        roi_columns = get_roi_columns_from_config(env.configs.data, roi_networks) if roi_networks else []
        return [c for c in roi_columns if c in df.columns][:n_imaging]

    imaging_cols = get_imaging_columns(df.iloc[:10], reg_config.get("imaging_prefixes", []))
    return imaging_cols[:n_imaging]


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
    residualized: bool = False,
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
        residualized: Whether target was residualized (disables [0, None] clipping)

    Returns:
        Dict with baseline and model results
    """
    from sklearn.preprocessing import StandardScaler

    reg_config = env.configs.regression
    use_pca = reg_config.get("use_pca", True)

    logger.info(
        f"  Train: {len(y_train)} (mean={y_train.mean():.2f}, std={y_train.std():.2f}) | "
        f"Test: {len(y_test)} (mean={y_test.mean():.2f}, std={y_test.std():.2f})"
    )

    # Apply PCA, ROI selection, or use all raw features
    feature_mode = reg_config.get("feature_mode", "pca" if use_pca else "raw")

    if feature_mode == "roi":
        # ROI selection using networks from data.yaml - filter columns BEFORE harmonization
        roi_networks = reg_config.get("roi_networks", [])
        roi_columns = get_roi_columns_from_config(env.configs.data, roi_networks) if roi_networks else []
        roi_cols_present = [c for c in roi_columns if c in train_df.columns]

        if fold_idx == 0:
            print(f"  ROI feature selection: {len(roi_cols_present)} features (networks: {roi_networks})")
            print(f"  Example features: {roi_cols_present[:5]}")

        # Filter dataframes to only ROI columns + metadata before harmonization
        meta_cols = [c for c in train_df.columns if not any(c.startswith(p) for p in reg_config.get("imaging_prefixes", []))]
        train_roi_df = train_df[meta_cols + roi_cols_present]
        test_roi_df = test_df[meta_cols + roi_cols_present]

        fitted_pipeline, X_train = fit_raw_features_on_train(train_roi_df, env, seed + fold_idx)
        X_test = apply_raw_features_to_fold(test_roi_df, fitted_pipeline, env)
    elif use_pca:
        fitted_pipeline = fit_pca_on_dev(train_df, env, seed + fold_idx)
        X_train, _ = apply_pca_to_fold(train_df, train_df, fitted_pipeline, env)
        X_test, _ = apply_pca_to_fold(test_df, test_df, fitted_pipeline, env)
    else:
        fitted_pipeline, X_train = fit_raw_features_on_train(train_df, env, seed + fold_idx)
        X_test = apply_raw_features_to_fold(test_df, fitted_pipeline, env)

    # Optional: transform L/R features to asymmetry index (AI) features
    feature_transform = reg_config.get("feature_transform", "raw")
    if feature_transform != "raw" and feature_mode == "roi":
        from .univariate import extract_bilateral_pairs, compute_asymmetry_features

        # Inverse scale to get ComBat-harmonized (but unscaled) values
        X_train_harm = fitted_pipeline["scaler"].inverse_transform(X_train)
        X_test_harm = fitted_pipeline["scaler"].inverse_transform(X_test)

        # Get bilateral pairs and compute asymmetry features
        bilateral_pairs, _ = extract_bilateral_pairs(env.configs.data, roi_networks)
        # Map surviving columns (after valid_features mask)
        surviving_cols = [roi_cols_present[i] for i, v in enumerate(fitted_pipeline["valid_features"]) if v]
        valid_pairs = [(n, l, r) for n, l, r in bilateral_pairs
                       if l in surviving_cols and r in surviving_cols]

        train_asym = compute_asymmetry_features(X_train_harm, surviving_cols, valid_pairs)
        test_asym = compute_asymmetry_features(X_test_harm, surviving_cols, valid_pairs)

        # Select which derived features to use
        if feature_transform == "asymmetry":
            transform_names = sorted(k for k in train_asym if k.endswith("_AI"))
        elif feature_transform == "ai_total":
            ai_names = sorted(k for k in train_asym if k.endswith("_AI"))
            tot_names = sorted(k for k in train_asym if k.endswith("_total"))
            transform_names = ai_names + tot_names
        else:
            transform_names = sorted(train_asym.keys())

        X_train = np.column_stack([train_asym[k] for k in transform_names])
        X_test = np.column_stack([test_asym[k] for k in transform_names])

        # Re-scale the transformed features
        ai_scaler = StandardScaler()
        X_train = ai_scaler.fit_transform(X_train)
        X_test = ai_scaler.transform(X_test)

        if fold_idx == 0:
            print(f"  Feature transform: {feature_transform} -> {len(transform_names)} features: {transform_names}")

    # Add demographic covariates as features if enabled
    cov_cfg = reg_config.get("covariates", {})
    covariate_names = []
    if cov_cfg.get("include_as_features", False):
        cov_cols = cov_cfg.get("columns", [])
        cov_cols_present = [c for c in cov_cols if c in train_df.columns]

        if cov_cols_present:
            from sklearn.preprocessing import StandardScaler

            cov_train = []
            cov_test = []
            for col in cov_cols_present:
                if col == "sex_mapped" or not pd.api.types.is_numeric_dtype(train_df[col]):
                    # Binary encode sex: male=0, female=1
                    train_vals = (train_df[col] == "female").astype(float).values.reshape(-1, 1)
                    test_vals = (test_df[col] == "female").astype(float).values.reshape(-1, 1)
                    cov_train.append(train_vals)
                    cov_test.append(test_vals)
                else:
                    # Scale numeric
                    scaler = StandardScaler()
                    cov_train.append(scaler.fit_transform(train_df[col].values.reshape(-1, 1)))
                    cov_test.append(scaler.transform(test_df[col].values.reshape(-1, 1)))
                covariate_names.append(col)

            cov_train = np.hstack(cov_train)
            cov_test = np.hstack(cov_test)
            X_train = np.hstack([X_train, cov_train])
            X_test = np.hstack([X_test, cov_test])

            if fold_idx == 0:
                print(f"  Added {len(cov_cols_present)} covariates as features: {cov_cols_present}")

    # Apply sample weighting if enabled
    weighting_cfg = reg_config.get("sample_weighting", {})
    if weighting_cfg.get("enabled", False):
        method = weighting_cfg.get("method", "inverse_freq")
        result = apply_sample_weighting(y_train, target_name, env, method)

        if result is None:
            # No custom bins for this target - skip weighting
            sample_weights = None
            X_train_weighted = X_train
            y_train_weighted = y_train
        elif method in ["inverse_freq", "inverse_histogram"]:
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

    # Scale target for SVR/MLP to prevent flat predictions
    scale_target = model_name in ["svr", "mlp"]
    if scale_target:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_weighted.reshape(-1, 1)).ravel()
    else:
        y_scaler = None
        y_train_scaled = y_train_weighted

    # Train baseline (Ridge)
    baseline_model = create_baseline(reg_config, seed + fold_idx)
    if sample_weights is not None:
        baseline_model.fit(X_train_weighted, y_train_weighted, sample_weight=sample_weights)
    else:
        baseline_model.fit(X_train_weighted, y_train_weighted)
    baseline_pred = baseline_model.predict(X_test)
    # Clip to domain bounds only for raw scores (residualized targets can be negative)
    if not residualized:
        baseline_pred = np.clip(baseline_pred, 0, None)
    baseline_metrics = compute_regression_metrics(y_test, baseline_pred)

    # Train target model (with optional inner CV tuning)
    tuning_cfg = reg_config.get("tuning", {})
    param_grids = tuning_cfg.get("param_grids", {})
    tuning_enabled = tuning_cfg.get("enabled", False) and model_name in param_grids

    if tuning_enabled:
        from sklearn.model_selection import GridSearchCV, StratifiedKFold as SKF
        from sklearn.metrics import make_scorer
        from scipy.stats import pearsonr as _pearsonr

        param_grid = param_grids[model_name]
        inner_cv_folds = tuning_cfg.get("cv_folds", 3)
        scoring_name = tuning_cfg.get("scoring", "neg_mean_absolute_error")

        if scoring_name == "pearson_r":
            scoring = make_scorer(lambda y_true, y_pred: _pearsonr(y_true, y_pred)[0])
        else:
            scoring = scoring_name

        model_fn = MODEL_REGISTRY[model_name]
        base_model = model_fn(reg_config, seed + fold_idx)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=inner_cv_folds,
            scoring=scoring,
            n_jobs=tuning_cfg.get("n_jobs", -1),
            refit=True,
        )
        grid_search.fit(X_train_weighted, y_train_scaled)
        target_model = grid_search.best_estimator_

        print(f"  Fold {fold_idx+1} tuned {model_name}: {grid_search.best_params_} (score={grid_search.best_score_:.4f})")
    else:
        model_fn = MODEL_REGISTRY[model_name]
        target_model = model_fn(reg_config, seed + fold_idx)
        if sample_weights is not None and model_supports_sample_weight(model_name, reg_config):
            target_model.fit(X_train_weighted, y_train_scaled, sample_weight=sample_weights)
        else:
            if sample_weights is not None and fold_idx == 0:
                logger.warning(
                    f"  Model '{model_name}' has supports_sample_weight=False in config. "
                    f"Training without sample weighting. Set supports_sample_weight: true "
                    f"in regression.yaml models.{model_name} to enable."
                )
            target_model.fit(X_train_weighted, y_train_scaled)

    # Predict and inverse transform if target was scaled
    target_pred = target_model.predict(X_test)
    if scale_target and y_scaler is not None:
        target_pred = y_scaler.inverse_transform(target_pred.reshape(-1, 1)).ravel()

    if not residualized:
        target_pred = np.clip(target_pred, 0, None)

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
            "y_train": y_train,
            "X_test": X_test,
            "X_train": X_train,
            "pipeline": fitted_pipeline,
        },
    }


def run_target_with_nested_cv(
    env,
    full_df: pd.DataFrame,
    target_config: dict,
    model_name: str,
    verbose: bool = True,
):
    """Run regression for single target with nested CV."""
    target_name = target_config["name"]
    target_col = target_config["column"]
    seed = env.configs.run["seed"]
    reg_config = env.configs.regression
    use_pca = reg_config.get("use_pca", True)

    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Target: {target_name} ({target_col})")
        logger.info(f"Model: {model_name.upper()}")
        logger.info(f"{'='*60}")

    # Filter data for this target (drops NaN target + NaN ComBat covariate subjects)
    harm_config = env.configs.harmonize
    df_filtered, y = filter_target_data(full_df, target_config, harmonize_config=harm_config, verbose=verbose)

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
                if verbose:
                    print(f"Excluding {n_excluded} subjects outside bin range [{min_val}, {max_val})")
                    print(f"  Excluded range: [{excluded_values.min():.1f}, {excluded_values.max():.1f}]")
                df_filtered = df_filtered[valid_mask].reset_index(drop=True)
                y = y[valid_mask]

    # Check if residualization is configured (applied per-fold inside CV loop)
    residualized = False
    covariate_cols_for_residualize = None
    cov_cfg = reg_config.get("covariates", {})
    if cov_cfg.get("residualize", False):
        is_raw_score = target_name.endswith("_raw")
        apply_to_raw_only = cov_cfg.get("apply_to_raw_scores_only", True)

        if not apply_to_raw_only or is_raw_score:
            covariate_cols_for_residualize = cov_cfg.get("columns", ["age", "sex"])
            residualized = True
            if verbose:
                print(f"Will residualize target per-fold (removing {', '.join(covariate_cols_for_residualize)} effects)")

    # Remove outliers if configured
    outlier_cfg = reg_config.get("outliers", {})
    if outlier_cfg.get("enabled", False):
        threshold = outlier_cfg.get("threshold", 3.0)
        valid_mask = remove_outliers(y, threshold=threshold)
        n_outliers = (~valid_mask).sum()
        if n_outliers > 0 and verbose:
            logger.info(f"Removed {n_outliers} outliers ({100*n_outliers/len(y):.1f}%)")
            df_filtered = df_filtered[valid_mask].reset_index(drop=True)
            y = y[valid_mask]

    if verbose:
        logger.info(f"Total samples: {len(y)}")
        logger.info(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}, " f"range=[{y.min():.2f}, {y.max():.2f}]")

    # Create outer CV splitter (stratified by site AND binned target)
    # Site stratification ensures all MRI manufacturers appear in both train and test for Combat
    n_splits = reg_config.get("cv", {}).get("n_outer_splits", 5)

    # Get site column for stratification (critical for Combat harmonization)
    harm_config = env.configs.harmonize
    site_col = harm_config.get("site_column", "mri_info_manufacturer")
    use_pca = reg_config.get("use_pca", True)
    feature_mode = reg_config.get("feature_mode", "pca" if use_pca else "raw")

    # For Combat harmonization, filter out sites with too few samples
    # Combat requires all sites to be present in both train and test folds
    if feature_mode in ["raw", "roi"] and site_col in df_filtered.columns:
        site_counts = df_filtered[site_col].value_counts()
        # Need at least n_splits samples per site to guarantee representation in all folds
        min_required = n_splits
        small_sites = site_counts[site_counts < min_required].index.tolist()

        if small_sites:
            n_excluded = df_filtered[site_col].isin(small_sites).sum()
            if verbose:
                logger.warning(
                    f"Excluding {n_excluded} subjects from {len(small_sites)} sites with <{min_required} samples: {small_sites}"
                )
            valid_site_mask = ~df_filtered[site_col].isin(small_sites)
            df_filtered = df_filtered[valid_site_mask].reset_index(drop=True)
            y = y[valid_site_mask.values]
            if verbose:
                logger.info(f"Remaining samples after site filtering: {len(y)}")

    y_binned = pd.qcut(y, q=5, labels=False, duplicates="drop")

    # Determine stratification key based on feature mode
    stratify_key = y_binned  # Default: target bins only
    stratify_desc = "target bins"

    if feature_mode in ["raw", "roi"] and site_col in df_filtered.columns:
        # Site stratification is critical for Combat - ensures all sites in train/test
        site_codes = pd.Categorical(df_filtered[site_col]).codes
        n_sites = len(df_filtered[site_col].unique())

        # Try combined stratification (site + target bins)
        combined_key = y_binned * n_sites + site_codes
        min_group_size = pd.Series(combined_key).value_counts().min()

        if min_group_size >= n_splits:
            # Combined stratification is feasible
            stratify_key = combined_key
            stratify_desc = f"target bins AND site ({n_sites} sites)"
        else:
            # Fall back to site-only stratification (more important for Combat)
            site_codes_series = pd.Series(site_codes)
            min_site_size = site_codes_series.value_counts().min()
            if min_site_size >= n_splits:
                stratify_key = site_codes
                stratify_desc = f"site only ({n_sites} sites)"
                if verbose:
                    logger.warning(
                        f"Combined stratification not feasible (min group size {min_group_size}), "
                        f"using site-only stratification"
                    )
            else:
                # This shouldn't happen after filtering, but handle gracefully
                if verbose:
                    logger.warning(
                        f"Site stratification not feasible (min site size {min_site_size}), "
                        f"using target bins only. Combat may fail on some folds."
                    )

    if verbose:
        logger.info(f"Stratifying CV by {stratify_desc}")

    # Family-aware CV: ensure siblings stay in the same fold
    cv_cfg = reg_config.get("cv", {})
    use_family_aware = cv_cfg.get("family_aware", True)

    family_groups = None
    if use_family_aware and "rel_family_id" in df_filtered.columns:
        family_groups = pd.to_numeric(df_filtered["rel_family_id"], errors="coerce").values
        # Assign unique group IDs to subjects with missing family data
        missing_mask = np.isnan(family_groups)
        if missing_mask.any():
            max_id = np.nanmax(family_groups) if (~missing_mask).any() else 0
            if np.isnan(max_id):
                max_id = 0
            unique_ids = np.arange(max_id + 1, max_id + 1 + missing_mask.sum())
            family_groups[missing_mask] = unique_ids

    if family_groups is not None:
        outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        n_families = len(np.unique(family_groups))
        n_sibling_subjects = len(family_groups) - len(np.unique(family_groups))
        if verbose:
            logger.info(f"Family-aware CV: {n_families} families, {n_sibling_subjects} subjects share a family")
    else:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if verbose:
            if not use_family_aware:
                logger.info("Family-aware CV disabled (cv.family_aware: false) — using StratifiedKFold")
            else:
                logger.info("No family data available — using standard StratifiedKFold")

    # Storage for fold results
    baseline_folds = []
    model_folds = []
    resid_info = []  # Store residualization coefficients per fold

    # ── CV fold diagnostics ──
    split_args = (df_filtered, stratify_key, family_groups) if family_groups is not None else (df_filtered, stratify_key)
    scanner_col = env.configs.data["columns"]["mapping"].get("scanner_model", "mri_info_manufacturersmn")
    has_scanner = scanner_col in df_filtered.columns
    has_family = family_groups is not None

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  CV FOLD DIAGNOSTICS (n={len(df_filtered)}, {n_splits} folds)")
        print(f"{'─'*70}")
        header = f"  {'Fold':<6} {'n_train':>8} {'n_test':>7} {'y_train':>12} {'y_test':>12}"
        if has_scanner:
            header += f" {'sites_tr':>9} {'sites_te':>9}"
        if has_family:
            header += f" {'fam_leak':>9}"
        print(header)
        print(f"  {'─'*len(header)}")

        _diag_splits = list(outer_cv.split(*split_args))
        for fi, (tr_idx, te_idx) in enumerate(_diag_splits):
            y_tr, y_te = y[tr_idx], y[te_idx]
            line = f"  {fi+1:<6} {len(tr_idx):>8} {len(te_idx):>7} {y_tr.mean():>6.1f}±{y_tr.std():>4.1f} {y_te.mean():>6.1f}±{y_te.std():>4.1f}"
            if has_scanner:
                n_sites_tr = df_filtered.iloc[tr_idx][scanner_col].nunique()
                n_sites_te = df_filtered.iloc[te_idx][scanner_col].nunique()
                line += f" {n_sites_tr:>9} {n_sites_te:>9}"
            if has_family:
                fam_tr = set(family_groups[tr_idx])
                fam_te = set(family_groups[te_idx])
                leaked = len(fam_tr & fam_te)
                line += f" {leaked:>9}"
            print(line)

        if has_family:
            all_leaked = 0
            for fi, (tr_idx, te_idx) in enumerate(_diag_splits):
                fam_tr = set(family_groups[tr_idx])
                fam_te = set(family_groups[te_idx])
                all_leaked += len(fam_tr & fam_te)
            print(f"\n  Family leakage: {all_leaked} families shared across train/test ({'NONE ✓' if all_leaked == 0 else 'WARNING'})")
        print(f"{'─'*70}\n")

        # Re-create splitter (consumed by diagnostics)
        if has_family:
            outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Outer CV loop
    cv_iter = outer_cv.split(*split_args)
    if verbose:
        cv_iter = tqdm(cv_iter, total=outer_cv.n_splits, desc="CV Folds")
    for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
        if verbose:
            print(f"\nFold {fold_idx + 1}/{outer_cv.n_splits}")

        # Split data
        train_df = df_filtered.iloc[train_idx].reset_index(drop=True)
        test_df = df_filtered.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Residualize within fold (fit on train only, apply to both — no leakage)
        if covariate_cols_for_residualize is not None:
            resid_model = fit_residualize(y_train, train_df, covariate_cols_for_residualize)
            # Save residualization coefficients
            resid_info.append({
                "fold": fold_idx + 1,
                "intercept": resid_model.intercept_,
                "coef": dict(zip(covariate_cols_for_residualize, resid_model.coef_)),
                "r2": resid_model.score(
                    _prepare_covariates(train_df, covariate_cols_for_residualize), y_train
                ),
            })
            y_train = apply_residualize(y_train, train_df, covariate_cols_for_residualize, resid_model)
            y_test = apply_residualize(y_test, test_df, covariate_cols_for_residualize, resid_model)

        # Run this fold (suppress stdout from neuroHarmonize/inner prints when quiet)
        if verbose:
            fold_result = run_single_fold(
                env, train_df, test_df, y_train, y_test,
                fold_idx, seed, target_name, model_name,
                residualized=(covariate_cols_for_residualize is not None),
            )
        else:
            import sys, contextlib
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                fold_result = run_single_fold(
                    env, train_df, test_df, y_train, y_test,
                    fold_idx, seed, target_name, model_name,
                    residualized=(covariate_cols_for_residualize is not None),
                )

        baseline_folds.append(fold_result["baseline"])
        fold_entry = fold_result[model_name]
        fold_entry["train_idx"] = train_idx.tolist()
        fold_entry["test_idx"] = test_idx.tolist()
        model_folds.append(fold_entry)

    # Aggregate results across all folds
    baseline_agg = aggregate_cv_results(baseline_folds)
    model_agg = aggregate_cv_results(model_folds)

    # Skip file I/O and plots when running in quiet mode (e.g. robustness loops)
    if not verbose:
        return {
            "baseline": baseline_agg,
            model_name: model_agg,
        }

    # Setup output directory
    data_dir = env.repo_root / "outputs" / env.configs.run["run_name"] / env.configs.run["run_id"] / f"seed_{seed}"
    reg_dir = data_dir / "regression" / target_name / model_name
    reg_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reg_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Print results
    if verbose:
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
    all_y_pred_model = np.concatenate([f["y_pred"] for f in model_folds])

    # Main prediction plot
    plot_predictions(
        all_y_true,
        all_y_pred_model,
        f"{model_name.upper()} - {target_name}",
        plots_dir / f"predictions_{model_name}_{target_name}.png",
        residualized=residualized,
    )

    # Residual analysis
    plot_residuals(
        all_y_true,
        all_y_pred_model,
        f"{model_name.upper()} - {target_name}",
        plots_dir / f"residuals_{model_name}_{target_name}.png",
    )

    # Extract coefficients for linear models (including SVR with linear kernel)
    coefficients = None
    feature_names = None
    svr_is_linear = model_name == "svr" and reg_config.get("models", {}).get("svr", {}).get("kernel", "rbf") == "linear"
    if model_name in ["linear", "ridge", "elastic_net"] or svr_is_linear:
        # Get coefficients from last fold
        coefficients = model_folds[-1]["model"].coef_
        if coefficients.ndim == 2:  # SVR returns (1, n_features)
            coefficients = coefficients.ravel()

        # Average coefficients across folds if shapes match
        coef_shapes = [f["model"].coef_.shape for f in model_folds]
        if len(set(coef_shapes)) == 1:
            all_coefs = np.array([f["model"].coef_.ravel() for f in model_folds])
            coefficients = np.mean(all_coefs, axis=0)
        else:
            print(f"  Warning: Coefficient shapes vary across folds: {coef_shapes}")

        n_features = len(coefficients)

        # Get covariate names if they were added
        cov_cfg = reg_config.get("covariates", {})
        cov_names = []
        if cov_cfg.get("include_as_features", False):
            cov_names = [c for c in cov_cfg.get("columns", []) if c in df_filtered.columns]

        n_imaging = n_features - len(cov_names)
        feature_names = get_feature_names(env, df_filtered, n_imaging) + cov_names

        # Plot coefficients
        plot_coefficients(
            coefficients,
            feature_names,
            f"{model_name.upper()} Coefficients - {target_name}",
            plots_dir / f"coefficients_{model_name}_{target_name}.png",
            top_n=30,
        )

    # Permutation importance for non-linear models (aggregated across all folds)
    elif model_name in ["random_forest", "mlp"] or (model_name == "svr" and not svr_is_linear):
        from sklearn.inspection import permutation_importance as perm_imp

        n_folds = len(model_folds)
        logger.info(f"Computing permutation importance for {model_name.upper()} across {n_folds} folds...")

        fold_importances = []
        for fi, fold in enumerate(model_folds):
            result = perm_imp(
                fold["model"], fold["X_test"], fold["y_test"],
                n_repeats=10, random_state=seed + fi, scoring="r2"
            )
            fold_importances.append(result.importances_mean)

        # Average across folds
        avg_importance = np.mean(fold_importances, axis=0)
        avg_std = np.std(fold_importances, axis=0)

        # Create a simple namespace to match the old interface
        class PermResult:
            pass
        perm_result = PermResult()
        perm_result.importances_mean = avg_importance
        perm_result.importances_std = avg_std

        n_features = model_folds[0]["X_test"].shape[1]

        # Get covariate names if they were added
        cov_cfg = reg_config.get("covariates", {})
        cov_names = []
        if cov_cfg.get("include_as_features", False):
            cov_names = [c for c in cov_cfg.get("columns", []) if c in df_filtered.columns]

        n_imaging = n_features - len(cov_names)
        feature_names = get_feature_names(env, df_filtered, n_imaging) + cov_names

        plot_permutation_importance(
            feature_names,
            perm_result.importances_mean,
            perm_result.importances_std,
            f"{model_name.upper()} Permutation Importance - {target_name}",
            plots_dir / f"importance_{model_name}_{target_name}.png",
            top_n=30,
        )

        # Save importance CSV
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }).sort_values("importance_mean", ascending=False)
        importance_df.to_csv(reg_dir / f"importance_{model_name}.csv", index=False)

        coefficients = perm_result.importances_mean

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
    _id_col = env.configs.data.get("id", "src_subject_id")
    _subject_ids = (
        df_filtered[_id_col].tolist()
        if _id_col in df_filtered.columns
        else df_filtered.index.tolist()
    )
    results = {
        "baseline": baseline_agg,
        model_name: model_agg,
        "baseline_folds": baseline_folds,
        f"{model_name}_folds": model_folds,
        "coefficients": coefficients,
        "feature_names": feature_names,
        "residualization": resid_info if resid_info else None,
        "subject_ids": _subject_ids,  # exact subjects used (for downstream consistency)
        "n_subjects": len(_subject_ids),
    }
    with open(reg_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save run metadata + config snapshot for reproducibility tracking
    overall_r = model_agg.get("overall", {}).get("pearson_r")
    save_run_metadata(
        env,
        results_dir=reg_dir,
        metrics={
            "pearson_r": overall_r,
            "n_samples": model_agg.get("n_samples"),
            "model_name": model_name,
        },
    )

    logger.info(f"Results saved to {reg_dir}\n")

    return {
        "baseline": baseline_agg,
        model_name: model_agg,
    }


def run_svr_on_saved_folds(
    fold_data: list,
    rng: np.random.RandomState,
    shuffle: bool = False,
    feature_idx: int | None = None,
    residualized: bool = False,
    model_cfg: dict | None = None,
) -> tuple:
    """Run SVR on pre-saved CV fold splits (fast permutation / feature importance).

    Uses pre-computed fold assignments — no ComBat, no residualization, just SVR.
    Uses config-driven kernel/C (default: linear, C=1.0) for speed.

    Args:
        fold_data: list of dicts with keys: X_train, X_test, y_train, y_test.
                   These should already be harmonized and scaled.
        rng: RandomState for reproducibility.
        shuffle: If True, shuffle y_train labels (permutation test).
        feature_idx: If given, permute only this feature column (feature importance).
        residualized: Whether predictions should skip lower-bound clipping.
        model_cfg: Optional SVR config dict (kernel, C, epsilon). Defaults to linear/1.0/0.1.

    Returns:
        (all_true, all_pred) arrays concatenated across folds.
    """
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler

    if model_cfg is None:
        model_cfg = {}
    kernel = model_cfg.get("kernel", "linear")
    C = model_cfg.get("C", 1.0)
    epsilon = model_cfg.get("epsilon", 0.1)

    all_true, all_pred = [], []
    for fold in fold_data:
        X_tr = fold["X_train"].copy()
        X_te = fold["X_test"].copy()
        y_tr = fold["y_train"].copy()
        y_te = fold["y_test"].copy()

        if shuffle:
            rng.shuffle(y_tr)

        if feature_idx is not None:
            # Permute test features (standard permutation importance: model
            # trained on intact data, evaluated on feature-shuffled test set)
            X_te[:, feature_idx] = rng.permutation(X_te[:, feature_idx])

        y_scaler = StandardScaler()
        y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()

        svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
        svr.fit(X_tr, y_tr_s)
        pred = y_scaler.inverse_transform(svr.predict(X_te).reshape(-1, 1)).ravel()

        if not residualized:
            pred = np.clip(pred, 0, None)

        all_true.extend(y_te)
        all_pred.extend(pred)

    return np.array(all_true), np.array(all_pred)


def run_regression_pipeline(env):
    """Run complete regression pipeline for all targets and models."""
    logger.info("=" * 60)
    logger.info("Regression Pipeline: Psychosis Severity Prediction")
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


def run_lateralization_comparison(
    df: pd.DataFrame,
    y: np.ndarray,
    env,
    valid_feature_cols: list,
    valid_pairs: list,
    covariate_cols: list | None = None,
    residualized: bool = False,
    fold_splits: list | None = None,
) -> dict:
    """Compare SVR performance across four lateralization feature sets with per-fold ComBat.

    Feature sets compared:
        - "Asymmetry only (AI)"  — AI = (L−R)/(L+R) per bilateral pair
        - "Total volume only"     — Total = L+R per bilateral pair
        - "AI + Total"            — both AI and Total concatenated
        - "Original L/R"          — raw left and right volumes

    Runs ComBat harmonization per fold to avoid leakage, then trains a linear SVR
    on each feature set. Uses config-driven n_splits, seed, and family-aware CV.

    Args:
        df: Subject-level dataframe (already filtered to target population).
        y: Target array (residualized if covariate_cols is not None).
        env: Environment with configs.
        valid_feature_cols: Ordered list of raw feature column names in df.
        valid_pairs: List of (name, left_col, right_col) bilateral pairs.
        covariate_cols: Columns to residualize target per fold (or None).
        residualized: Whether y is already residualized (controls clipping).
        fold_splits: Optional list of dicts with "train_idx"/"test_idx" from a prior
            run (e.g. from results.pkl svr_folds). When provided, reuses those exact
            splits so "Original L/R" r matches the main SVR exactly.

    Returns:
        dict keyed by feature set name, each value a dict:
            {"all_true": np.ndarray, "all_pred": np.ndarray, "r": float, "p": float}
    """
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    from .univariate import build_lateralization_feature_sets

    reg_config = env.configs.regression
    harm_config = env.configs.harmonize
    seed = env.configs.run.get("seed", 42)
    n_splits = reg_config.get("cv", {}).get("n_outer_splits", 5)
    site_col = harm_config.get("site_column", "mri_info_manufacturer")
    eb = harm_config.get("empirical_bayes", True)

    # Stratification key
    y_binned = pd.qcut(y, q=5, labels=False, duplicates="drop")
    stratify_key = y_binned
    if site_col in df.columns:
        site_codes = pd.Categorical(df[site_col]).codes
        n_sites = len(df[site_col].unique())
        combined_key = y_binned * n_sites + site_codes
        if pd.Series(combined_key).value_counts().min() >= n_splits:
            stratify_key = combined_key

    # Family-aware CV
    reg_config = env.configs.regression
    use_family_aware = reg_config.get("cv", {}).get("family_aware", True)

    family_groups = None
    if use_family_aware and "rel_family_id" in df.columns:
        family_groups = pd.to_numeric(df["rel_family_id"], errors="coerce").values
        missing = np.isnan(family_groups)
        if missing.any():
            max_id = np.nanmax(family_groups) if (~missing).any() else 0
            family_groups[missing] = np.arange(max_id + 1, max_id + 1 + missing.sum())
        outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    set_names = ["Asymmetry only (AI)", "Total volume only", "AI + Total", "Original L/R"]
    fold_results = {name: {"all_true": [], "all_pred": []} for name in set_names}

    # Build split iterator — reuse saved indices when provided for exact consistency
    if fold_splits is not None:
        splits_iter = [
            (np.array(fs["train_idx"]), np.array(fs["test_idx"]))
            for fs in fold_splits
        ]
        print(f"  Running {len(splits_iter)}-fold CV with ComBat per fold ({len(set_names)} feature sets) [reusing saved fold splits]...")
    else:
        split_args = (df, stratify_key, family_groups) if family_groups is not None else (df, stratify_key)
        splits_iter = list(outer_cv.split(*split_args))
        print(f"  Running {n_splits}-fold CV with ComBat per fold ({len(set_names)} feature sets)...")

    for fold_idx, (train_idx, test_idx) in enumerate(splits_iter):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        y_train = y[train_idx].copy()
        y_test = y[test_idx].copy()

        # Per-fold residualization
        if covariate_cols:
            resid_model = fit_residualize(y_train, df.iloc[train_idx], covariate_cols)
            y_train = apply_residualize(y_train, df.iloc[train_idx], covariate_cols, resid_model)
            y_test = apply_residualize(y_test, df.iloc[test_idx], covariate_cols, resid_model)

        # Per-fold ComBat
        X_train_raw = train_df[valid_feature_cols].values.astype(float)
        X_test_raw = test_df[valid_feature_cols].values.astype(float)

        train_covars = train_df[[site_col] + harm_config.get("covariates", [])].copy()
        train_covars = train_covars.rename(columns={site_col: "SITE"})
        test_covars = test_df[[site_col] + harm_config.get("covariates", [])].copy()
        test_covars = test_covars.rename(columns={site_col: "SITE"})
        for _cov in list(train_covars.columns):
            if _cov == "SITE":
                continue
            if not pd.api.types.is_numeric_dtype(train_covars[_cov]):
                train_covars[_cov] = pd.Categorical(train_covars[_cov]).codes.astype(float)
                test_covars[_cov] = pd.Categorical(test_covars[_cov]).codes.astype(float)

        try:
            combat_model_fold, X_train_harm = harmonizationLearn(X_train_raw, train_covars, eb=eb)
            X_test_harm = harmonizationApply(X_test_raw, test_covars, combat_model_fold)
        except Exception as _combat_err:
            import warnings
            warnings.warn(
                f"ComBat harmonization failed on fold {fold_idx} "
                f"({type(_combat_err).__name__}: {_combat_err}). "
                "Falling back to unharmonized features — results for this fold may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
            X_train_harm, X_test_harm = X_train_raw, X_test_raw

        train_sets = build_lateralization_feature_sets(X_train_harm, valid_feature_cols, valid_pairs)
        test_sets = build_lateralization_feature_sets(X_test_harm, valid_feature_cols, valid_pairs)

        for set_name in set_names:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_sets[set_name])
            X_te = scaler.transform(test_sets[set_name])
            y_scaler = StandardScaler()
            y_tr_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            svr = SVR(kernel="linear", C=1.0)
            svr.fit(X_tr, y_tr_s)
            pred = y_scaler.inverse_transform(svr.predict(X_te).reshape(-1, 1)).ravel()
            if not residualized:
                pred = np.clip(pred, 0, None)
            fold_results[set_name]["all_true"].extend(y_test)
            fold_results[set_name]["all_pred"].extend(pred)

    # Aggregate
    output = {}
    for set_name in set_names:
        all_true = np.array(fold_results[set_name]["all_true"])
        all_pred = np.array(fold_results[set_name]["all_pred"])
        r, p = pearsonr(all_true, all_pred)
        output[set_name] = {"all_true": all_true, "all_pred": all_pred, "r": r, "p": p}

    return output
