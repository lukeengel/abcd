"""Nested preprocessing for cross-validation folds."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from neuroHarmonize import harmonizationLearn, harmonizationApply

from ..tsne.embeddings import get_imaging_columns, get_roi_columns_from_config


# Fit PCA once on full dev set (not per-fold) to avoid "mixing apples and pears"
def fit_pca_on_dev(dev_df: pd.DataFrame, env, seed: int) -> dict:
    """Fit PCA transformation once on full dev set."""
    pca_config = env.configs.pca
    harm_config = env.configs.harmonize

    X, covars = extract_harmonization_data(dev_df, env)

    # Remove zero-variance features
    feature_vars = np.var(X, axis=0)
    valid_features = feature_vars > 1e-10
    X = X[:, valid_features]

    # Harmonization
    eb = harm_config.get("empirical_bayes", True)
    smooth_terms = harm_config.get("smooth_terms", [])
    combat_model, X_harm = harmonizationLearn(
        X, covars, eb=eb, smooth_terms=smooth_terms
    )

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_harm)

    # Skip PCA if disabled in config
    svm_config = env.configs.svm
    if not svm_config.get("use_pca", True):
        return {
            "combat_model": combat_model,
            "scaler": scaler,
            "pca": None,
            "valid_features": valid_features,
            "n_components": X_scaled.shape[1],
            "variance_explained": 1.0,
        }

    # PCA configuration
    n_components_config = pca_config.get("n_components", 0.90)

    whiten = pca_config.get("whiten", False)
    pca = PCA(n_components=n_components_config, whiten=whiten, random_state=seed)
    pca.fit(X_scaled)

    # Filter components by minimum variance threshold (optional)
    min_variance = pca_config.get("min_component_variance", None)

    if min_variance is not None:
        valid_components = pca.explained_variance_ratio_ >= min_variance
        n_valid = valid_components.sum()

        if n_valid < pca.n_components_:
            print(
                f"PCA: Filtered {pca.n_components_} → {n_valid} components "
                f"(min variance: {min_variance:.1%})"
            )
            # Refit with exact number
            pca = PCA(n_components=n_valid, whiten=whiten, random_state=seed)
            pca.fit(X_scaled)

    return {
        "combat_model": combat_model,
        "scaler": scaler,
        "pca": pca,
        "valid_features": valid_features,
        "n_components": pca.n_components_,
        "variance_explained": pca.explained_variance_ratio_.sum(),
    }


# Apply pre-fitted PCA to train/val folds
def apply_pca_to_fold(
    train_df: pd.DataFrame, val_df: pd.DataFrame, fitted_pipeline: dict, env
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pre-fitted PCA pipeline to a CV fold."""
    X_train, train_covars = extract_harmonization_data(train_df, env)
    X_val, val_covars = extract_harmonization_data(val_df, env)

    # Apply valid features mask
    X_train = X_train[:, fitted_pipeline["valid_features"]]
    X_val = X_val[:, fitted_pipeline["valid_features"]]

    # Apply harmonization
    X_train_harm = harmonizationApply(
        X_train, train_covars, fitted_pipeline["combat_model"]
    )
    X_val_harm = harmonizationApply(X_val, val_covars, fitted_pipeline["combat_model"])

    # Apply scaling
    X_train_scaled = fitted_pipeline["scaler"].transform(X_train_harm)
    X_val_scaled = fitted_pipeline["scaler"].transform(X_val_harm)

    # Apply PCA (skip if disabled)
    if fitted_pipeline["pca"] is not None:
        X_train_pca = fitted_pipeline["pca"].transform(X_train_scaled)
        X_val_pca = fitted_pipeline["pca"].transform(X_val_scaled)
    else:
        X_train_pca = X_train_scaled
        X_val_pca = X_val_scaled

    return X_train_pca, X_val_pca


def extract_harmonization_data(
    df: pd.DataFrame, env
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract imaging features and covariates for harmonization."""
    svm_config = env.configs.svm
    harm_config = env.configs.harmonize

    # ROI feature selection (same pattern as MLP/regression)
    roi_columns = None
    if svm_config.get("feature_mode") == "roi":
        roi_networks = svm_config.get("roi_networks", [])
        if roi_networks:
            roi_columns = get_roi_columns_from_config(env.configs.data, roi_networks)

    imaging_cols = get_imaging_columns(df, svm_config["imaging_prefixes"], roi_columns)
    X = df[imaging_cols].values

    site_col = harm_config["site_column"]
    covariate_cols = [site_col] + harm_config.get("covariates", [])
    covars = df[covariate_cols].copy()
    covars = covars.rename(columns={site_col: "SITE"})

    # Encode string covariates as float64 (neuroHarmonize requires numeric input;
    # int8 from pd.Categorical can cause OLS failures in some neuroHarmonize versions)
    for col in list(covars.columns):
        if col == "SITE":
            continue
        if not pd.api.types.is_numeric_dtype(covars[col]):
            covars[col] = pd.Categorical(covars[col]).codes.astype(float)
        else:
            covars[col] = covars[col].astype(float)
    # Drop constant covariates (e.g. sex when data is sex-stratified)
    for col in list(covars.columns):
        if col == "SITE":
            continue
        if covars[col].nunique() <= 1:
            covars = covars.drop(columns=col)

    return X, covars


def preprocess_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    env,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply Harmonize→Scale→PCA pipeline.

    Fits on train and transforms both splits.
    """
    pca_config = env.configs.pca
    harm_config = env.configs.harmonize

    # Extract data and covariates
    X_train, train_covars = extract_harmonization_data(train_df, env)
    X_val, val_covars = extract_harmonization_data(val_df, env)

    # Remove zero-variance features (fit on train only)
    feature_vars = np.var(X_train, axis=0)
    valid_features = feature_vars > 1e-10
    X_train = X_train[:, valid_features]
    X_val = X_val[:, valid_features]

    # Harmonization (using config parameters)
    eb = harm_config.get("empirical_bayes", True)
    smooth_terms = harm_config.get("smooth_terms", [])
    combat_model, X_train_harm = harmonizationLearn(
        X_train, train_covars, eb=eb, smooth_terms=smooth_terms
    )
    X_val_harm = harmonizationApply(X_val, val_covars, combat_model)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_harm)
    X_val_scaled = scaler.transform(X_val_harm)

    # PCA (skip if disabled)
    svm_config = env.configs.svm
    if not svm_config.get("use_pca", True):
        pipeline = {
            "combat_model": combat_model,
            "scaler": scaler,
            "pca": None,
            "valid_features": valid_features,
            "n_components": X_train_scaled.shape[1],
            "variance_explained": 1.0,
        }
        return X_train_scaled, X_val_scaled, pipeline

    pca = PCA(
        n_components=pca_config.get("n_components", 0.90),
        whiten=pca_config.get("whiten", False),
        random_state=seed,
    )
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    pipeline = {
        "combat_model": combat_model,
        "scaler": scaler,
        "pca": pca,
        "valid_features": valid_features,
        "n_components": pca.n_components_,
        "variance_explained": pca.explained_variance_ratio_.sum(),
    }

    return X_train_pca, X_val_pca, pipeline
