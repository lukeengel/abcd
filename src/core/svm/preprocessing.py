"""Nested preprocessing for cross-validation folds."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from neuroHarmonize import harmonizationLearn, harmonizationApply

from ..tsne.embeddings import get_imaging_columns


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

    # PCA
    n_components = pca_config.get("n_components", 0.90)
    whiten = pca_config.get("whiten", False)
    pca = PCA(n_components=n_components, whiten=whiten, random_state=seed)
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

    # Apply PCA
    X_train_pca = fitted_pipeline["pca"].transform(X_train_scaled)
    X_val_pca = fitted_pipeline["pca"].transform(X_val_scaled)

    return X_train_pca, X_val_pca


def extract_harmonization_data(
    df: pd.DataFrame, env
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract imaging features and covariates for harmonization."""
    svm_config = env.configs.svm
    harm_config = env.configs.harmonize

    imaging_cols = get_imaging_columns(df, svm_config["imaging_prefixes"])
    X = df[imaging_cols].values

    site_col = harm_config["site_column"]
    covariate_cols = [site_col] + harm_config.get("covariates", [])
    covars = df[covariate_cols].copy()
    covars = covars.rename(columns={site_col: "SITE"})

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

    # PCA
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
