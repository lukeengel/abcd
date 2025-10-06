"""Nested preprocessing for cross-validation folds."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from neuroHarmonize import harmonizationLearn, harmonizationApply

from ..tsne.embeddings import get_imaging_columns


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
