"""Core PCA computation and persistence."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_pca(
    X_train: np.ndarray, pca_dir: Path, pca_config: dict, seed: int
) -> tuple[PCA, StandardScaler]:
    """Fit PCA on training data and return fitted models."""
    print(f"Fitting PCA with n_components={pca_config['n_components']}...")
    pca_dir.mkdir(parents=True, exist_ok=True)

    # Fit scaler on training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Fit PCA on scaled training data
    pca = PCA(
        n_components=pca_config["n_components"],
        whiten=pca_config["whiten"],
        random_state=seed,
    )
    pca.fit(X_scaled)

    # Save fitted models
    with open(pca_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(pca_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    n_components_actual = pca.n_components_
    variance_explained = pca.explained_variance_ratio_.sum()
    print(
        f"PCA fitted: {n_components_actual} components "
        f"explain {variance_explained:.1%} variance"
    )

    return pca, scaler


def load_or_fit_pca(
    X_train: np.ndarray, pca_dir: Path, pca_config: dict, seed: int
) -> tuple[PCA, StandardScaler]:
    """Load existing PCA models or fit if needed."""
    scaler_path = pca_dir / "scaler.pkl"
    pca_path = pca_dir / "pca_model.pkl"

    if scaler_path.exists() and pca_path.exists():
        print("Loading existing PCA models...")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        n_components_actual = pca.n_components_
        variance_explained = pca.explained_variance_ratio_.sum()
        print(
            f"Loaded PCA: {n_components_actual} components "
            f"explain {variance_explained:.1%} variance"
        )
        return pca, scaler

    return fit_pca(X_train, pca_dir, pca_config, seed)


def transform_and_save(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    name: str,
    pca_dir: Path,
) -> np.ndarray:
    """Transform data using fitted PCA and save."""
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    save_path = pca_dir / f"{name}_pca.npy"
    np.save(save_path, X_pca)
    print(f"Saved {name} PCA: {X_pca.shape}")

    return X_pca
