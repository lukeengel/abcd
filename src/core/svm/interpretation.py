"""Feature importance analysis and brain region mapping."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_feature_importance_linear(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importance from linear model coefficients."""
    if hasattr(model, "coef_"):
        coef = model.coef_
        # Handle binary vs multiclass
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            # Average across classes for multiclass
            importance = np.abs(coef).mean(axis=0)

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    raise ValueError("Model does not have linear coefficients")


def get_feature_importance_permutation(
    model, X, y, feature_names: list[str], seed: int, n_repeats: int = 10
) -> pd.DataFrame:
    """Compute permutation importance for any model (linear or RBF).

    Uses 10 repeats for robust importance estimates with confidence intervals.
    """
    n_features = len(feature_names)
    print(f"Computing permutation importance ({n_repeats} repeats, {n_features} features)...")

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    print("Permutation importance complete!")

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def map_pca_to_brain_regions(
    pca_importance: pd.DataFrame,
    pca_model,
    original_feature_names: list[str],
    top_n_components: int = 10,
    top_n_features: int = 20,
) -> pd.DataFrame:
    """Map PCA component importance back to original brain regions."""
    top_components = pca_importance.head(top_n_components)

    brain_region_scores = {}
    for _, row in top_components.iterrows():
        pc_idx = int(row["feature"].replace("PC", "")) - 1  # PC1 -> index 0
        pc_importance = row["importance"]

        # Get loadings for this component
        loadings = pca_model.components_[pc_idx]

        # Weight original features by PC importance
        for feat_idx, loading in enumerate(loadings):
            feat_name = original_feature_names[feat_idx]
            contribution = abs(loading) * pc_importance
            brain_region_scores[feat_name] = (
                brain_region_scores.get(feat_name, 0) + contribution
            )

    # Create DataFrame and sort
    df = pd.DataFrame(
        [
            {"brain_region": feat, "importance": score}
            for feat, score in brain_region_scores.items()
        ]
    )
    return (
        df.sort_values("importance", ascending=False)
        .head(top_n_features)
        .reset_index(drop=True)
    )
