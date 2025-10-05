"""Core t-SNE embedding computation and persistence."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def get_imaging_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    """Get imaging column names based on prefixes."""
    return [col for col in df.columns if any(col.startswith(p) for p in prefixes)]


def load_or_compute_tsne(
    X: np.ndarray, name: str, embeddings_dir: Path, tsne_config: dict, seed: int
) -> np.ndarray:
    """Load existing t-SNE embedding or compute if needed."""
    complexity = tsne_config["complexity"]
    save_path = embeddings_dir / f"{name}_complexity{complexity}.pkl"

    if save_path.exists():
        print(f"Loading existing {name} embedding (complexity {complexity})...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    print(f"Computing {name} t-SNE embedding (complexity {complexity})...")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    X_scaled = StandardScaler().fit_transform(X)
    perplexity = min(complexity, max(5, (X.shape[0] - 1) // 3))

    tsne = TSNE(
        n_components=tsne_config["n_components"],
        random_state=seed,
        perplexity=perplexity,
        learning_rate=tsne_config["learning_rate"],
        init=tsne_config["init"],
    )
    embedding = tsne.fit_transform(X_scaled)

    with open(save_path, "wb") as f:
        pickle.dump(embedding, f)

    return embedding


def prepare_metadata(baseline_preqc: pd.DataFrame, all_orig: pd.DataFrame, env) -> dict:
    """Prepare metadata for all datasets."""

    def extract_metadata(df: pd.DataFrame) -> dict:
        return {
            "surface_holes": df["apqc_smri_topo_ndefect"].values,
            "scanner": df["mri_info_manufacturer"].values,
            "research_groups": df[
                env.configs.data["columns"]["mapping"]["research_group"]
            ].values,
            "age": df["demo_brthdat_v2"].astype(int).values,
            "sex": df["demo_sex_v2"]
            .map({1: "Male", 2: "Female"})
            .fillna("Unknown")
            .values,
        }

    return {
        "preqc": extract_metadata(baseline_preqc),
        "postqc": extract_metadata(all_orig),
        "harmonized": extract_metadata(all_orig),  # Same metadata as postqc
    }


def save_metadata(metadata: dict, embeddings_dir: Path, research_question: str) -> None:
    """Save metadata with research question aliases for compatibility."""
    # Add aliases so notebook can access by research question name
    enhanced_metadata = {}
    for phase_key, phase_data in metadata.items():
        enhanced_metadata[phase_key] = phase_data.copy()
        # Create alias: metadata['postqc']['anxiety'] -> research_groups
        enhanced_metadata[phase_key][research_question] = phase_data["research_groups"]

    metadata_path = embeddings_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(enhanced_metadata, f)
