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


def load_or_compute_tsne(X: np.ndarray, name: str, embeddings_dir: Path, tsne_config: dict, seed: int) -> np.ndarray:
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
    """Prepare metadata for all datasets using column mappings from config."""

    # get column mappings from config
    col_map = env.configs.data["columns"]["mapping"]
    qc_cols = env.configs.data["columns"]["qc"]
    metadata_cols = env.configs.data["columns"]["metadata"]
    sex_map = env.configs.data["derived_variables"]["sex"]["map"]

    def extract_metadata(df: pd.DataFrame) -> dict:
        metadata = {}

        # QC metric (use first QC column if multiple)
        qc_col = qc_cols[0] if isinstance(qc_cols, list) else qc_cols
        if qc_col in df.columns:
            metadata["surface_holes"] = df[qc_col].values

        # scanner info (use first metadata column that contains 'manufacturer')
        scanner_col = next((col for col in metadata_cols if "manufacturer" in col.lower()), None)
        if scanner_col and scanner_col in df.columns:
            metadata["scanner"] = df[scanner_col].values

        # research groups
        metadata["research_groups"] = df[col_map["research_group"]].values

        # age
        if col_map["age"] in df.columns:
            metadata["age"] = df[col_map["age"]].astype(int).values

        # sex with configurable mapping
        if col_map["sex"] in df.columns:
            metadata["sex"] = df[col_map["sex"]].map(sex_map).fillna("Unknown").values

        return metadata

    return {
        "preqc": extract_metadata(baseline_preqc),
        "postqc": extract_metadata(all_orig),
        "harmonized": extract_metadata(all_orig),
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
