"""Persistence helpers for preprocessing outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


def _formats(env) -> list[str]:
    """Return output formats (parquet always, csv optional)."""

    run_cfg = getattr(env.configs, "run", {}) or {}
    outputs_cfg = run_cfg.get("outputs", {}) or {}

    formats: list[str] = ["parquet"]
    if outputs_cfg.get("include_csv"):
        formats.append("csv")
    return formats


def _run_dir(env, *parts: str) -> Path:
    run_cfg = getattr(env.configs, "run", {}) or {}
    run_name = run_cfg.get("run_name", "default")
    run_id = str(run_cfg.get("run_id", "run-unknown"))
    seed = run_cfg.get("seed", 42)

    base = env.repo_root / "outputs" / run_name / run_id / f"seed_{seed}"
    return base.joinpath(*parts)


def _write_dataframe(
    df: pd.DataFrame, base_path: Path, name: str, formats: Iterable[str]
) -> None:
    for fmt in formats:
        target = base_path / f"{name}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(target, index=False)
        elif fmt == "csv":
            df.to_csv(target, index=False)
        else:
            raise ValueError(f"Unsupported format '{fmt}'")


def save_processed_data(env, **datasets: pd.DataFrame) -> None:
    """Save all datasets under the run-scoped folder."""

    formats = _formats(env)
    run_dir = _run_dir(env, "datasets")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(datasets)} datasets to {run_dir}")
    for name, df in datasets.items():
        _write_dataframe(df, run_dir, name, formats)


def save_qc_artifacts(env, merged_df: pd.DataFrame, mask: pd.DataFrame) -> None:
    """Persist merged data snapshot and QC mask for the current run."""

    qc_dir = _run_dir(env, "artifacts", "qc")
    qc_dir.mkdir(parents=True, exist_ok=True)

    formats = _formats(env)
    _write_dataframe(merged_df, qc_dir, "merged_raw", formats)
    _write_dataframe(mask, qc_dir, "qc_mask", formats)


def save_split_map(env, split_map: pd.DataFrame) -> None:
    """Persist split assignments for the current run."""

    split_dir = _run_dir(env, "artifacts", "splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    formats = _formats(env)
    _write_dataframe(split_map, split_dir, "split_map", formats)


def save_provenance(env, qc_mask: pd.DataFrame, split_map: pd.DataFrame) -> None:
    """Persist provenance summaries for QC outcomes and splits."""

    provenance_dir = _run_dir(env, "provenance")
    provenance_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = getattr(env.configs, "run", {}) or {}
    run_id = str(run_cfg.get("run_id", "run-unknown"))
    seed = run_cfg.get("seed", 42)

    threshold = env.configs.data["qc_thresholds"]["surface_holes_max"]

    qc_doc = {
        "run_id": run_id,
        "thresholds": {"surface_holes_max": threshold},
        "counts": {
            "total": int(len(qc_mask)),
            "pass": int(qc_mask["qc_pass"].sum()),
            "fail": int((~qc_mask["qc_pass"]).sum()),
        },
        "fail_reasons": {
            (reason if reason is not None else "pass"): int(count)
            for reason, count in (
                qc_mask["qc_reason"].fillna("pass").value_counts().items()
            )
        },
    }

    split_doc = {
        "run_id": run_id,
        "seed": seed,
        "counts": split_map["split"].value_counts().to_dict(),
    }

    with open(provenance_dir / "qc.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(qc_doc, fh, sort_keys=False)

    with open(provenance_dir / "splits.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(split_doc, fh, sort_keys=False)
