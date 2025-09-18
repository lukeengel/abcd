"""Config-driven data preprocessing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import fnmatch
import pandas as pd
import yaml

# Type aliases for readability
FilterFn = Callable[[pd.DataFrame, Mapping[str, object]], pd.DataFrame]
DerivedFn = Callable[[pd.DataFrame, Mapping[str, object]], pd.Series]


@dataclass(frozen=True)
class PreprocessResult:
    dataframe: pd.DataFrame
    feature_families: Dict[str, Sequence[str]]
    config: MutableMapping[str, object]


def load_config(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_tables(files: Sequence[Path], index_cols: Sequence[str]) -> pd.DataFrame:
    frames = []
    for file_path in files:
        table = pd.read_csv(file_path)
        missing = [col for col in index_cols if col not in table.columns]
        if missing:
            raise ValueError(f"{file_path} missing index columns: {missing}")
        frames.append(table.set_index(index_cols))
    if not frames:
        return pd.DataFrame(columns=index_cols)
    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()].reset_index()
    return merged


def _default_derived_registry() -> Dict[str, DerivedFn]:
    def anx_group(df: pd.DataFrame, cfg: Mapping[str, object]) -> pd.Series:
        scores = pd.to_numeric(df[cfg["columns"]["t_score"]], errors="coerce")
        bins = [-float("inf"), 59, 64, float("inf")]
        labels = ["control", "subclinical", "clinical"]
        return pd.cut(scores, bins=bins, labels=labels, right=True).astype("string")

    return {"anx_group": anx_group}


def _apply_derived_columns(
    df: pd.DataFrame,
    cfg: Mapping[str, object],
    derived_fns: Optional[Mapping[str, DerivedFn]],
) -> pd.DataFrame:
    requested = cfg.get("columns", {}).get("derived_columns", [])
    registry = {"anx_group": _default_derived_registry()["anx_group"]}
    if derived_fns:
        registry.update(derived_fns)
    for name in requested:
        if name not in registry:
            raise KeyError(f"Derived column '{name}' is not registered")
        df[name] = registry[name](df, cfg)
    return df


def _apply_filters(
    df: pd.DataFrame, cfg: Mapping[str, object], filters: Optional[Iterable[FilterFn]]
) -> pd.DataFrame:
    for flt in filters or ():
        df = flt(df, cfg)
    return df


def _collect_feature_families(
    df: pd.DataFrame, cfg: Mapping[str, object]
) -> Dict[str, Sequence[str]]:
    families = {}
    for family, patterns in (cfg.get("feature_families") or {}).items():
        matches = set()
        for pattern in patterns:
            matches.update(fnmatch.filter(df.columns, pattern))
        families[family] = sorted(matches)
    return families


def preprocess(
    config_path: Path | str = "configs/data.yaml",
    data_root: Path | str = ".",
    filters: Optional[Iterable[FilterFn]] = None,
    derived_columns: Optional[Mapping[str, DerivedFn]] = None,
) -> PreprocessResult:
    root = Path(data_root)
    config_file = Path(config_path)
    config = load_config(config_file)

    columns = config.get("columns", {})
    index_cols = [columns["id"], columns["timepoint"]]

    file_groups = config.get("files", {})
    metadata_files = [root / Path(p) for p in file_groups.get("metadata::", [])]
    imaging_files = [root / Path(p) for p in file_groups.get("imaging", [])]

    metadata = _read_tables(metadata_files, index_cols)
    imaging = _read_tables(imaging_files, index_cols)

    if metadata.empty:
        combined = imaging
    elif imaging.empty:
        combined = metadata
    else:
        combined = metadata.merge(imaging, on=index_cols, how="outer")

    combined = _apply_derived_columns(combined, config, derived_columns)
    combined = _apply_filters(combined, config, filters)
    feature_families = _collect_feature_families(combined, config)

    return PreprocessResult(combined, feature_families, config)


__all__ = ["PreprocessResult", "preprocess", "load_config"]
