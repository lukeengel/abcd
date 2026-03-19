"""Run metadata tracking for reproducible experiments.

Saves a run_metadata.yaml and config_snapshot.yaml in the results directory
so every run is self-documenting and comparable to prior runs.

Usage in pipeline (end of run_target_with_nested_cv):
    from .run_tracker import save_run_metadata
    save_run_metadata(env, results_dir, description="baseline SVR run")

CLI listing:
    from src.core.regression.run_tracker import list_runs
    for meta in list_runs("outputs", run_name="regression"):
        print(meta.run_id, meta.pearson_r, meta.description)
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunMetadata:
    """Metadata for a single regression run."""

    run_id: str
    run_name: str
    seed: int
    timestamp: str  # ISO-8601 UTC
    git_commit: str  # "unknown" if not in a git repo
    config_hash: str  # SHA-256 of serialized configs (first 12 chars)
    description: str = ""
    changes_from_last_run: str = ""
    # Key metrics populated after run completes (optional)
    pearson_r: float | None = None
    pearson_p_emp: float | None = None
    n_samples: int | None = None
    model_name: str | None = None
    # Extra key-value store for ad-hoc notes
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_commit(repo_root: str | Path | None = None) -> str:
    """Return current git commit hash (short), or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _config_hash(configs: dict) -> str:
    """SHA-256 of JSON-serialized config dict (first 12 chars)."""
    try:
        serialized = json.dumps(configs, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def _serialize_configs(env) -> dict:
    """Extract all configs from env into a plain dict."""
    out = {}
    for attr in ("data", "regression", "harmonize", "run"):
        cfg = getattr(env.configs, attr, None)
        if cfg is not None:
            out[attr] = cfg
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_run_metadata(
    env,
    results_dir: str | Path,
    description: str = "",
    changes_from_last_run: str = "",
    metrics: dict | None = None,
) -> RunMetadata:
    """Save run_metadata.yaml and config_snapshot.yaml to results_dir.

    Args:
        env: Environment object with env.configs.{data, regression, harmonize, run}.
        results_dir: Directory where results are stored for this run.
        description: Human-readable description of what this run tests.
        changes_from_last_run: Free-text diff from previous run.
        metrics: Optional dict with pearson_r, pearson_p_emp, n_samples, model_name.

    Returns:
        RunMetadata instance (also written to disk).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = env.configs.run
    all_configs = _serialize_configs(env)

    meta = RunMetadata(
        run_id=run_cfg.get("run_id", "unknown"),
        run_name=run_cfg.get("run_name", "unknown"),
        seed=run_cfg.get("seed", 42),
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        git_commit=_git_commit(),
        config_hash=_config_hash(all_configs),
        description=description or run_cfg.get("description", ""),
        changes_from_last_run=changes_from_last_run,
    )

    if metrics:
        meta.pearson_r = metrics.get("pearson_r")
        meta.pearson_p_emp = metrics.get("pearson_p_emp")
        meta.n_samples = metrics.get("n_samples")
        meta.model_name = metrics.get("model_name")

    # Write run_metadata.yaml
    meta_path = results_dir / "run_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta.to_dict(), f, default_flow_style=False, sort_keys=True)

    # Write config_snapshot.yaml (full config at time of run)
    snapshot_path = results_dir / "config_snapshot.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(all_configs, f, default_flow_style=False, sort_keys=True)

    logger.info(f"Run metadata saved: {meta_path}")
    return meta


def load_run_metadata(results_dir: str | Path) -> RunMetadata | None:
    """Load run_metadata.yaml from a results directory. Returns None if not found."""
    meta_path = Path(results_dir) / "run_metadata.yaml"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        d = yaml.safe_load(f)
    return RunMetadata.from_dict(d)


def list_runs(
    outputs_root: str | Path,
    run_name: str | None = None,
) -> list[RunMetadata]:
    """List all runs under outputs_root, optionally filtered by run_name.

    Walks outputs_root looking for run_metadata.yaml files and returns them
    sorted by timestamp (oldest first).

    Args:
        outputs_root: Root outputs directory (e.g. "outputs/" or "results/").
        run_name: If set, only return runs with matching run_name.

    Returns:
        List of RunMetadata, sorted by timestamp ascending.
    """
    outputs_root = Path(outputs_root)
    metas: list[RunMetadata] = []

    for meta_path in sorted(outputs_root.rglob("run_metadata.yaml")):
        meta = load_run_metadata(meta_path.parent)
        if meta is None:
            continue
        if run_name is not None and meta.run_name != run_name:
            continue
        metas.append(meta)

    metas.sort(key=lambda m: m.timestamp)
    return metas


def compare_runs(
    run_a: RunMetadata,
    run_b: RunMetadata,
) -> dict:
    """Return a dict highlighting differences between two runs."""
    fields = ["seed", "git_commit", "config_hash", "description",
              "pearson_r", "pearson_p_emp", "n_samples", "model_name"]
    diff = {}
    for f in fields:
        va, vb = getattr(run_a, f, None), getattr(run_b, f, None)
        if va != vb:
            diff[f] = {"run_a": va, "run_b": vb}
    return diff
