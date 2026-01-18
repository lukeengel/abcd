"""Configuration loading utilities."""

from __future__ import annotations

import yaml
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


def initialize_notebook(
    run_name: str | None = None, *, regenerate_run_id: bool = False
):
    """Initialize project environment"""

    repo_root = _set_dir()
    loaded_configs = _load_configs()

    run_cfg = dict(loaded_configs.run)

    if run_name is not None:
        run_cfg["run_name"] = run_name
    run_cfg.setdefault("run_name", "anxiety")

    raw_run_id = run_cfg.get("run_id")
    if not isinstance(raw_run_id, str):
        raw_run_id = str(raw_run_id) if raw_run_id is not None else ""
    if regenerate_run_id or not raw_run_id or not raw_run_id.startswith("run-"):
        raw_run_id = f"run-{uuid4().hex[:10]}"
    run_cfg["run_id"] = raw_run_id

    _persist_run_config(repo_root, run_cfg)

    loaded_configs.run = run_cfg

    # Adapt data config based on research question
    _adapt_data_config(loaded_configs.data, run_cfg["run_name"])

    env = SimpleNamespace(
        repo_root=repo_root,
        configs=loaded_configs,
    )

    output_dir = _create_output_folder(env)
    print(f"Initialized notebook for run '{run_cfg['run_name']}'")
    print(f"Saved output summary to {output_dir}")
    return env


def _adapt_data_config(data_config: dict, run_name: str) -> None:
    """Adapt data configuration based on research question."""

    # Supported research questions
    valid_questions = ["anxiety", "psychosis", "comorbidity"]

    if run_name not in valid_questions:
        raise ValueError(
            f"Unknown research question: '{run_name}'. "
            f"Supported options: {', '.join(valid_questions)}. "
            f"To add a new research question, update _adapt_data_config() "
            f"in src/core/config.py and add color scheme in configs/tsne.yaml"
        )

    if run_name == "psychosis":
        # Psychosis: use psych_group derived from group_last_final
        data_config["columns"]["mapping"]["research_group"] = "psych_group"
        data_config["columns"]["derived"] = ["sex_mapped", "psych_group"]
        # Ensure group_last_final is in metadata
        if "group_last_final" not in data_config["columns"]["metadata"]:
            data_config["columns"]["metadata"].append("group_last_final")
    elif run_name == "anxiety":
        # Anxiety: use anx_group derived from t_score
        data_config["columns"]["mapping"]["research_group"] = "anx_group"
        data_config["columns"]["derived"] = ["sex_mapped", "anx_group"]
        # Remove group_last_final from metadata if present
        if "group_last_final" in data_config["columns"]["metadata"]:
            data_config["columns"]["metadata"].remove("group_last_final")
    elif run_name == "comorbidity":
        # Comorbidity: use comorbid_group (anxiety OR psychosis)
        data_config["columns"]["mapping"]["research_group"] = "comorbid_group"
        data_config["columns"]["derived"] = [
            "sex_mapped",
            "anx_group",
            "psych_group",
            "comorbid_group",
        ]
        # Ensure group_last_final is in metadata
        if "group_last_final" not in data_config["columns"]["metadata"]:
            data_config["columns"]["metadata"].append("group_last_final")


def _set_dir():
    """Set directory - find project root by looking for configs/ directory."""
    cwd = Path.cwd()

    # Walk up the directory tree until we find configs/
    current = cwd
    max_levels = 5  # Safety limit
    for _ in range(max_levels):
        if (current / "configs").exists():
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    # Fallback to original behavior
    raise FileNotFoundError(
        f"Could not find 'configs' directory. Started from: {cwd}\n"
        f"Make sure you're running from within the project directory."
    )


def _load_configs():
    """Load configuration from YAML files."""

    repo = _set_dir()
    config_dir = repo / "configs"

    configs = {}
    for file in config_dir.glob("*.yaml"):
        with open(file, encoding="utf-8") as fh:
            configs[file.stem] = yaml.safe_load(fh)

    return SimpleNamespace(**configs)


def _create_output_folder(env):
    """Create the run output directory."""

    run_cfg = env.configs.run
    run_name = run_cfg.get("run_name", "anxiety")
    run_id = str(run_cfg.get("run_id", "run-unknown"))
    output_dir = env.repo_root / "outputs" / run_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _persist_run_config(repo_root: Path, run_cfg: dict) -> None:
    """Persist run configuration back to configs/run.yaml for reuse."""

    config_path = repo_root / "configs" / "run.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(run_cfg, fh, sort_keys=False)
