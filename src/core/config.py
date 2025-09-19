"""Configuration loading utilities"""

import yaml
from pathlib import Path
from types import SimpleNamespace


def initialize_notebook(notebook: str, name: str):
    """Initialize project with reproducibility"""

    repo_root = _set_dir()
    loaded_configs = _load_configs()

    # Create unified namespace with variables
    env = SimpleNamespace(
        repo_root=repo_root,
        configs=loaded_configs,
    )

    print(
        f"Initialized notebook: {notebook} \n"
        f'Use "env.variable" to access project variables\n'
        f'Use "env.configs.yaml_file[key]" to access yaml files'
    )
    print(f"Saved output summary to {_create_output_folder(env, name)}")
    return env


def save_summary(env):
    """Save summary of environment"""

    return None


def _set_dir():
    """Set directory"""
    cwd = Path.cwd()
    repo_root = cwd if (cwd / "configs").exists() else cwd.parent
    assert (repo_root / "configs").exists(), f"Couldn't find 'configs/' in {repo_root}"
    return repo_root


def _load_configs():
    """Load configuration from YAML files."""

    repo = _set_dir()
    config_dir = repo / "configs"

    configs = {}
    for file in config_dir.glob("*.yaml"):
        with open(file) as f:
            configs[file.stem] = yaml.safe_load(f)

    return SimpleNamespace(**configs)


def _create_output_folder(env, name: str):
    """Creates output folder from run"""

    output_dir = env.repo_root / "outputs" / f'{name}:{env.configs.run["run_id"]}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
