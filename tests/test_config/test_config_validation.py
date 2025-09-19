"""Test configuration validation and loading."""

import pytest
from core.config import _load_configs, _set_dir, initialize_notebook


def test_set_dir_finds_configs():
    """Test that _set_dir correctly identifies repo root."""
    repo_root = _set_dir()
    assert (repo_root / "configs").exists()


def test_load_configs_structure():
    """Test that configs are loaded with correct structure."""
    configs = _load_configs()

    # Check that configs object is created
    assert configs is not None

    # Check that at least one config file is loaded
    config_attrs = [attr for attr in dir(configs) if not attr.startswith("_")]
    assert len(config_attrs) > 0, "No configuration files were loaded"


def test_initialize_notebook_returns_valid_env():
    """Test that initialize_notebook returns valid environment."""
    env = initialize_notebook("test_notebook", "test_run")

    assert hasattr(env, "repo_root")
    assert hasattr(env, "configs")
    assert (env.repo_root / "configs").exists()


def test_referenced_files_structure():
    """Test that file references are properly structured."""
    configs = _load_configs()

    # Skip if no data config or files section
    if not hasattr(configs, "data") or "files" not in configs.data:
        pytest.skip("No files section in configuration")

    files_config = configs.data["files"]

    # Check that files section contains lists
    for file_type, file_list in files_config.items():
        assert isinstance(file_list, list), f"File type '{file_type}' should be a list"
        for file_path in file_list:
            assert isinstance(
                file_path, str
            ), f"File path should be string, got {type(file_path)}"
            assert len(file_path) > 0, "File path should not be empty"


def test_column_mappings_structure():
    """Test that column mappings are properly structured."""
    configs = _load_configs()

    # Skip if no data config or columns section
    if not hasattr(configs, "data") or "columns" not in configs.data:
        pytest.skip("No columns section in configuration")

    columns = configs.data["columns"]

    # Check that the values are strings (actual column names)
    for key, value in columns.items():
        if isinstance(value, str):  # Skip nested structures
            assert len(value) > 0, f"Column mapping {key} should not be empty"


def test_dtypes_are_valid():
    """Test that dtypes are valid pandas types."""
    configs = _load_configs()
    dtypes = configs.data["dtypes"]

    # Valid pandas dtypes
    valid_dtypes = {
        "string",
        "category",
        "int8",
        "int16",
        "int32",
        "int64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",  # Nullable integers
        "float16",
        "float32",
        "float64",
        "bool",
    }

    for column, dtype in dtypes.items():
        assert dtype in valid_dtypes, f"Invalid dtype for {column}: {dtype}"


def test_cross_reference_dtypes_and_columns():
    """Test that dtypes reference valid column mappings."""
    configs = _load_configs()
    dtypes = configs.data["dtypes"]
    columns = configs.data["columns"]

    # Get all valid column names from column mappings
    valid_columns = set()
    for value in columns.values():
        if isinstance(value, str):
            valid_columns.add(value)

    # Check that dtype keys (except generic 'imaging') reference valid columns
    for dtype_col in dtypes.keys():
        if dtype_col != "imaging":  # Skip generic imaging dtype
            assert (
                dtype_col in valid_columns
            ), f"Dtype column '{dtype_col}' not found in column mappings"


def test_feature_families_structure():
    """Test that feature families have valid patterns."""
    configs = _load_configs()
    feature_families = configs.data.get("feature_families", {})

    for family_name, patterns in feature_families.items():
        assert isinstance(
            patterns, list
        ), f"Feature family '{family_name}' should be a list"
        assert len(patterns) > 0, f"Feature family '{family_name}' should not be empty"

        for pattern in patterns:
            assert isinstance(
                pattern, str
            ), f"Pattern in '{family_name}' should be string, got {type(pattern)}"
            assert len(pattern) > 0, f"Empty pattern in feature family '{family_name}'"


def test_config_sections_are_dictionaries():
    """Test that config sections are properly structured as dictionaries."""
    configs = _load_configs()

    # Test each loaded config section
    for attr_name in dir(configs):
        if not attr_name.startswith("_"):
            config_section = getattr(configs, attr_name)
            assert isinstance(
                config_section, dict
            ), f"Config section '{attr_name}' should be a dictionary"


def test_output_folder_creation():
    """Test that output folder creation works."""
    from core.config import _create_output_folder

    env = initialize_notebook("test_validation", "validation_test")
    output_dir = _create_output_folder(env, "test_validation")

    assert output_dir.exists(), "Output folder was not created"
    assert (
        output_dir.parent.name == "outputs"
    ), "Output folder not in correct parent directory"
