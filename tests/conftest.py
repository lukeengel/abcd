"""Pytest fixtures for testing"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from types import SimpleNamespace
import shutil


@pytest.fixture
def mock_env():
    """Create a mock environment with temporary CSV files for testing."""

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create test data
    metadata_df1 = pd.DataFrame(
        {
            "src_subject_id": ["SUBJ001", "SUBJ002", "SUBJ003"],
            "eventname": [
                "baseline_year_1_arm_1",
                "baseline_year_1_arm_1",
                "baseline_year_1_arm_1",
            ],
            "demo_sex_v2": [1, 2, 1],
            "demo_brthdat_v2": [10, 9, 10],
        }
    )

    metadata_df2 = pd.DataFrame(
        {
            "src_subject_id": ["SUBJ001", "SUBJ002", "SUBJ003"],
            "eventname": [
                "baseline_year_1_arm_1",
                "baseline_year_1_arm_1",
                "baseline_year_1_arm_1",
            ],
            "cbcl_scr_dsm5_anxdisord_t": [55.5, 68.2, 42.1],
        }
    )

    imaging_df1 = pd.DataFrame(
        {
            "src_subject_id": ["SUBJ001", "SUBJ002", "SUBJ003"],
            "eventname": [
                "baseline_year_1_arm_1",
                "baseline_year_1_arm_1",
                "baseline_year_1_arm_1",
            ],
            "dmri_dti_fa_001": [0.45, 0.52, 0.38],
            "dmri_dti_fa_002": [0.41, 0.49, 0.35],
        }
    )

    # Write files
    temp_files = {"metadata": [], "imaging": []}

    for i, df in enumerate([metadata_df1, metadata_df2]):
        file_path = Path(temp_dir) / f"metadata_{i}.csv"
        df.to_csv(file_path, index=False)
        temp_files["metadata"].append(str(file_path))

    file_path = Path(temp_dir) / "imaging_1.csv"
    imaging_df1.to_csv(file_path, index=False)
    temp_files["imaging"].append(str(file_path))

    # Create mock environment
    env = SimpleNamespace(
        repo_root=Path(temp_dir),
        configs=SimpleNamespace(
            data={
                "files": temp_files,
                "columns": {
                    "id": "src_subject_id",
                    "timepoint": "eventname",
                    "sex": "demo_sex_v2",
                    "age": "demo_brthdat_v2",
                    "t_score": "cbcl_scr_dsm5_anxdisord_t",
                },
                "dtypes": {
                    "src_subject_id": "string",
                    "eventname": "category",
                    "demo_sex_v2": "category",
                    "demo_brthdat_v2": "int16",
                    "cbcl_scr_dsm5_anxdisord_t": "float32",
                },
            }
        ),
    )

    yield env

    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def expected_merged_data():
    """Expected result from merging the mock data."""
    return pd.DataFrame(
        {
            "src_subject_id": ["SUBJ001", "SUBJ002", "SUBJ003"],
            "eventname": [
                "baseline_year_1_arm_1",
                "baseline_year_1_arm_1",
                "baseline_year_1_arm_1",
            ],
            "demo_sex_v2": [1, 2, 1],
            "demo_brthdat_v2": [10, 9, 10],
            "cbcl_scr_dsm5_anxdisord_t": [55.5, 68.2, 42.1],
            "dmri_dti_fa_001": [0.45, 0.52, 0.38],
            "dmri_dti_fa_002": [0.41, 0.49, 0.35],
        }
    )
