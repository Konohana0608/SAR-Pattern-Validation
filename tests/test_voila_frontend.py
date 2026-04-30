import json
import os
from pathlib import Path

import pandas as pd

from sar_pattern_validation.sample_catalog import (
    DatabaseSampleCatalog,
    DatabaseSampleColumn,
    DatabaseSampleFilterOption,
    DatabaseSampleFilters,
)
from sar_pattern_validation.voila_frontend.runner import SarPatternValidationRunner
from sar_pattern_validation.voila_frontend.runtime import WorkspacePaths
from sar_pattern_validation.voila_frontend.state import load_or_migrate_ui_state


def test_database_sample_catalog_scans_and_filters(tmp_path: Path) -> None:
    database_path = tmp_path / "database"
    database_path.mkdir()
    for name in (
        "dipole_900MHz_Flat_15mm_1g.csv",
        "dipole_900MHz_Flat_15mm_10g.csv",
        "patch_2450MHz_Flat_5mm_1g.csv",
    ):
        (database_path / name).write_text("x,y,sar\n0,0,1\n", encoding="utf-8")

    catalog = DatabaseSampleCatalog.scan(database_path)

    assert len(catalog.samples) == 3
    unique_values = catalog.unique_entries_in_columns()
    assert unique_values[DatabaseSampleColumn.ANTENNA_TYPE.value] == ["dipole", "patch"]

    filtered = catalog.filter_dataframe(
        DatabaseSampleFilters(
            frequency=DatabaseSampleFilterOption(
                column_name=DatabaseSampleColumn.FREQUENCY,
                value=900.0,
            )
        )
    )
    assert len(filtered) == 2


def test_runner_uses_remote_default_and_local_override(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_workspace(tmp_path)
    runner = SarPatternValidationRunner(paths)

    original_mode = os.environ.get("SAR_PATTERN_VALIDATION_BACKEND_MODE")
    original_local = os.environ.get("SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE")
    original_url = os.environ.get("GITHUB_PACKAGE_URL")
    original_branch = os.environ.get("BRANCH")
    try:
        os.environ.pop("SAR_PATTERN_VALIDATION_BACKEND_MODE", None)
        os.environ["GITHUB_PACKAGE_URL"] = "https://example.invalid/repo"
        os.environ["BRANCH"] = "feature"
        assert runner.backend_source_spec() == "git+https://example.invalid/repo@feature"

        os.environ["SAR_PATTERN_VALIDATION_BACKEND_MODE"] = "local"
        os.environ["SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE"] = "/tmp/local-repo"
        assert runner.backend_source_spec() == "/tmp/local-repo"
        assert runner.build_command("--help")[:5] == [
            "uvx",
            "--no-cache",
            "--from",
            "/tmp/local-repo",
            "sar-pattern-validation",
        ]
    finally:
        _restore_env("SAR_PATTERN_VALIDATION_BACKEND_MODE", original_mode)
        _restore_env("SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE", original_local)
        _restore_env("GITHUB_PACKAGE_URL", original_url)
        _restore_env("BRANCH", original_branch)


def test_load_or_migrate_ui_state_from_legacy_files(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    paths = WorkspacePaths.from_workspace(workspace_root)
    paths.system_state_dir.mkdir(parents=True, exist_ok=True)

    paths.legacy_state_path.write_text(
        json.dumps(
            {
                "measured_file_name": "measured.csv",
                "power_level": 17.5,
                "timestamp": 123.0,
            }
        ),
        encoding="utf-8",
    )
    paths.legacy_workflow_results_path.write_text(
        json.dumps(
            {
                "pass_rate_percent": 98.0,
                "evaluated_pixel_count": 100,
                "passed_pixel_count": 98,
                "failed_pixel_count": 2,
                "gamma_image_path": str(workspace_root / "images" / "gamma.png"),
                "failure_image_path": None,
                "registered_overlay_path": None,
                "loaded_images_path": None,
                "reference_image_path": None,
                "measured_image_path": None,
                "aligned_measured_path": None,
                "measured_pssar": 1.0,
                "reference_pssar": 1.1,
                "scaling_error": 0.1,
                "dose_to_agreement": 5.0,
                "distance_to_agreement": 2.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                DatabaseSampleColumn.ANTENNA_TYPE.value: "dipole",
                DatabaseSampleColumn.FREQUENCY.value: 900.0,
                DatabaseSampleColumn.DISTANCE.value: 15.0,
                DatabaseSampleColumn.MASS.value: 1.0,
                DatabaseSampleColumn.FILE_PATH.value: "/tmp/reference.csv",
            }
        ]
    ).to_csv(paths.legacy_filtered_db_csv_path, index=False)

    state = load_or_migrate_ui_state(paths)

    assert state is not None
    assert state.measured_file_name == "measured.csv"
    assert state.power_level == 17.5
    assert state.last_result is not None
    assert state.active_filters.antenna_type is not None
    assert state.active_filters.antenna_type.value == "dipole"
    assert paths.ui_state_path.exists()


def test_notebook_bootstrap_is_thin_and_v2_only() -> None:
    notebook = json.loads(Path("notebooks/voila.ipynb").read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "root_validator" not in sources
    assert "validator(" not in sources
    assert ".dict(" not in sources
    assert "class Config:" not in sources
    assert 'pydantic>=2.8,<3' in sources
    assert "bootstrap_voila_ui" in sources


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
