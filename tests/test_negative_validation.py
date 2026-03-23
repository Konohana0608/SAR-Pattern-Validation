from pathlib import Path

import pytest

from sar_pattern_validation.errors import (
    ConfigValidationError,
    CsvFormatError,
    WorkflowExecutionError,
)
from sar_pattern_validation.image_loader import SARImageLoader
from sar_pattern_validation.workflow_schema import validate_workflow_config
from sar_pattern_validation.workflows import complete_workflow


def test_loader_rejects_missing_header_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    good_csv = tmp_path / "good.csv"
    bad_csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    good_csv.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")

    with pytest.raises(CsvFormatError, match="recognizable x/y coordinate columns"):
        SARImageLoader(measured_path=str(bad_csv), reference_path=str(good_csv))


def test_loader_rejects_missing_sar_column(tmp_path: Path) -> None:
    missing_sar = tmp_path / "missing_sar.csv"
    good_csv = tmp_path / "good.csv"
    missing_sar.write_text("x,y,z\n0,0,1\n", encoding="utf-8")
    good_csv.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")

    with pytest.raises(CsvFormatError, match="Missing SAR column"):
        SARImageLoader(measured_path=str(missing_sar), reference_path=str(good_csv))


def test_complete_workflow_reports_missing_input_paths() -> None:
    with pytest.raises(WorkflowExecutionError, match="Input file\\(s\\) not found"):
        complete_workflow(
            measured_file_path="does/not/exist.csv",
            reference_file_path="also/missing.csv",
        )


def test_complete_workflow_rejects_unexpected_parameter_name() -> None:
    with pytest.raises(ConfigValidationError, match="extra_forbidden"):
        complete_workflow(
            measured_file_path="data/example/measured_sSAR1g.csv",
            reference_file_path="data/example/reference_sSAR1g.csv",
            unsupported_field=True,
        )


def test_complete_workflow_rejects_invalid_policy_value() -> None:
    with pytest.raises(ConfigValidationError, match="evaluation_roi_policy"):
        complete_workflow(
            measured_file_path="data/example/measured_sSAR1g.csv",
            reference_file_path="data/example/reference_sSAR1g.csv",
            evaluation_roi_policy="invalid_policy",
        )


def test_complete_workflow_rejects_invalid_log_level() -> None:
    with pytest.raises(ConfigValidationError, match="log_level"):
        complete_workflow(
            measured_file_path="data/example/measured_sSAR1g.csv",
            reference_file_path="data/example/reference_sSAR1g.csv",
            log_level="TRACE",
        )


def test_complete_workflow_rejects_invalid_stage_value() -> None:
    with pytest.raises(ConfigValidationError, match="translation_step"):
        complete_workflow(
            measured_file_path="data/example/measured_sSAR1g.csv",
            reference_file_path="data/example/reference_sSAR1g.csv",
            stages=[
                {
                    "translation_step": 0.0,
                    "rot_step_deg": 1.0,
                    "rot_span_deg": 90.0,
                    "tx_steps": 1,
                    "ty_steps": 1,
                }
            ],
        )


def test_complete_workflow_rejects_removed_affine_transform_type() -> None:
    with pytest.raises(ConfigValidationError, match="transform_type"):
        complete_workflow(
            measured_file_path="data/example/measured_sSAR1g.csv",
            reference_file_path="data/example/reference_sSAR1g.csv",
            transform_type="affine",
        )


def test_validate_workflow_config_allows_zero_rotation_fields_for_translate() -> None:
    config = validate_workflow_config(
        {
            "transform_type": "translate",
            "stages": [
                {
                    "translation_step": 0.001,
                    "rot_step_deg": 0.0,
                    "rot_span_deg": 0.0,
                    "tx_steps": 2,
                    "ty_steps": 2,
                }
            ],
        }
    )

    assert config.transform_type == "translate"
    assert config.stages[0]["rot_step_deg"] == 0.0
    assert config.stages[0]["rot_span_deg"] == 0.0
