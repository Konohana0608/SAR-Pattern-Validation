import json
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from sar_pattern_validation.workflow_config import WorkflowConfig
from sar_pattern_validation.workflows import _complete_workflow

from .helpers import compare_gamma_maps

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "measurement_validation"
LOG_DIR = ARTIFACT_DIR / "logs"
REGENERATE_ENV = "REGENERATE_MEASUREMENT_VALIDATION_ARTIFACTS"
GAMMA_DIFF_THRESHOLD_ENV = "MEASUREMENT_GAMMA_DIFF_THRESHOLD"
SAVE_PLOTS_ENV = "SAVE_MEASUREMENT_VALIDATION_PLOTS"
NOISE_FLOOR_WKG = 0.05


@dataclass(frozen=True)
class MeasurementValidationCase:
    case_id: str
    measured_csv: str
    reference_csv: str
    power_level_dbm: float


@dataclass(frozen=True)
class MeasurementValidationResult:
    actual: dict
    gamma_map: np.ndarray
    evaluation_mask: np.ndarray


@dataclass(frozen=True)
class Artifact:
    artifact_version: int
    dataset: dict
    inputs: dict
    success_thresholds: dict
    expected: dict


CASES = tuple(
    MeasurementValidationCase(
        case_id=f"2450_10mm_1g_{i + 1}",
        measured_csv=f"data/measurements/dipole_2450MHz_Flat_10mm_17dBm_1g_{i + 1}.csv",
        reference_csv="data/database/dipole_2450MHz_Flat_10mm_1g.csv",
        power_level_dbm=17.0,
    )
    for i in range(9)
)


def _artifact_json_path(case: MeasurementValidationCase) -> Path:
    return ARTIFACT_DIR / f"{case.case_id}_metrics.json"


def _artifact_gamma_path(case: MeasurementValidationCase) -> Path:
    return ARTIFACT_DIR / f"{case.case_id}_gamma_field.npz"


def _artifact_payload(case: MeasurementValidationCase, actual: dict) -> dict:
    return {
        "artifact_version": 1,
        "dataset": {
            "measured_csv": case.measured_csv,
            "reference_csv": case.reference_csv,
        },
        "inputs": {
            "power_level_dbm": case.power_level_dbm,
            "noise_floor_wkg": NOISE_FLOOR_WKG,
            "dose_to_agreement_percent": 10.0,
            "distance_to_agreement_mm": 3.0,
            "gamma_cap": 2.0,
            "evaluation_roi_policy": "intersection",
        },
        "success_thresholds": {
            "require_zero_failed_pixels": True,
        },
        "expected": actual,
    }


def _case_plot_dir(case: MeasurementValidationCase) -> Path:
    return ARTIFACT_DIR / "plots" / case.case_id


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _case_debug_log_path(case: MeasurementValidationCase, test_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_filename(test_name)
    return LOG_DIR / f"{timestamp}_{safe_name}_{case.case_id}.log"


@contextmanager
def _debug_file_logging(log_path: Path):
    root = logging.getLogger()
    prev_root_level = root.level

    reg_logger = logging.getLogger("Rigid2DRegistration")
    prev_reg_level = reg_logger.level
    matplotlib_logger = logging.getLogger("matplotlib")
    prev_matplotlib_level = matplotlib_logger.level

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    reg_logger.setLevel(logging.DEBUG)
    matplotlib_logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(prev_root_level)
        reg_logger.setLevel(prev_reg_level)
        matplotlib_logger.setLevel(prev_matplotlib_level)


def _write_artifacts(
    case: MeasurementValidationCase, result: MeasurementValidationResult
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _artifact_json_path(case).write_text(
        json.dumps(_artifact_payload(case, result.actual), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        _artifact_gamma_path(case),
        gamma_map=result.gamma_map.astype(np.float32),
        evaluation_mask=result.evaluation_mask.astype(np.uint8),
    )


def _load_artifact(case: MeasurementValidationCase) -> Artifact:
    data = json.loads(_artifact_json_path(case).read_text(encoding="utf-8"))
    return Artifact(**data)


def _load_gamma_field(case: MeasurementValidationCase) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(_artifact_gamma_path(case))
    return data["gamma_map"], data["evaluation_mask"]


def _compute_case(
    case: MeasurementValidationCase, *, save_plots: bool
) -> MeasurementValidationResult:
    plot_dir = _case_plot_dir(case)
    config = WorkflowConfig(
        measured_file_path=case.measured_csv,
        reference_file_path=case.reference_csv,
        power_level_dbm=case.power_level_dbm,
        log_level="DEBUG",
        noise_floor=NOISE_FLOOR_WKG,
        dose_to_agreement=10.0,
        distance_to_agreement=3.0,
        gamma_cap=2.0,
        registration_stage_policy="adaptive",
        adaptive_assume_axial_symmetry=True,
        evaluation_roi_policy="intersection",
        render_plots=save_plots,
        loaded_images_save_path=str(plot_dir / "01_loader_comparison.png")
        if save_plots
        else None,
        aligned_meas_save_path=str(plot_dir / "02_registered_measured.png")
        if save_plots
        else None,
        registered_image_save_path=str(plot_dir / "02_registration_overlay.png")
        if save_plots
        else None,
        gamma_comparison_image_path=str(plot_dir / "03_gamma_map.png")
        if save_plots
        else None,
        save_failures_overlay=save_plots,
    )
    wf = _complete_workflow(config)

    assert wf.gamma_map is not None
    assert wf.evaluation_mask is not None

    actual = {
        "evaluated_pixel_count": wf.evaluated_pixel_count,
        "passed_pixel_count": wf.passed_pixel_count,
        "failed_pixel_count": wf.failed_pixel_count,
        "pass_rate_percent": wf.pass_rate_percent,
        "measured_pssar": wf.measured_pssar,
        "reference_pssar": wf.reference_pssar,
        "scaling_error": wf.scaling_error,
    }
    return MeasurementValidationResult(
        actual=actual,
        gamma_map=wf.gamma_map.astype(np.float32),
        evaluation_mask=wf.evaluation_mask.astype(np.uint8),
    )


@pytest.fixture(params=CASES, ids=lambda case: case.case_id)
def measurement_case(request: pytest.FixtureRequest) -> MeasurementValidationCase:
    return request.param


@pytest.mark.slow
def test_measurement_workflow_cases_match_reference_artifacts(
    measurement_case: MeasurementValidationCase,
    request: pytest.FixtureRequest,
) -> None:
    case = measurement_case
    save_plots = os.getenv(SAVE_PLOTS_ENV, "1") == "1"
    log_path = _case_debug_log_path(case, request.node.name)
    with _debug_file_logging(log_path):
        result = _compute_case(case, save_plots=save_plots)

    actual = result.actual
    assert actual["failed_pixel_count"] == 0, (
        f"{case.case_id} expected zero failed pixels, got "
        f"{actual['failed_pixel_count']} out of {actual['evaluated_pixel_count']}"
    )

    if os.getenv(REGENERATE_ENV) == "1":
        _write_artifacts(case, result)
        return
    else:
        artifact = _load_artifact(case)
        assert artifact.dataset["measured_csv"] == case.measured_csv
        assert artifact.dataset["reference_csv"] == case.reference_csv
        assert artifact.inputs["evaluation_roi_policy"] == "intersection"
        assert (
            artifact.success_thresholds.get("require_zero_failed_pixels", True) is True
        )

        expected = artifact.expected
        assert expected is not None, f"{case.case_id} artifact missing expected results"
        assert actual["evaluated_pixel_count"] == expected["evaluated_pixel_count"]
        assert actual["passed_pixel_count"] == expected["passed_pixel_count"]
        assert actual["failed_pixel_count"] == expected["failed_pixel_count"]
        assert actual["pass_rate_percent"] == pytest.approx(
            expected["pass_rate_percent"], abs=1e-9
        )
        assert actual["measured_pssar"] == pytest.approx(
            expected["measured_pssar"], abs=1e-9
        )
        assert actual["reference_pssar"] == pytest.approx(
            expected["reference_pssar"], abs=1e-9
        )
        assert actual["scaling_error"] == pytest.approx(
            expected["scaling_error"], abs=1e-9
        )

        expected_gamma, expected_mask = _load_gamma_field(case)
        assert expected_gamma is not None, (
            f"{case.case_id} artifact missing expected gamma field"
        )
        assert expected_mask is not None, (
            f"{case.case_id} artifact missing expected mask"
        )

        compare_gamma_maps(
            expected_gamma, expected_mask, result.gamma_map, result.evaluation_mask
        )
