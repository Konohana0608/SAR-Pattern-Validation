from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

matplotlib.use("Agg")

import sar_pattern_validation.workflows as workflows_module
from sar_pattern_validation.errors import ConfigValidationError
from sar_pattern_validation.gamma_eval import GammaMapEvaluator
from sar_pattern_validation.image_loader import SARImageLoader
from sar_pattern_validation.workflow_config import PlottingConfig
from sar_pattern_validation.workflow_schema import validate_workflow_config
from sar_pattern_validation.workflows import _apply_roi_policy, complete_workflow

from .helpers import (
    gaussian_2d,
    make_rect_grid,
    punch_rect_hole,
    rigid_transform_points,
    write_sar_csv,
)


def _make_image(arr: np.ndarray) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((0.001, 0.001))
    return img


def _write_synthetic_workflow_pair(
    tmp_path: Path,
    *,
    tx: float = 0.0,
    ty: float = 0.0,
    theta_deg: float = 0.0,
) -> tuple[Path, Path]:
    x, y = make_rect_grid(xmin=-0.04, xmax=0.04, ymin=-0.04, ymax=0.04, step=0.005)
    _, _, main_peak = gaussian_2d(
        x, y, x0=0.012, y0=-0.008, sx=0.018, sy=0.022, peak=1.0
    )
    _, _, side_peak = gaussian_2d(
        x, y, x0=-0.016, y0=0.018, sx=0.012, sy=0.016, peak=0.6
    )
    Z = main_peak + side_peak

    reference_csv = tmp_path / "reference.csv"
    write_sar_csv(reference_csv, x, y, Z)

    measured_df = pd.read_csv(reference_csv)
    if tx != 0.0 or ty != 0.0 or theta_deg != 0.0:
        measured_df = rigid_transform_points(
            measured_df, tx=tx, ty=ty, theta_deg=theta_deg
        )

    measured_csv = tmp_path / "measured.csv"
    measured_df.to_csv(measured_csv, index=False)
    return measured_csv, reference_csv


def _write_truncated_support_workflow_pair(tmp_path: Path) -> tuple[Path, Path]:
    measured_csv, reference_csv = _write_synthetic_workflow_pair(tmp_path)
    measured_df = pd.read_csv(measured_csv)
    measured_df = punch_rect_hole(
        measured_df,
        xmin=0.015,
        xmax=0.040,
        ymin=-0.040,
        ymax=0.040,
    )
    measured_df.to_csv(measured_csv, index=False)
    return measured_csv, reference_csv


def test_validate_workflow_config_rejects_negative_distance() -> None:
    with pytest.raises(ConfigValidationError):
        validate_workflow_config({"distance_to_agreement": -1.0})


def test_validate_workflow_config_accepts_plotting_config() -> None:
    config = validate_workflow_config(
        {
            "plotting": {
                "window_mm": (-40, 40, -35, 35),
                "font_size": 16,
                "save_dpi": 180,
            }
        }
    )

    assert config.plotting.window_mm == (-40.0, 40.0, -35.0, 35.0)
    assert config.plotting.font_size == 16
    assert config.plotting.save_dpi == 180


def test_validate_workflow_config_defaults_to_intersection_roi() -> None:
    config = validate_workflow_config({})

    assert config.evaluation_roi_policy == "intersection"


def test_validate_workflow_config_rejects_invalid_plotting_config() -> None:
    with pytest.raises(ConfigValidationError):
        validate_workflow_config({"plotting": {"save_dpi": 0}})


def test_apply_roi_policy_sets_expected_masks() -> None:
    reference = _make_image(np.ones((8, 8), dtype=np.float32))
    measured = _make_image(np.ones((8, 8), dtype=np.float32))
    reference_mask = sitk.Cast(reference > 0, sitk.sitkUInt8)
    measured_mask = sitk.Cast(measured > 0, sitk.sitkUInt8)
    evaluator = GammaMapEvaluator(
        reference_sar_linear=reference,
        measured_sar_linear=measured,
        reference_to_measured_transform=sitk.Euler2DTransform(),
    )

    _apply_roi_policy(
        evaluator,
        reference_mask_u8=reference_mask,
        measured_mask_u8=measured_mask,
        policy="reference_only",
    )
    assert evaluator.reference_mask_u8 is reference_mask
    assert evaluator.measured_mask_u8 is None

    _apply_roi_policy(
        evaluator,
        reference_mask_u8=reference_mask,
        measured_mask_u8=measured_mask,
        policy="intersection",
    )
    assert evaluator.reference_mask_u8 is reference_mask
    assert evaluator.measured_mask_u8 is measured_mask

    _apply_roi_policy(
        evaluator,
        reference_mask_u8=reference_mask,
        measured_mask_u8=measured_mask,
        policy="none",
    )
    assert evaluator.reference_mask_u8 is None
    assert evaluator.measured_mask_u8 is None


@pytest.mark.validation
def test_complete_workflow_integration_saves_overlay_outputs(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    measured_csv = project_root / "data/example/measured_sSAR1g.csv"
    reference_csv = project_root / "data/example/reference_sSAR1g.csv"

    loaded_images = tmp_path / "loaded.png"
    registered_overlay = tmp_path / "registered.png"
    gamma_image = tmp_path / "gamma.png"

    result = complete_workflow(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        loaded_images_save_path=str(loaded_images),
        registered_image_save_path=str(registered_overlay),
        gamma_comparison_image_path=str(gamma_image),
        transform_type="rigid",
        resample_resolution=0.001,
        stages=[
            {
                "translation_step": 0.010,
                "rot_step_deg": 4.0,
                "rot_span_deg": 180.0,
                "tx_steps": 6,
                "ty_steps": 6,
            }
        ],
        evaluation_roi_policy="intersection",
        save_failures_overlay=True,
        log_level="WARNING",
    )

    assert result.evaluated_pixel_count > 0
    assert 0 <= result.pass_rate_percent <= 100

    assert result.registered_overlay_path is not None
    assert result.gamma_image_path is not None
    assert result.failure_image_path is not None

    assert result.registered_overlay_path.exists()
    assert result.gamma_image_path.exists()
    assert result.failure_image_path.exists()


@pytest.mark.validation
def test_complete_workflow_passes_shared_plotting_config(tmp_path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    measured_csv = project_root / "data/example/measured_sSAR1g.csv"
    reference_csv = project_root / "data/example/reference_sSAR1g.csv"
    received: dict[str, object] = {}

    def capture_loader_plot(self, *args, **kwargs) -> None:
        received["loader"] = kwargs["plotting_config"]

    def capture_aligned_plot(self, *args, **kwargs) -> None:
        received["aligned"] = kwargs["plotting_config"]

    def capture_overlay(*args, **kwargs) -> None:
        received["overlay"] = kwargs["plotting_config"]

    def capture_gamma_show(self, *args, **kwargs) -> None:
        received["gamma"] = kwargs["plotting_config"]

    monkeypatch.setattr(SARImageLoader, "plot", capture_loader_plot)
    monkeypatch.setattr(SARImageLoader, "plot_aligned", capture_aligned_plot)
    monkeypatch.setattr(workflows_module, "show_registration_overlay", capture_overlay)
    monkeypatch.setattr(GammaMapEvaluator, "show", capture_gamma_show)

    result = complete_workflow(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        transform_type="rigid",
        resample_resolution=0.001,
        show_plot=True,
        plotting={
            "window_mm": (-50, 55, -45, 65),
            "font_size": 18,
            "save_dpi": 175,
        },
        stages=[
            {
                "translation_step": 0.010,
                "rot_step_deg": 4.0,
                "rot_span_deg": 180.0,
                "tx_steps": 6,
                "ty_steps": 6,
            }
        ],
        evaluation_roi_policy="intersection",
        save_failures_overlay=False,
        log_level="WARNING",
    )

    assert result.evaluated_pixel_count > 0
    assert set(received) == {"loader", "aligned", "overlay", "gamma"}
    for plotting_config in received.values():
        assert isinstance(plotting_config, PlottingConfig)
        assert plotting_config.window_mm == (-50.0, 55.0, -45.0, 65.0)
        assert plotting_config.font_size == 18
        assert plotting_config.save_dpi == 175


@pytest.mark.slow
def test_complete_workflow_recovers_high_pass_rate_for_shifted_synthetic_input(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    shifted_dir = tmp_path / "shifted"
    shifted_dir.mkdir()

    baseline_measured, baseline_reference = _write_synthetic_workflow_pair(baseline_dir)
    shifted_measured, shifted_reference = _write_synthetic_workflow_pair(
        shifted_dir, tx=0.004, ty=-0.003, theta_deg=0.0
    )

    common_kwargs = dict(
        transform_type="translate",
        resample_resolution=0.005,
        render_plots=False,
        show_plot=False,
        distance_to_agreement=2.0,
        dose_to_agreement=5.0,
        stages=[
            {
                "translation_step": 0.001,
                "rot_step_deg": 0.0,
                "rot_span_deg": 0.0,
                "tx_steps": 6,
                "ty_steps": 6,
            },
        ],
    )

    baseline = complete_workflow(
        measured_file_path=str(baseline_measured),
        reference_file_path=str(baseline_reference),
        **common_kwargs,
    )
    shifted = complete_workflow(
        measured_file_path=str(shifted_measured),
        reference_file_path=str(shifted_reference),
        **common_kwargs,
    )

    shifted_loader = SARImageLoader(
        measured_path=str(shifted_measured),
        reference_path=str(shifted_reference),
        resample_resolution=0.005,
        show_plot=False,
        warn=True,
    )
    _, reference_mask_u8 = shifted_loader.make_metric_masks()
    unregistered = GammaMapEvaluator(
        reference_sar_linear=shifted_loader.reference_image_linear,
        measured_sar_linear=shifted_loader.measured_image_linear,
        reference_to_measured_transform=sitk.TranslationTransform(2),
        dose_to_agreement_percent=5.0,
        distance_to_agreement_mm=2.0,
        gamma_cap=2.0,
    )
    unregistered.reference_mask_u8 = reference_mask_u8
    unregistered.compute()
    assert unregistered.pass_rate_percent is not None

    assert baseline.pass_rate_percent >= 98.0
    assert shifted.pass_rate_percent >= 85.0
    assert shifted.pass_rate_percent >= unregistered.pass_rate_percent + 20.0


@pytest.mark.slow
def test_complete_workflow_default_roi_matches_intersection(tmp_path: Path) -> None:
    measured_csv, reference_csv = _write_truncated_support_workflow_pair(tmp_path)

    common_kwargs = dict(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        transform_type="translate",
        resample_resolution=0.005,
        render_plots=False,
        show_plot=False,
        distance_to_agreement=2.0,
        dose_to_agreement=5.0,
        stages=[
            {
                "translation_step": 0.001,
                "rot_step_deg": 0.0,
                "rot_span_deg": 0.0,
                "tx_steps": 1,
                "ty_steps": 1,
            }
        ],
    )

    default_result = complete_workflow(**common_kwargs)
    intersection_result = complete_workflow(
        evaluation_roi_policy="intersection",
        **common_kwargs,
    )

    assert (
        default_result.evaluated_pixel_count
        == intersection_result.evaluated_pixel_count
    )
    assert default_result.passed_pixel_count == intersection_result.passed_pixel_count
    assert default_result.failed_pixel_count == intersection_result.failed_pixel_count
    assert default_result.pass_rate_percent == pytest.approx(
        intersection_result.pass_rate_percent,
        abs=1e-9,
    )


@pytest.mark.slow
def test_complete_workflow_roi_policies_change_evaluated_region_consistently(
    tmp_path: Path,
) -> None:
    measured_csv, reference_csv = _write_truncated_support_workflow_pair(tmp_path)

    common_kwargs = dict(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        transform_type="translate",
        resample_resolution=0.005,
        render_plots=False,
        show_plot=False,
        distance_to_agreement=2.0,
        dose_to_agreement=5.0,
        stages=[
            {
                "translation_step": 0.001,
                "rot_step_deg": 0.0,
                "rot_span_deg": 0.0,
                "tx_steps": 1,
                "ty_steps": 1,
            }
        ],
    )

    none_result = complete_workflow(
        evaluation_roi_policy="none",
        **common_kwargs,
    )
    reference_only_result = complete_workflow(
        evaluation_roi_policy="reference_only",
        **common_kwargs,
    )
    intersection_result = complete_workflow(
        evaluation_roi_policy="intersection",
        **common_kwargs,
    )

    assert (
        none_result.evaluated_pixel_count > reference_only_result.evaluated_pixel_count
    )
    assert (
        reference_only_result.evaluated_pixel_count
        >= intersection_result.evaluated_pixel_count
    )
    assert (
        intersection_result.pass_rate_percent >= reference_only_result.pass_rate_percent
    )


@pytest.mark.slow
def test_complete_workflow_emits_mask_too_small_issue(tmp_path: Path) -> None:
    """MASK_TOO_SMALL ValidationIssue is emitted when min_inscribed_square_mm exceeds mask size."""
    measured_csv, reference_csv = _write_synthetic_workflow_pair(tmp_path)

    result = complete_workflow(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        render_plots=False,
        show_plot=False,
        min_inscribed_square_mm=1000.0,  # 1 m — impossible to satisfy
    )

    assert not result.mask_fits_min_inscribed_square
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.code == "MASK_TOO_SMALL"
    assert issue.severity == "warning"
    assert "1000" in issue.message


def test_complete_workflow_v1_empty_measured_mask_raises_issue(tmp_path: Path) -> None:
    """V1: noise_floor > measured peak → EMPTY_MEASURED_MASK issue, not a raw ITK crash."""
    from sar_pattern_validation.errors import WorkflowExecutionError

    _, reference_csv = _write_synthetic_workflow_pair(tmp_path)
    # Write a measured CSV whose peak (0.001 W/kg) is below the default noise_floor (0.05)
    x, y = make_rect_grid(xmin=-0.04, xmax=0.04, ymin=-0.04, ymax=0.04, step=0.005)
    _, _, Z = gaussian_2d(x, y, x0=0.0, y0=0.0, sx=0.02, sy=0.02, peak=0.001)
    sub_floor_csv = tmp_path / "sub_floor_measured.csv"
    write_sar_csv(sub_floor_csv, x, y, Z)

    with pytest.raises(WorkflowExecutionError) as exc_info:
        complete_workflow(
            measured_file_path=str(sub_floor_csv),
            reference_file_path=str(reference_csv),
            render_plots=False,
            show_plot=False,
        )

    issue = exc_info.value.issue
    assert issue is not None
    assert issue.code == "EMPTY_MEASURED_MASK"
    assert issue.severity == "error"
    assert "noise floor" in issue.message.lower()


def test_complete_workflow_emits_csv_format_error_issue(tmp_path: Path) -> None:
    """CSV_FORMAT_ERROR issue is carried on WorkflowExecutionError for malformed input."""
    from sar_pattern_validation.errors import WorkflowExecutionError

    _, reference_csv = _write_synthetic_workflow_pair(tmp_path)
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("not,a,valid,sar,header\n1,2,3,4,5\n")

    with pytest.raises(WorkflowExecutionError) as exc_info:
        complete_workflow(
            measured_file_path=str(bad_csv),
            reference_file_path=str(reference_csv),
            render_plots=False,
            show_plot=False,
        )

    assert exc_info.value.issue is not None
    assert exc_info.value.issue.code == "CSV_FORMAT_ERROR"
    assert exc_info.value.issue.severity == "error"
