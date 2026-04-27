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
from sar_pattern_validation.registration2d import Rigid2DRegistration, Transform2D
from sar_pattern_validation.workflow_config import PlottingConfig
from sar_pattern_validation.workflow_schema import validate_workflow_config
from sar_pattern_validation.workflows import (
    _apply_roi_policy,
    _select_registration_mask,
    complete_workflow,
)

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


@pytest.mark.parametrize(
    "x_mm,y_mm",
    [
        (22.0, 100.0),  # x at exclusive lower bound
        (22.0001, 22.0001),  # below lower bound on y
        (601.0, 200.0),  # x above upper bound
        (300.0, 401.0),  # y above upper bound
        (None, 100.0),  # only one of the pair set
        (100.0, None),
    ],
)
def test_validate_workflow_config_rejects_out_of_range_measurement_area(
    x_mm: float | None, y_mm: float | None
) -> None:
    payload: dict[str, float | None] = {}
    if x_mm is not None:
        payload["measurement_area_x_mm"] = x_mm
    if y_mm is not None:
        payload["measurement_area_y_mm"] = y_mm
    # 22.0001/22.0001 is inside individual bounds; ensure the only failure cases
    # in this parametrization are the out-of-range / unpaired ones.
    if x_mm == 22.0001 and y_mm == 22.0001:
        # both individually valid; should NOT raise
        config = validate_workflow_config(payload)
        assert config.measurement_area_x_mm == pytest.approx(22.0001)
        assert config.measurement_area_y_mm == pytest.approx(22.0001)
        return
    with pytest.raises(ConfigValidationError):
        validate_workflow_config(payload)


def test_validate_workflow_config_measurement_area_derives_square_window() -> None:
    config = validate_workflow_config(
        {"measurement_area_x_mm": 300.0, "measurement_area_y_mm": 200.0}
    )
    assert config.measurement_area_x_mm == 300.0
    assert config.measurement_area_y_mm == 200.0
    # window is centered, square, side = max(x, y) = 300
    assert config.plotting.window_mm == (-150.0, 150.0, -150.0, 150.0)


def test_validate_workflow_config_measurement_area_square_uses_y_when_larger() -> None:
    config = validate_workflow_config(
        {"measurement_area_x_mm": 100.0, "measurement_area_y_mm": 400.0}
    )
    assert config.plotting.window_mm == (-200.0, 200.0, -200.0, 200.0)


def test_validate_workflow_config_no_measurement_area_keeps_default_window() -> None:
    config = validate_workflow_config({})
    assert config.measurement_area_x_mm is None
    assert config.measurement_area_y_mm is None
    assert config.plotting.window_mm == (-120.0, 120.0, -120.0, 120.0)


def test_select_registration_mask_falls_back_to_support_for_sparse_measurement() -> (
    None
):
    project_root = Path(__file__).resolve().parents[1]
    measured_csv = (
        project_root / "data/measurements/D900_Flat HSL_15 mm_10 dBm_1g_10.csv"
    )
    reference_csv = project_root / "data/database/dipole_900MHz_Flat_15mm_1g.csv"

    loader = SARImageLoader(
        measured_path=str(measured_csv),
        reference_path=str(reference_csv),
        power_level_dbm=10.0,
        noise_floor_wkg=0.05,
        show_plot=False,
        warn=True,
    )

    measured_metric_mask_u8, reference_metric_mask_u8 = loader.make_metric_masks()
    measured_support_u8, reference_support_u8 = loader.make_support_masks()

    registration_measured_mask_u8, measured_source = _select_registration_mask(
        metric_mask_u8=measured_metric_mask_u8,
        support_mask_u8=measured_support_u8,
        metric_mask_sufficient=loader.measured_metric_mask_sufficient,
    )
    registration_reference_mask_u8, reference_source = _select_registration_mask(
        metric_mask_u8=reference_metric_mask_u8,
        support_mask_u8=reference_support_u8,
        metric_mask_sufficient=loader.reference_metric_mask_sufficient,
    )

    assert measured_source == "support"
    assert reference_source == "metric"
    assert registration_measured_mask_u8 is measured_support_u8
    assert registration_reference_mask_u8 is reference_metric_mask_u8

    reg = Rigid2DRegistration(
        fixed_image=loader.reference_image_db,
        moving_image=loader.measured_image_db,
        transform_type=Transform2D.RIGID,
    )
    stages = reg.build_adaptive_stages(
        fixed_image=loader.reference_image_db,
        moving_image=loader.measured_image_db,
        transform_type=Transform2D.RIGID,
        fixed_mask=registration_reference_mask_u8,
        moving_mask=registration_measured_mask_u8,
    )

    assert stages[0]["tx_steps"] > 1
    assert stages[0]["ty_steps"] > 2


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


@pytest.mark.integration
@pytest.mark.slow
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


@pytest.mark.integration
@pytest.mark.slow
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
