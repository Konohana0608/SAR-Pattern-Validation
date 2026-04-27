from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from sar_pattern_validation.errors import WorkflowExecutionError
from sar_pattern_validation.gamma_eval import GammaMapEvaluator
from sar_pattern_validation.image_loader import SARImageLoader
from sar_pattern_validation.plotting import show_registration_overlay
from sar_pattern_validation.registration2d import Rigid2DRegistration, Transform2D
from sar_pattern_validation.utils import configure_root_logging, ensure_output_path
from sar_pattern_validation.workflow_config import (
    DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY,
    DEFAULT_ADAPTIVE_MAX_STAGE_EVALS,
    DEFAULT_ADAPTIVE_MAX_STAGES,
    DEFAULT_COMBINED_FIGURE_SIZE,
    DEFAULT_DISTANCE_TO_AGREEMENT,
    DEFAULT_DOSE_TO_AGREEMENT,
    DEFAULT_GAMMA_CAP,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MEASURED_FILE_PATH,
    DEFAULT_NOISE_FLOOR,
    DEFAULT_PLOT_DARK_AXES_FACECOLOR,
    DEFAULT_PLOT_FIGURE_FACECOLOR,
    DEFAULT_PLOT_FONT_SIZE,
    DEFAULT_PLOT_LIGHT_AXES_FACECOLOR,
    DEFAULT_PLOT_SAVE_DPI,
    DEFAULT_PLOT_WINDOW_MM,
    DEFAULT_POWER_LEVEL_DBM,
    DEFAULT_REFERENCE_FILE_PATH,
    DEFAULT_REGISTRATION_STAGE_POLICY,
    DEFAULT_RENDER_PLOTS,
    DEFAULT_SINGLE_FIGURE_SIZE,
    ROI_POLICY_CHOICES,
    WorkflowConfig,
)
from sar_pattern_validation.workflow_schema import (
    ensure_input_files_exist,
    validate_workflow_config,
)

LOGGER = logging.getLogger(__name__)


class WorkflowResultCLIExcludedFields(Enum):
    """
    Fields to exclude from CLI JSON serialization.

    Note: Update these values if WorkflowResult field names change.
    """

    GAMMA_MAP = "gamma_map"
    EVALUATION_MASK = "evaluation_mask"


@dataclass
class WorkflowResult:
    pass_rate_percent: float
    evaluated_pixel_count: int
    passed_pixel_count: int
    failed_pixel_count: int
    gamma_image_path: Path | None
    failure_image_path: Path | None
    registered_overlay_path: Path | None
    loaded_images_path: Path | None
    reference_image_path: Path | None
    measured_image_path: Path | None
    aligned_measured_path: Path | None
    measured_pssar: float
    reference_pssar: float
    scaling_error: float
    dose_to_agreement: float
    distance_to_agreement: float
    gamma_map: np.ndarray | None = field(default=None, compare=False)
    evaluation_mask: np.ndarray | None = field(default=None, compare=False)


def _failure_overlay_path(
    gamma_comparison_image_path: Path | None, save_failures_overlay: bool
) -> Path | None:
    if not save_failures_overlay:
        return None
    if gamma_comparison_image_path is None:
        LOGGER.warning(
            "save_failures_overlay is enabled but gamma_comparison_image_path is not set. "
            "Failure overlay will not be saved."
        )
        return None
    return gamma_comparison_image_path.with_name(
        f"{gamma_comparison_image_path.stem}_failures{gamma_comparison_image_path.suffix}"
    )


def _apply_roi_policy(
    evaluator: GammaMapEvaluator,
    *,
    reference_mask_u8: sitk.Image,
    measured_mask_u8: sitk.Image,
    policy: str,
) -> None:
    if policy == "reference_only":
        evaluator.reference_mask_u8 = reference_mask_u8
        evaluator.measured_mask_u8 = None
        return
    if policy == "intersection":
        evaluator.reference_mask_u8 = reference_mask_u8
        evaluator.measured_mask_u8 = measured_mask_u8
        return
    if policy == "none":
        evaluator.reference_mask_u8 = None
        evaluator.measured_mask_u8 = None
        return
    raise ValueError(f"Unsupported evaluation_roi_policy: {policy}")


def _mask_has_active_pixels(mask: sitk.Image | None) -> bool:
    if mask is None:
        return False
    return bool(np.any(sitk.GetArrayFromImage(mask) > 0))


def _select_registration_mask(
    *,
    metric_mask_u8: sitk.Image,
    support_mask_u8: sitk.Image,
    metric_mask_sufficient: bool,
) -> tuple[sitk.Image | None, str]:
    if metric_mask_sufficient and _mask_has_active_pixels(metric_mask_u8):
        return metric_mask_u8, "metric"
    if _mask_has_active_pixels(support_mask_u8):
        return support_mask_u8, "support"
    return None, "none"


def _complete_workflow(config: WorkflowConfig) -> WorkflowResult:
    try:
        ensure_input_files_exist(config)

        loaded_images_save_path = (
            ensure_output_path(config.loaded_images_save_path)
            if config.render_plots
            else None
        )
        reference_image_save_path = (
            ensure_output_path(config.reference_image_save_path)
            if config.render_plots
            else None
        )
        measured_image_save_path = (
            ensure_output_path(config.measured_image_save_path)
            if config.render_plots
            else None
        )
        aligned_meas_save_path = (
            ensure_output_path(config.aligned_meas_save_path)
            if config.render_plots
            else None
        )
        registered_image_save_path = (
            ensure_output_path(config.registered_image_save_path)
            if config.render_plots
            else None
        )
        gamma_comparison_image_path = (
            ensure_output_path(config.gamma_comparison_image_path)
            if config.render_plots
            else None
        )
        failure_image_path = _failure_overlay_path(
            gamma_comparison_image_path, config.save_failures_overlay
        )

        LOGGER.info("Step 1/3: Loading and preprocessing SAR images")
        loader = SARImageLoader(
            measured_path=config.measured_file_path,
            reference_path=config.reference_file_path,
            power_level_dbm=config.power_level_dbm,
            resample_resolution=config.resample_resolution,
            noise_floor_wkg=config.noise_floor,
            show_plot=False,
            warn=True,
        )

        reference_db, measured_db = loader.get_images()

        if config.render_plots and (
            config.show_plot
            or loaded_images_save_path is not None
            or reference_image_save_path is not None
            or measured_image_save_path is not None
        ):
            loader.plot(
                image_save_path=loaded_images_save_path,
                reference_save_path=reference_image_save_path,
                measured_save_path=measured_image_save_path,
                plotting_config=config.plotting,
            )

        LOGGER.info("Step 2/3: Registering reference SAR onto measured grid")
        measured_mask_u8, reference_mask_u8 = loader.make_metric_masks()
        measured_support_u8, reference_support_u8 = loader.make_support_masks()
        registration_measured_mask_u8, measured_mask_source = _select_registration_mask(
            metric_mask_u8=measured_mask_u8,
            support_mask_u8=measured_support_u8,
            metric_mask_sufficient=loader.measured_metric_mask_sufficient,
        )
        registration_reference_mask_u8, reference_mask_source = (
            _select_registration_mask(
                metric_mask_u8=reference_mask_u8,
                support_mask_u8=reference_support_u8,
                metric_mask_sufficient=loader.reference_metric_mask_sufficient,
            )
        )
        LOGGER.info(
            "Registration mask sources: measured=%s, reference=%s",
            measured_mask_source,
            reference_mask_source,
        )

        reg = Rigid2DRegistration(
            fixed_image=measured_db,
            moving_image=reference_db,
            transform_type=config.transform_type,
        )

        registration_stages = config.stages
        if config.registration_stage_policy == "adaptive":
            registration_stages = reg.build_adaptive_stages(
                fixed_image=measured_db,
                moving_image=reference_db,
                transform_type=config.transform_type,
                fixed_mask=registration_measured_mask_u8,
                moving_mask=registration_reference_mask_u8,
                assume_axial_symmetry=config.adaptive_assume_axial_symmetry,
                max_stages=config.adaptive_max_stages,
                max_stage_evals=config.adaptive_max_stage_evals,
            )
            LOGGER.info("Adaptive registration stages: %s", registration_stages)

        aligned_db, final_tx = reg.run(
            stages=registration_stages,
            fixed_mask=registration_measured_mask_u8,
            moving_mask=registration_reference_mask_u8,
        )

        if config.render_plots and (
            config.show_plot or aligned_meas_save_path is not None
        ):
            aligned_lin = reg.db_to_linear(aligned_db, floor_norm=-120.0)
            loader.plot_aligned(
                aligned_lin,
                aligned_meas_save_path,
                plotting_config=config.plotting,
            )

        if config.render_plots:
            show_registration_overlay(
                measured_db,
                aligned_db,
                title="Rigid Registration Overlay",
                image_save_path=registered_image_save_path,
                plotting_config=config.plotting,
            )

        LOGGER.info(
            "Step 3/3: Computing gamma map (policy=%s, dta=%.3f mm, dose=%.3f%%)",
            config.evaluation_roi_policy,
            config.distance_to_agreement,
            config.dose_to_agreement,
        )
        evaluator = GammaMapEvaluator(
            reference_sar_linear=loader.reference_image_linear,
            measured_sar_linear=loader.measured_image_linear,
            reference_to_measured_transform=final_tx,
            dose_to_agreement_percent=float(config.dose_to_agreement),
            distance_to_agreement_mm=float(config.distance_to_agreement),
            gamma_cap=float(config.gamma_cap),
        )
        _apply_roi_policy(
            evaluator,
            reference_mask_u8=reference_mask_u8,
            measured_mask_u8=measured_support_u8,
            policy=config.evaluation_roi_policy,
        )

        evaluator.compute()
        gamma_map = evaluator.gamma_map
        evaluation_mask = evaluator.evaluation_mask
        if config.render_plots:
            evaluator.show(
                gamma_image_save_path=gamma_comparison_image_path,
                failure_image_save_path=failure_image_path,
                plotting_config=config.plotting,
            )

        if (
            evaluator.pass_rate_percent is None
            or evaluator.evaluated_pixel_count is None
            or evaluator.passed_pixel_count is None
            or evaluator.failed_pixel_count is None
        ):
            raise RuntimeError("Gamma evaluation completed without summary metrics.")

        LOGGER.info(
            "Gamma completed: pass_rate=%.2f%%, evaluated=%d, passed=%d, failed=%d",
            evaluator.pass_rate_percent,
            evaluator.evaluated_pixel_count,
            evaluator.passed_pixel_count,
            evaluator.failed_pixel_count,
        )
        return WorkflowResult(
            pass_rate_percent=evaluator.pass_rate_percent,
            evaluated_pixel_count=evaluator.evaluated_pixel_count,
            passed_pixel_count=evaluator.passed_pixel_count,
            failed_pixel_count=evaluator.failed_pixel_count,
            gamma_image_path=gamma_comparison_image_path,
            failure_image_path=failure_image_path,
            registered_overlay_path=registered_image_save_path,
            loaded_images_path=loaded_images_save_path,
            reference_image_path=reference_image_save_path,
            measured_image_path=measured_image_save_path,
            aligned_measured_path=aligned_meas_save_path,
            measured_pssar=loader.measured_peak_30dbm,
            reference_pssar=loader.reference_peak,
            scaling_error=loader.scaling_error,
            dose_to_agreement=config.dose_to_agreement,
            distance_to_agreement=config.distance_to_agreement,
            gamma_map=gamma_map,
            evaluation_mask=evaluation_mask,
        )
    except Exception as exc:
        raise WorkflowExecutionError(f"Workflow failed: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    defaults = WorkflowConfig()
    parser = argparse.ArgumentParser(description="Run a SAR gamma comparison workflow.")
    parser.add_argument(
        "--measured_file_path",
        type=str,
        default=DEFAULT_MEASURED_FILE_PATH,
    )
    parser.add_argument(
        "--reference_file_path",
        dest="reference_file_path",
        type=str,
        default=DEFAULT_REFERENCE_FILE_PATH,
    )
    parser.add_argument(
        "--power_level_dbm",
        type=float,
        default=DEFAULT_POWER_LEVEL_DBM,
    )
    parser.add_argument("--loaded_images_save_path", type=str, default=None)
    parser.add_argument("--reference_image_save_path", type=str, default=None)
    parser.add_argument("--measured_image_save_path", type=str, default=None)
    parser.add_argument("--aligned_meas_save_path", type=str, default=None)
    parser.add_argument("--registered_image_save_path", type=str, default=None)
    parser.add_argument("--gamma_comparison_image_path", type=str, default=None)

    parser.add_argument("--resample_resolution", type=float, default=None)
    parser.add_argument("--noise_floor", type=float, default=DEFAULT_NOISE_FLOOR)
    parser.add_argument(
        "--show_plot",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_plot,
    )
    parser.add_argument(
        "--render_plots",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RENDER_PLOTS,
    )

    parser.add_argument(
        "--transform_type",
        type=Transform2D,
        choices=list(Transform2D),
        default=defaults.transform_type,
    )
    parser.add_argument(
        "--registration_stage_policy",
        type=str,
        choices=["static", "adaptive"],
        default=DEFAULT_REGISTRATION_STAGE_POLICY,
    )
    parser.add_argument(
        "--adaptive_assume_axial_symmetry",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY,
    )
    parser.add_argument(
        "--adaptive_max_stages",
        type=int,
        default=DEFAULT_ADAPTIVE_MAX_STAGES,
    )
    parser.add_argument(
        "--adaptive_max_stage_evals",
        type=int,
        default=DEFAULT_ADAPTIVE_MAX_STAGE_EVALS,
    )

    parser.add_argument(
        "--dose_to_agreement",
        type=float,
        default=DEFAULT_DOSE_TO_AGREEMENT,
    )
    parser.add_argument(
        "--distance_to_agreement",
        type=float,
        default=DEFAULT_DISTANCE_TO_AGREEMENT,
    )
    parser.add_argument("--gamma_cap", type=float, default=DEFAULT_GAMMA_CAP)
    parser.add_argument(
        "--evaluation_roi_policy",
        type=str,
        default=defaults.evaluation_roi_policy,
        choices=list(ROI_POLICY_CHOICES),
    )
    parser.add_argument(
        "--save_failures_overlay",
        action=argparse.BooleanOptionalAction,
        default=defaults.save_failures_overlay,
    )
    parser.add_argument(
        "--plot-window-mm",
        nargs=4,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    parser.add_argument("--plot-font-size", type=float, default=None)
    parser.add_argument(
        "--plot-single-figure-size",
        nargs=2,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--plot-combined-figure-size",
        nargs=2,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--plot-figure-facecolor", type=str, default=None)
    parser.add_argument("--plot-dark-axes-facecolor", type=str, default=None)
    parser.add_argument("--plot-light-axes-facecolor", type=str, default=None)
    parser.add_argument("--plot-save-dpi", type=int, default=None)
    parser.add_argument("--log_level", type=str, default=DEFAULT_LOG_LEVEL)
    return parser


def _normalize_plotting_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw_config)

    plotting = normalized.get("plotting")
    if plotting is None:
        plotting = {}
    elif is_dataclass(plotting):
        plotting = asdict(plotting)  # type: ignore
    elif hasattr(plotting, "items"):
        plotting = dict(plotting.items())
    else:
        plotting = dict(plotting)

    cli_plotting = {
        "window_mm": normalized.pop("plot_window_mm", None),
        "font_size": normalized.pop("plot_font_size", None),
        "single_figure_size": normalized.pop("plot_single_figure_size", None),
        "combined_figure_size": normalized.pop("plot_combined_figure_size", None),
        "figure_facecolor": normalized.pop("plot_figure_facecolor", None),
        "dark_axes_facecolor": normalized.pop("plot_dark_axes_facecolor", None),
        "light_axes_facecolor": normalized.pop("plot_light_axes_facecolor", None),
        "save_dpi": normalized.pop("plot_save_dpi", None),
    }

    defaults = {
        "window_mm": DEFAULT_PLOT_WINDOW_MM,
        "font_size": DEFAULT_PLOT_FONT_SIZE,
        "single_figure_size": DEFAULT_SINGLE_FIGURE_SIZE,
        "combined_figure_size": DEFAULT_COMBINED_FIGURE_SIZE,
        "figure_facecolor": DEFAULT_PLOT_FIGURE_FACECOLOR,
        "dark_axes_facecolor": DEFAULT_PLOT_DARK_AXES_FACECOLOR,
        "light_axes_facecolor": DEFAULT_PLOT_LIGHT_AXES_FACECOLOR,
        "save_dpi": DEFAULT_PLOT_SAVE_DPI,
    }
    for key, value in cli_plotting.items():
        if value is not None:
            plotting[key] = tuple(value) if isinstance(value, list) else value

    normalized["plotting"] = {**defaults, **plotting}
    return normalized


def complete_workflow(*args, **kwargs) -> WorkflowResult:
    parser = _build_parser()

    raw_config: dict[str, Any]
    if kwargs:
        raw_config = kwargs
    else:
        parsed = parser.parse_args([str(a) for a in args] if args else None)
        raw_config = vars(parsed)

    raw_config = _normalize_plotting_config(raw_config)

    config = validate_workflow_config(raw_config)

    configure_root_logging(config.log_level)

    LOGGER.info("WORKFLOW START with config: %s", config)
    return _complete_workflow(config)


if __name__ == "__main__":
    complete_workflow()
