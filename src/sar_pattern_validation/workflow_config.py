from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sar_pattern_validation.registration2d import Transform2D

DEFAULT_MEASURED_FILE_PATH: Final[str] = "measured.csv"
DEFAULT_REFERENCE_FILE_PATH: Final[str] = "reference.csv"
DEFAULT_POWER_LEVEL_DBM: Final[float] = 30.0
DEFAULT_NOISE_FLOOR: Final[float] = 0.05
NOISE_FLOOR_MAX_WKG: Final[float] = 0.1
DEFAULT_SHOW_PLOT: Final[bool] = False
DEFAULT_RENDER_PLOTS: Final[bool] = True
DEFAULT_DOSE_TO_AGREEMENT: Final[float] = 5.0
DEFAULT_DISTANCE_TO_AGREEMENT: Final[float] = 2.0
DEFAULT_GAMMA_CAP: Final[float] = 2.0
DEFAULT_EVALUATION_ROI_POLICY: Final[
    Literal["reference_only", "intersection", "none"]
] = "intersection"
DEFAULT_REGISTRATION_STAGE_POLICY: Final[Literal["static", "adaptive"]] = "adaptive"
DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY: Final[bool] = True
DEFAULT_ADAPTIVE_MAX_STAGES: Final[int] = 5
DEFAULT_ADAPTIVE_MAX_STAGE_EVALS: Final[int] = 50000
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_PLOT_WINDOW_MM: Final[tuple[float, float, float, float]] = (
    -120.0,
    120.0,
    -120.0,
    120.0,
)

# Measurement-area bounds (per MGD 2026-04-24 feedback). Inclusive upper bound,
# exclusive lower bound — a 22 mm × 22 mm 10 g cube face must fit strictly
# inside the area, so the area itself must exceed 22 mm on each axis.
MEASUREMENT_AREA_MIN_MM_EXCLUSIVE: Final[float] = 22.0
MEASUREMENT_AREA_MAX_X_MM: Final[float] = 600.0
MEASUREMENT_AREA_MAX_Y_MM: Final[float] = 400.0
DEFAULT_PLOT_FONT_SIZE: Final[float] = 14.0
DEFAULT_SINGLE_FIGURE_SIZE: Final[tuple[float, float]] = (6.0, 6.0)
DEFAULT_COMBINED_FIGURE_SIZE: Final[tuple[float, float]] = (12.0, 5.0)
DEFAULT_PLOT_FIGURE_FACECOLOR: Final[str] = "white"
DEFAULT_PLOT_DARK_AXES_FACECOLOR: Final[str] = "black"
DEFAULT_PLOT_LIGHT_AXES_FACECOLOR: Final[str] = "white"
DEFAULT_PLOT_SAVE_DPI: Final[int] = 200

ROI_POLICY_CHOICES: Final[tuple[str, ...]] = ("reference_only", "intersection", "none")
EvaluationRoiPolicy = Literal["reference_only", "intersection", "none"]
LOG_LEVEL_CHOICES: Final[tuple[str, ...]] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)


def default_registration_stages() -> list[dict[str, Any]]:
    return [
        dict(
            translation_step=0.010,
            rot_step_deg=4.0,
            rot_span_deg=180.0,
            tx_steps=6,
            ty_steps=6,
        ),
        dict(
            translation_step=0.001,
            rot_step_deg=1.0,
            rot_span_deg=90.0,
            tx_steps=6,
            ty_steps=6,
        ),
    ]


class RegistrationStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translation_step: float = Field(gt=0)
    rot_step_deg: float = Field(ge=0)
    rot_span_deg: float = Field(ge=0)
    tx_steps: int = Field(ge=0)
    ty_steps: int = Field(ge=0)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_mm: tuple[float, float, float, float] = DEFAULT_PLOT_WINDOW_MM
    font_size: float = Field(default=DEFAULT_PLOT_FONT_SIZE, gt=0)
    single_figure_size: tuple[float, float] = DEFAULT_SINGLE_FIGURE_SIZE
    combined_figure_size: tuple[float, float] = DEFAULT_COMBINED_FIGURE_SIZE
    figure_facecolor: str = DEFAULT_PLOT_FIGURE_FACECOLOR
    dark_axes_facecolor: str = DEFAULT_PLOT_DARK_AXES_FACECOLOR
    light_axes_facecolor: str = DEFAULT_PLOT_LIGHT_AXES_FACECOLOR
    save_dpi: int = Field(default=DEFAULT_PLOT_SAVE_DPI, gt=0)

    @field_validator("single_figure_size", "combined_figure_size")
    @classmethod
    def _validate_figure_size(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("figure sizes must be positive")
        return value


def _centered_square_window_mm(side_mm: float) -> tuple[float, float, float, float]:
    half_side_mm = float(side_mm) / 2.0
    return (-half_side_mm, half_side_mm, -half_side_mm, half_side_mm)


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measured_file_path: str = DEFAULT_MEASURED_FILE_PATH
    reference_file_path: str = DEFAULT_REFERENCE_FILE_PATH
    power_level_dbm: float = DEFAULT_POWER_LEVEL_DBM

    loaded_images_save_path: str | None = None
    reference_image_save_path: str | None = None
    measured_image_save_path: str | None = None
    aligned_meas_save_path: str | None = None
    registered_image_save_path: str | None = None
    gamma_comparison_image_path: str | None = None

    resample_resolution: float | None = Field(default=None, gt=0)
    noise_floor: float = Field(default=DEFAULT_NOISE_FLOOR, gt=0)
    show_plot: bool = DEFAULT_SHOW_PLOT
    render_plots: bool = DEFAULT_RENDER_PLOTS

    transform_type: Transform2D = Transform2D.RIGID
    stages: list[RegistrationStage] = Field(
        default_factory=lambda: [
            RegistrationStage.model_validate(stage)
            for stage in default_registration_stages()
        ]
    )
    registration_stage_policy: Literal["static", "adaptive"] = (
        DEFAULT_REGISTRATION_STAGE_POLICY
    )
    adaptive_assume_axial_symmetry: bool = DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY
    adaptive_max_stages: int = Field(default=DEFAULT_ADAPTIVE_MAX_STAGES, ge=1, le=8)
    adaptive_max_stage_evals: int = Field(
        default=DEFAULT_ADAPTIVE_MAX_STAGE_EVALS, ge=100, le=1_000_000
    )

    dose_to_agreement: float = Field(default=DEFAULT_DOSE_TO_AGREEMENT, gt=0)
    distance_to_agreement: float = Field(default=DEFAULT_DISTANCE_TO_AGREEMENT, gt=0)
    gamma_cap: float = Field(default=DEFAULT_GAMMA_CAP, gt=0)
    evaluation_roi_policy: EvaluationRoiPolicy = DEFAULT_EVALUATION_ROI_POLICY
    save_failures_overlay: bool = True
    log_level: str = DEFAULT_LOG_LEVEL
    plotting: PlottingConfig = Field(default_factory=PlottingConfig)
    measurement_area_x_mm: float | None = Field(
        default=None,
        gt=MEASUREMENT_AREA_MIN_MM_EXCLUSIVE,
        le=MEASUREMENT_AREA_MAX_X_MM,
    )
    measurement_area_y_mm: float | None = Field(
        default=None,
        gt=MEASUREMENT_AREA_MIN_MM_EXCLUSIVE,
        le=MEASUREMENT_AREA_MAX_Y_MM,
    )
    noise_floor_wkg: float | None = Field(default=None, ge=0, le=NOISE_FLOOR_MAX_WKG)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = str(value).upper()
        if normalized not in LOG_LEVEL_CHOICES:
            allowed = ", ".join(LOG_LEVEL_CHOICES)
            raise ValueError(f"log_level must be one of: {allowed}")
        return normalized

    @model_validator(mode="after")
    def _validate_measurement_area(self) -> WorkflowConfig:
        x_mm = self.measurement_area_x_mm
        y_mm = self.measurement_area_y_mm

        if (x_mm is None) != (y_mm is None):
            raise ValueError(
                "measurement_area_x_mm and measurement_area_y_mm must be provided together"
            )

        if x_mm is None or y_mm is None:
            return self

        self.plotting = self.plotting.model_copy(
            update={"window_mm": _centered_square_window_mm(max(x_mm, y_mm))}
        )
        return self


class WorkflowResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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
    measured_pssar_at_input_power: float | None = None
    reference_pssar: float
    scaling_error: float
    dose_to_agreement: float
    distance_to_agreement: float
    input_power_level_dbm: float | None = None
    gamma_map: np.ndarray | None = Field(default=None, exclude=True, repr=False)
    evaluation_mask: np.ndarray | None = Field(default=None, exclude=True, repr=False)

    @field_validator(
        "gamma_image_path",
        "failure_image_path",
        "registered_overlay_path",
        "loaded_images_path",
        "reference_image_path",
        "measured_image_path",
        "aligned_measured_path",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return str(value)

    @field_validator("pass_rate_percent")
    @classmethod
    def _validate_pass_rate(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("pass_rate_percent must be between 0 and 100")
        return value

    @field_validator(
        "evaluated_pixel_count", "passed_pixel_count", "failed_pixel_count"
    )
    @classmethod
    def _validate_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("pixel counts must be non-negative")
        return value

    @field_validator("scaling_error")
    @classmethod
    def _validate_scaling_error(cls, value: float) -> float:
        if not isinstance(value, float):
            raise ValueError("scaling_error must be a float")
        return value

    def __str__(self) -> str:
        items: list[str] = []
        max_len = max(len(key) for key in self.__class__.model_fields)
        for key, value in self:
            value_str = str(value) if isinstance(value, Path) else repr(value)
            items.append(f"{key.ljust(max_len)} : {value_str}")
        return "\n".join(items)

    def save_to_json(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text(
            self.model_dump_json(indent=2, exclude={"gamma_map", "evaluation_mask"}),
            encoding="utf-8",
        )

    @classmethod
    def load_from_json(cls, path: str | Path) -> WorkflowResult:
        import json

        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
