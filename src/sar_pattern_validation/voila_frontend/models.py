from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sar_pattern_validation.sample_catalog import DatabaseSampleFilters
from sar_pattern_validation.workflow_config import DEFAULT_NOISE_FLOOR, WorkflowResult


class WorkflowResultPayload(WorkflowResult):
    """Frontend-facing workflow payload."""

    reference_file_path: str | None = None
    measured_file_sha256: str | None = None
    input_noise_floor_wkg: float | None = None


class UiState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measured_file_name: str = ""
    power_level: float = 23.0
    noise_floor_wkg: float = DEFAULT_NOISE_FLOOR
    active_filters: DatabaseSampleFilters = Field(default_factory=DatabaseSampleFilters)
    last_result: WorkflowResultPayload | None = None
    updated_at_epoch_s: float | None = None
