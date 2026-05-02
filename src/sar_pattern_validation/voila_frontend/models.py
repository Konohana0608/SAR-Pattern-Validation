from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sar_pattern_validation.sample_catalog import DatabaseSampleFilters
from sar_pattern_validation.workflow_config import WorkflowResult


class WorkflowResultPayload(WorkflowResult):
    """Frontend-facing workflow payload."""


class UiState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measured_file_name: str = ""
    power_level: float = 23.0
    active_filters: DatabaseSampleFilters = Field(default_factory=DatabaseSampleFilters)
    last_result: WorkflowResultPayload | None = None
    updated_at_epoch_s: float | None = None
