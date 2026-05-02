from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sar_pattern_validation.errors import ConfigValidationError
from sar_pattern_validation.workflow_config import WorkflowConfig


def validate_workflow_config(raw: dict[str, Any] | WorkflowConfig) -> WorkflowConfig:
    if isinstance(raw, WorkflowConfig):
        return raw

    try:
        return WorkflowConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc


def ensure_input_files_exist(config: WorkflowConfig) -> None:
    missing = [
        p
        for p in (config.measured_file_path, config.reference_file_path)
        if not Path(p).is_file()
    ]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise ConfigValidationError(f"Input file(s) not found: {missing_list}")
