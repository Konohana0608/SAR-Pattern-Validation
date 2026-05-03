from __future__ import annotations

import json
import logging
import time

import pandas as pd
from pydantic import ValidationError

from sar_pattern_validation.sample_catalog import (
    DatabaseSampleColumn,
    DatabaseSampleFilterOption,
    DatabaseSampleFilters,
)

from .models import UiState, WorkflowResultPayload
from .runtime import WorkspacePaths

FILTER_ATTR_BY_COLUMN = {
    DatabaseSampleColumn.ANTENNA_TYPE.value: "antenna_type",
    DatabaseSampleColumn.FREQUENCY.value: "frequency",
    DatabaseSampleColumn.DISTANCE.value: "distance",
    DatabaseSampleColumn.MASS.value: "mass",
}


def save_ui_state(paths: WorkspacePaths, state: UiState) -> None:
    paths.system_state_dir.mkdir(parents=True, exist_ok=True)
    payload = state.model_copy(update={"updated_at_epoch_s": time.time()})
    temp_path = paths.ui_state_path.with_suffix(".tmp")
    temp_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    temp_path.replace(paths.ui_state_path)


def load_ui_state(paths: WorkspacePaths) -> UiState | None:
    if not paths.ui_state_path.exists():
        return None
    try:
        return UiState.model_validate_json(
            paths.ui_state_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        logging.getLogger(__name__).warning(
            "Saved UI state at %s is not valid JSON — starting fresh.",
            paths.ui_state_path,
        )
        logging.getLogger(__name__).debug("UI state JSON decode error", exc_info=error)
        return None
    except ValidationError as error:
        logging.getLogger(__name__).warning(
            "Saved UI state at %s failed validation — starting fresh.",
            paths.ui_state_path,
        )
        logging.getLogger(__name__).debug("UI state validation error", exc_info=error)
        return None


def load_or_migrate_ui_state(paths: WorkspacePaths) -> UiState | None:
    current = load_ui_state(paths)
    if current is not None:
        return current
    migrated = migrate_legacy_state(paths)
    if migrated is not None:
        save_ui_state(paths, migrated)
    return migrated


def migrate_legacy_state(paths: WorkspacePaths) -> UiState | None:
    if not paths.legacy_state_path.exists():
        return None

    state_data = json.loads(paths.legacy_state_path.read_text(encoding="utf-8"))
    measured_file_name = str(state_data.get("measured_file_name", ""))
    power_level = float(state_data.get("power_level", 23.0))

    filters = DatabaseSampleFilters()
    if paths.legacy_filtered_db_csv_path.exists():
        filtered_df = pd.read_csv(paths.legacy_filtered_db_csv_path)
        for column_name, attr_name in FILTER_ATTR_BY_COLUMN.items():
            if column_name not in filtered_df.columns:
                continue
            unique_values = filtered_df[column_name].dropna().unique()
            if len(unique_values) == 1:
                value = unique_values[0]
                if hasattr(value, "item"):
                    value = value.item()
                setattr(
                    filters,
                    attr_name,
                    DatabaseSampleFilterOption(
                        column_name=DatabaseSampleColumn(column_name),
                        value=value,
                    ),
                )

    last_result = None
    if paths.legacy_workflow_results_path.exists():
        last_result = WorkflowResultPayload.model_validate(
            json.loads(paths.legacy_workflow_results_path.read_text(encoding="utf-8"))
        )

    return UiState(
        measured_file_name=measured_file_name,
        power_level=power_level,
        active_filters=filters,
        last_result=last_result,
        updated_at_epoch_s=state_data.get("timestamp"),
    )
