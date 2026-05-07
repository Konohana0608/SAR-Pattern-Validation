from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .catalog import (
    DatabaseSampleColumn,
    DatabaseSampleFilterOption,
    DatabaseSampleFilters,
)
from .runtime import WorkspacePaths

FILTER_ATTR_BY_COLUMN = {
    DatabaseSampleColumn.ANTENNA_TYPE.value: "antenna_type",
    DatabaseSampleColumn.FREQUENCY.value: "frequency",
    DatabaseSampleColumn.DISTANCE.value: "distance",
    DatabaseSampleColumn.MASS.value: "mass",
}


@dataclass
class UiState:
    measured_file_name: str = ""
    power_level: float = 23.0
    active_filters: DatabaseSampleFilters = field(default_factory=DatabaseSampleFilters)
    last_result: dict[str, Any] | None = None
    updated_at_epoch_s: float | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "measured_file_name": self.measured_file_name,
            "power_level": self.power_level,
            "active_filters": self.active_filters.to_jsonable(),
            "last_result": self.last_result,
            "updated_at_epoch_s": self.updated_at_epoch_s,
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> UiState:
        return cls(
            measured_file_name=str(payload.get("measured_file_name", "")),
            power_level=float(payload.get("power_level", 23.0)),
            active_filters=DatabaseSampleFilters.from_jsonable(
                payload.get("active_filters")
            ),
            last_result=payload.get("last_result"),
            updated_at_epoch_s=payload.get("updated_at_epoch_s"),
        )


def save_ui_state(paths: WorkspacePaths, state: UiState) -> None:
    paths.system_state_dir.mkdir(parents=True, exist_ok=True)
    payload = state.to_jsonable()
    payload["updated_at_epoch_s"] = time.time()
    temp_path = paths.ui_state_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(paths.ui_state_path)


def load_ui_state(paths: WorkspacePaths) -> UiState | None:
    if not paths.ui_state_path.exists():
        return None
    try:
        payload = json.loads(paths.ui_state_path.read_text(encoding="utf-8"))
        return UiState.from_jsonable(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        logging.getLogger(__name__).warning(
            "Saved UI state at %s is invalid — starting fresh.",
            paths.ui_state_path,
        )
        logging.getLogger(__name__).debug("UI state decode error", exc_info=error)
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
        last_result = json.loads(
            paths.legacy_workflow_results_path.read_text(encoding="utf-8")
        )

    return UiState(
        measured_file_name=measured_file_name,
        power_level=power_level,
        active_filters=filters,
        last_result=last_result,
        updated_at_epoch_s=state_data.get("timestamp"),
    )
