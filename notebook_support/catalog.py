from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd


class DatabaseSampleColumn(str, Enum):
    ANTENNA_TYPE = "Antenna Type"
    FREQUENCY = "Frequency [MHz]"
    DISTANCE = "Distance [mm]"
    MASS = "Mass [g]"
    FILE_PATH = "File Path"


@dataclass
class DatabaseSampleFilterOption:
    column_name: DatabaseSampleColumn
    value: Any

    @property
    def column(self) -> str:
        return self.column_name.value

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name.value,
            "value": self.value,
        }

    @classmethod
    def from_jsonable(
        cls, payload: dict[str, Any] | None
    ) -> DatabaseSampleFilterOption | None:
        if payload is None:
            return None
        column_name = DatabaseSampleColumn(str(payload["column_name"]))
        return cls(column_name=column_name, value=payload["value"])


@dataclass
class DatabaseSampleFilters:
    antenna_type: DatabaseSampleFilterOption | None = None
    frequency: DatabaseSampleFilterOption | None = None
    distance: DatabaseSampleFilterOption | None = None
    mass: DatabaseSampleFilterOption | None = None

    def iter_active_filters(self) -> list[DatabaseSampleFilterOption]:
        items = [
            self.antenna_type,
            self.frequency,
            self.distance,
            self.mass,
        ]
        return [item for item in items if item is not None]

    def copy(self) -> DatabaseSampleFilters:
        return copy.deepcopy(self)

    def to_jsonable(self) -> dict[str, dict[str, Any] | None]:
        return {
            "antenna_type": None
            if self.antenna_type is None
            else self.antenna_type.to_jsonable(),
            "frequency": None
            if self.frequency is None
            else self.frequency.to_jsonable(),
            "distance": None if self.distance is None else self.distance.to_jsonable(),
            "mass": None if self.mass is None else self.mass.to_jsonable(),
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any] | None) -> DatabaseSampleFilters:
        payload = payload or {}
        return cls(
            antenna_type=DatabaseSampleFilterOption.from_jsonable(
                payload.get("antenna_type")
            ),
            frequency=DatabaseSampleFilterOption.from_jsonable(
                payload.get("frequency")
            ),
            distance=DatabaseSampleFilterOption.from_jsonable(payload.get("distance")),
            mass=DatabaseSampleFilterOption.from_jsonable(payload.get("mass")),
        )


@dataclass
class DatabaseSampleCatalog:
    samples: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> DatabaseSampleCatalog:
        raw_samples = list(payload.get("samples", []))
        samples = []
        for sample in raw_samples:
            normalized = dict(sample)
            normalized[DatabaseSampleColumn.FILE_PATH.value] = str(
                normalized[DatabaseSampleColumn.FILE_PATH.value]
            )
            samples.append(normalized)
        return cls(samples=samples)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.samples,
            columns=pd.Index([column.value for column in DatabaseSampleColumn]),
        )

    def unique_entries_in_columns(self) -> dict[str, list[Any]]:
        dataframe = self.to_dataframe()
        unique_entries: dict[str, list[Any]] = {}
        for column_name_raw, column in dataframe.items():
            column_name = str(column_name_raw)
            if column_name == DatabaseSampleColumn.FILE_PATH.value:
                continue
            unique_entries[column_name] = sorted(
                [value for value in column.dropna().unique()]
            )
        return unique_entries

    def filter_dataframe(self, filters: DatabaseSampleFilters) -> pd.DataFrame:
        dataframe = self.to_dataframe()
        for option in filters.iter_active_filters():
            dataframe = dataframe[dataframe[option.column] == option.value]
        return dataframe

    def selected_reference_path(self, filters: DatabaseSampleFilters) -> Path | None:
        dataframe = self.filter_dataframe(filters)
        if len(dataframe) != 1:
            return None
        return Path(str(dataframe[DatabaseSampleColumn.FILE_PATH.value].iloc[0]))
