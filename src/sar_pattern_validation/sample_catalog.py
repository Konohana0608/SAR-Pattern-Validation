from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatabaseSampleColumn(str, Enum):
    ANTENNA_TYPE = "Antenna Type"
    FREQUENCY = "Frequency [MHz]"
    DISTANCE = "Distance [mm]"
    MASS = "Mass [g]"
    FILE_PATH = "File Path"


class DatabaseSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    antenna_type: str
    frequency_mhz: float
    distance_mm: float
    mass_g: float
    file_path: Path

    def as_row(self) -> dict[str, Any]:
        return {
            DatabaseSampleColumn.ANTENNA_TYPE.value: self.antenna_type,
            DatabaseSampleColumn.FREQUENCY.value: self.frequency_mhz,
            DatabaseSampleColumn.DISTANCE.value: self.distance_mm,
            DatabaseSampleColumn.MASS.value: self.mass_g,
            DatabaseSampleColumn.FILE_PATH.value: str(self.file_path),
        }


class DatabaseSampleFilterOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_name: DatabaseSampleColumn
    value: Any

    @field_validator("column_name")
    @classmethod
    def _validate_column(cls, value: DatabaseSampleColumn) -> DatabaseSampleColumn:
        if value == DatabaseSampleColumn.FILE_PATH:
            raise ValueError("FILE_PATH cannot be used as a filter column")
        return value

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("Filter value cannot be None")
        return value

    @property
    def column(self) -> str:
        return self.column_name.value


class DatabaseSampleFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    antenna_type: DatabaseSampleFilterOption | None = None
    frequency: DatabaseSampleFilterOption | None = None
    distance: DatabaseSampleFilterOption | None = None
    mass: DatabaseSampleFilterOption | None = None

    @model_validator(mode="after")
    def _validate_columns(self) -> DatabaseSampleFilters:
        mapping = {
            "antenna_type": DatabaseSampleColumn.ANTENNA_TYPE,
            "frequency": DatabaseSampleColumn.FREQUENCY,
            "distance": DatabaseSampleColumn.DISTANCE,
            "mass": DatabaseSampleColumn.MASS,
        }
        for field_name, expected_column in mapping.items():
            option = getattr(self, field_name)
            if option is not None and option.column_name != expected_column:
                raise ValueError(
                    f"{field_name} filter must use column {expected_column.value}"
                )
        return self

    def iter_active_filters(self) -> list[DatabaseSampleFilterOption]:
        return [option for _, option in self if option is not None]


class DatabaseSampleCatalog(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    database_path: Path
    samples: list[DatabaseSample] = Field(default_factory=list)

    @classmethod
    def scan(cls, database_path: Path) -> DatabaseSampleCatalog:
        samples: list[DatabaseSample] = []
        for file_path in sorted(database_path.glob("**/*.csv")):
            parts = file_path.stem.split("_")
            if "Flat" in parts:
                parts.remove("Flat")
            if len(parts) != 4:
                raise ValueError(f"Unexpected filename format: {file_path.name}")

            antenna_type, frequency, distance, mass = parts
            samples.append(
                DatabaseSample(
                    antenna_type=antenna_type,
                    frequency_mhz=float(frequency.replace("MHz", "")),
                    distance_mm=float(distance.replace("mm", "")),
                    mass_g=float(mass.replace("g", "")),
                    file_path=file_path,
                )
            )
        return cls(database_path=database_path, samples=samples)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [sample.as_row() for sample in self.samples],
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
                [value for value in column.unique() if value is not np.nan]
            )
        return unique_entries

    def filter_dataframe(self, filters: DatabaseSampleFilters) -> pd.DataFrame:
        dataframe = self.to_dataframe()
        for option in filters.iter_active_filters():
            dataframe = dataframe[dataframe[option.column] == option.value]
        return dataframe
