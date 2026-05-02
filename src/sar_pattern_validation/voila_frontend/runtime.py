from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class WorkspacePaths(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    workspace_root: Path
    project_root: Path

    @property
    def database_path(self) -> Path:
        return self.project_root / "data" / "database"

    @property
    def no_data_image(self) -> Path:
        return self.project_root / "assets" / "no-data-transparent.png"

    @property
    def uploaded_data_dir(self) -> Path:
        return self.workspace_root / "uploaded_data"

    @property
    def measured_file_path(self) -> Path:
        return self.uploaded_data_dir / "measured_data.csv"

    @property
    def images_dir(self) -> Path:
        return self.workspace_root / "images"

    @property
    def reference_image_path(self) -> Path:
        return self.images_dir / "reference_image.png"

    @property
    def measured_image_path(self) -> Path:
        return self.images_dir / "measured_image.png"

    @property
    def aligned_means_path(self) -> Path:
        return self.images_dir / "aligned_means_image.png"

    @property
    def aligned_means_colorbar_path(self) -> Path:
        return self.images_dir / "aligned_means_image_colorbar.png"

    @property
    def registered_image_path(self) -> Path:
        return self.images_dir / "registered_image.png"

    @property
    def gamma_comparison_path(self) -> Path:
        return self.images_dir / "gamma_comparison_image.png"

    @property
    def gamma_comparison_colorbar_path(self) -> Path:
        return self.images_dir / "gamma_comparison_image_colorbar_vertical.png"

    @property
    def gamma_comparison_failures_path(self) -> Path:
        return self.images_dir / "gamma_comparison_image_failures.png"

    @property
    def system_state_dir(self) -> Path:
        return self.workspace_root / "system_state"

    @property
    def ui_state_path(self) -> Path:
        return self.system_state_dir / "ui_state.json"

    @property
    def legacy_state_path(self) -> Path:
        return self.system_state_dir / "state.json"

    @property
    def legacy_workflow_results_path(self) -> Path:
        return self.system_state_dir / "workflow_results.json"

    @property
    def legacy_filtered_db_csv_path(self) -> Path:
        return self.system_state_dir / "filtered_db.csv"

    def ensure_runtime_dirs(self) -> None:
        for path in (self.uploaded_data_dir, self.images_dir, self.system_state_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_workspace(
        cls,
        workspace_root: Path,
        *,
        project_dir_name: str = "sar-pattern-validation",
    ) -> WorkspacePaths:
        return cls(
            workspace_root=workspace_root.resolve(),
            project_root=(workspace_root / project_dir_name).resolve(),
        )


def ensure_notebook_prerequisites() -> None:
    if shutil.which("uv") is None:
        subprocess.run(
            ["bash", "-c", "wget -qO- https://astral.sh/uv/install.sh | sh"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.local/bin")


def default_workspace_paths() -> WorkspacePaths:
    return WorkspacePaths.from_workspace(Path.cwd().resolve())


def extend_notebook_sys_path(paths: WorkspacePaths) -> None:
    entries = [paths.project_root, paths.project_root / "src"]
    for entry in entries:
        entry_str = str(entry)
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)
