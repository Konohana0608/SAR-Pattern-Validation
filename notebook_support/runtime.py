from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkspacePaths:
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
    def from_repo_notebook_dir(cls, notebook_dir: Path) -> WorkspacePaths:
        notebook_dir = notebook_dir.resolve()
        return cls(
            workspace_root=notebook_dir,
            project_root=notebook_dir.parent,
        )

    @classmethod
    def from_workspace(
        cls,
        workspace_root: Path,
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
    user_bin = os.path.expanduser("~/.local/bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if user_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([user_bin] + path_parts)


def _looks_like_project_root(path: Path) -> bool:
    return (path / "src" / "sar_pattern_validation").exists() and (
        path / "data" / "database"
    ).exists()


def _discover_workspace_paths(start_dir: Path) -> WorkspacePaths:
    start_dir = start_dir.resolve()

    if _looks_like_project_root(start_dir):
        notebook_dir = start_dir / "notebooks"
        workspace_root = notebook_dir if notebook_dir.exists() else start_dir
        return WorkspacePaths(
            workspace_root=workspace_root,
            project_root=start_dir,
        )

    if start_dir.name == "notebooks" and _looks_like_project_root(start_dir.parent):
        return WorkspacePaths.from_repo_notebook_dir(start_dir)

    sibling_project_root = start_dir / "sar-pattern-validation"
    if _looks_like_project_root(sibling_project_root):
        return WorkspacePaths(
            workspace_root=start_dir,
            project_root=sibling_project_root,
        )

    return WorkspacePaths.from_workspace(start_dir)


def default_workspace_paths() -> WorkspacePaths:
    return _discover_workspace_paths(Path.cwd())
