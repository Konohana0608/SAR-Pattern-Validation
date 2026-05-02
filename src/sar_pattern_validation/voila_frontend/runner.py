from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from .models import WorkflowResultPayload
from .runtime import WorkspacePaths

LOGGER = logging.getLogger(__name__)


class WorkflowExecutionError(RuntimeError):
    """Frontend-safe workflow execution failure."""


def _install_hint(stdout: str, stderr: str) -> str:
    combined_output = f"{stdout}\n{stderr}".lower()
    if "git" not in combined_output:
        return ""
    return (
        " Hint: set SAR_PATTERN_VALIDATION_BACKEND_MODE=local and "
        "SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE=<repo-path> to avoid "
        "remote git installation."
    )


def _extract_error_message(payload: object) -> str:
    if not isinstance(payload, dict):
        return "Workflow execution failed. Check backend logs for details."

    error = payload.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        if message:
            return message

    return "Workflow execution failed. Check backend logs for details."


class SarPatternValidationRunner:
    def __init__(self, paths: WorkspacePaths):
        self.paths = paths

    def backend_source_spec(self) -> str:
        mode = (
            os.getenv("SAR_PATTERN_VALIDATION_BACKEND_MODE", "remote").strip().lower()
        )
        if mode == "local":
            return os.getenv(
                "SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE",
                str(self.paths.project_root),
            )

        package_url = os.getenv(
            "GITHUB_PACKAGE_URL",
            "https://github.com/ITISFoundation/SAR-Pattern-Validation",
        )
        branch = os.getenv("BRANCH", "main")
        return f"git+{package_url}@{branch}"

    def build_command(self, *args: str) -> list[str]:
        return [
            "uvx",
            "--no-cache",
            "--from",
            self.backend_source_spec(),
            "sar-pattern-validation",
            *args,
        ]

    def run_workflow(
        self,
        *,
        reference_file_path: Path,
        power_level_dbm: float,
    ) -> WorkflowResultPayload:
        cmd = self.build_command(
            "--measured_file_path",
            str(self.paths.measured_file_path),
            "--reference_file_path",
            str(reference_file_path),
            "--reference_image_save_path",
            str(self.paths.reference_image_path),
            "--measured_image_save_path",
            str(self.paths.measured_image_path),
            "--aligned_meas_save_path",
            str(self.paths.aligned_means_path),
            "--registered_image_save_path",
            str(self.paths.registered_image_path),
            "--gamma_comparison_image_path",
            str(self.paths.gamma_comparison_path),
            "--power_level_dbm",
            str(power_level_dbm),
        )
        env = os.environ.copy()
        env["MPLBACKEND"] = "agg"
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            message = (
                "Workflow backend returned an invalid response."
                " Check backend logs for details."
                f"{_install_hint(result.stdout, result.stderr)}"
            )
            LOGGER.error(message)
            raise WorkflowExecutionError(message) from error

        if result.returncode != 0:
            message = _extract_error_message(payload)
            LOGGER.error("Workflow backend failed: %s", message)
            raise WorkflowExecutionError(message)

        for line in result.stderr.splitlines():
            if line.strip():
                LOGGER.info(line)

        return WorkflowResultPayload.model_validate(payload["result"])
