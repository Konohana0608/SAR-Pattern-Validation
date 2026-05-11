from __future__ import annotations

import contextvars
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .models import WorkflowResultPayload
from .runtime import WorkspacePaths

LOGGER = logging.getLogger(__name__)


class WorkflowExecutionError(RuntimeError):
    """Frontend-safe workflow execution failure."""

    def __init__(self, message: str, *, validation_issue: dict | None = None) -> None:
        super().__init__(message)
        self.validation_issue = validation_issue


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


def _extract_validation_issue(payload: object) -> dict | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    issue = error.get("validation_issue")
    if isinstance(issue, dict):
        return issue
    return None


def _log_backend_stderr(stderr: str) -> None:
    for line in stderr.splitlines():
        line = line.strip()
        if line:
            LOGGER.debug(line)


def _stream_backend_log_file(
    backend_log_path: Path,
    *,
    stop_event: threading.Event,
    on_log_activity: Callable[[], None] | None = None,
) -> None:
    last_position = 0
    partial_line = ""

    while True:
        if backend_log_path.exists():
            with backend_log_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as handle:
                handle.seek(last_position)
                chunk = handle.read()
                last_position = handle.tell()
            if chunk:
                combined = partial_line + chunk
                lines = combined.splitlines(keepends=True)
                partial_line = ""
                for line in lines:
                    if line.endswith("\n") or line.endswith("\r"):
                        formatted = line.strip()
                        if formatted:
                            LOGGER.info("[backend] %s", formatted)
                            if on_log_activity is not None:
                                on_log_activity()
                    else:
                        partial_line = line

        if stop_event.is_set():
            break
        time.sleep(0.2)

    if partial_line.strip():
        LOGGER.info("[backend] %s", partial_line.strip())
        if on_log_activity is not None:
            on_log_activity()


class SarPatternValidationRunner:
    def __init__(self, paths: WorkspacePaths):
        self.paths = paths

    def _has_local_project_checkout(self) -> bool:
        project_root = self.paths.project_root
        return (project_root / "pyproject.toml").exists() and (
            project_root / "src" / "sar_pattern_validation"
        ).exists()

    def _should_default_to_local_checkout(self) -> bool:
        return self.paths.workspace_root == self.paths.project_root / "notebooks"

    def _backend_log_path(self) -> Path:
        self.paths.system_state_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return self.paths.system_state_dir / f"backend-{timestamp}.log"

    def backend_source_spec(self) -> str:
        mode = os.getenv("SAR_PATTERN_VALIDATION_BACKEND_MODE", "").strip().lower()
        if mode == "local":
            return os.getenv(
                "SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE",
                str(self.paths.project_root),
            )
        if mode == "remote":
            package_url = os.getenv(
                "GITHUB_PACKAGE_URL",
                "https://github.com/ITISFoundation/SAR-Pattern-Validation",
            )
            branch = os.getenv("BRANCH", "main")
            return f"git+{package_url}@{branch}"

        if (
            self._has_local_project_checkout()
            and self._should_default_to_local_checkout()
        ):
            return str(self.paths.project_root)

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
        noise_floor_wkg: float = 0.05,
        on_log_activity: Callable[[], None] | None = None,
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
            "--noise_floor_wkg",
            str(noise_floor_wkg),
        )
        env = os.environ.copy()
        env["MPLBACKEND"] = "agg"
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        backend_log_path = self._backend_log_path()
        env["SAR_PATTERN_VALIDATION_BACKEND_LOG_FILE"] = str(backend_log_path)
        LOGGER.info(
            "Starting comparison: measured=%s reference=%s power_level_dbm=%.1f",
            self.paths.measured_file_path,
            reference_file_path,
            power_level_dbm,
        )
        LOGGER.info("Backend log file: %s", backend_log_path)
        LOGGER.debug("Backend source spec: %s", self.backend_source_spec())
        stop_event = threading.Event()
        log_ctx = contextvars.copy_context()
        log_thread = threading.Thread(
            target=log_ctx.run,
            args=(_stream_backend_log_file,),
            kwargs={
                "backend_log_path": backend_log_path,
                "stop_event": stop_event,
                "on_log_activity": on_log_activity,
            },
            daemon=True,
        )
        log_thread.start()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            stdout, stderr = process.communicate()
        finally:
            stop_event.set()
            log_thread.join(timeout=2.0)

        LOGGER.info(
            "Backend subprocess exited: returncode=%s stdout_bytes=%d stderr_bytes=%d",
            process.returncode,
            len(stdout) if stdout else 0,
            len(stderr) if stderr else 0,
        )

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            message = (
                "Workflow backend returned an invalid response."
                " Check backend logs for details."
                f" Backend log: {backend_log_path}."
                f"{_install_hint(stdout, stderr)}"
            )
            LOGGER.error(message)
            raise WorkflowExecutionError(message) from error

        if process.returncode != 0:
            _log_backend_stderr(stderr)
            issue = _extract_validation_issue(payload)
            if issue is not None:
                base_message = str(issue.get("message") or "").strip()
                message = (
                    base_message or _extract_error_message(payload)
                ) + f" Backend log: {backend_log_path}."
                LOGGER.error("Workflow backend validation issue: %s", message)
                raise WorkflowExecutionError(message, validation_issue=issue)
            message = (
                f"{_extract_error_message(payload)} Backend log: {backend_log_path}."
            )
            LOGGER.error("Workflow backend failed: %s", message)
            raise WorkflowExecutionError(message)

        _log_backend_stderr(stderr)
        LOGGER.info("Comparison completed successfully.")

        return WorkflowResultPayload.model_validate(payload["result"])
