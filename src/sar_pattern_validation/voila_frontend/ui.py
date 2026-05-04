from __future__ import annotations

import hashlib
import html
import io
import logging
import os
import threading
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython import get_ipython
from IPython.display import display
from PIL import Image
from traitlets import Bunch

from sar_pattern_validation.sample_catalog import (
    DatabaseSampleCatalog,
    DatabaseSampleColumn,
    DatabaseSampleFilterOption,
    DatabaseSampleFilters,
)

from .models import UiState, WorkflowResultPayload
from .runner import SarPatternValidationRunner
from .runtime import (
    WorkspacePaths,
    default_workspace_paths,
    ensure_notebook_prerequisites,
)
from .state import load_or_migrate_ui_state, save_ui_state

TRANSPARENT_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00"
    b"\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


class GuiColors(str, Enum):
    PRIMARY = "#0090D0"
    WHITE = "#FFFFFF"
    LOADING = "#566670"
    FAIL = "#9B2423"
    TEXT_PRIMARY = "#FFFFFF"


_MAX_LOG_LINES = 200
_UI_CALLBACK_TIMEOUT_S = 10.0
_RUN_STALL_TIMEOUT_S = 60.0
_RUN_STALL_POLL_INTERVAL_S = 0.5
_EXACT_REPEAT_WARNING = (
    "These inputs already match the current results; no new run was started."
)
_RUN_STALLED_ERROR = "The comparison run stopped making progress. Check the backend logs below and try again."
_SERVER_UNREACHABLE_ERROR = (
    "Could not reach the Voila server. Stop any stale Voila processes, restart "
    "Voila, and reload this page."
)
_SERVER_STATUS_BANNER_ID = "sar-pattern-validation-server-status"
_SERVER_STATUS_MONITOR_KEY = "__sarPatternValidationServerMonitorInstalled"
_SERVER_STATUS_POLL_INTERVAL_MS = 5000

_TH = "border:1px solid #555;padding:6px 10px;text-align:center;font-weight:bold;"
_TD = "border:1px solid #555;padding:6px 10px;text-align:center;"


class OutputWidgetHandler(logging.Handler):
    """Custom logging handler rendering into a fixed-height scrollable Output widget.

    Uses widgets.Output (not HTML) so emit() is safe to call from background threads.
    Each call replaces the single display_data output with freshly rendered HTML so
    the log list stays bounded and styled correctly.
    """

    _ROW_STYLE = "margin:0;padding:1px 0;white-space:pre-wrap;word-break:break-all;"
    _CONTAINER_STYLE = (
        "font-family:monospace;font-size:12px;background:#fff;color:#333;padding:4px;"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lines: list[str] = []
        self.out = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="300px",
                border="1px solid black",
                overflow_y="auto",
            )
        )

    def _html(self) -> str:
        import html as _html

        rows = "".join(
            f"<div style='{self._ROW_STYLE}'>{_html.escape(line)}</div>"
            for line in self._lines
        )
        return f"<div style='{self._CONTAINER_STYLE}'>{rows}</div>"

    def emit(self, record) -> None:
        self._lines.insert(0, self.format(record))
        if len(self._lines) > _MAX_LOG_LINES:
            self._lines = self._lines[:_MAX_LOG_LINES]
        self.out.outputs = (
            {
                "output_type": "display_data",
                "data": {"text/html": self._html()},
                "metadata": {},
            },
        )

    def show_logs(self) -> widgets.Output:
        return self.out


def resample_colorbar_to_match_plot_inplace(
    colorbar_path: Path, plot_path: Path
) -> None:
    with Image.open(plot_path) as plot_img:
        target_height = plot_img.size[1]
    with Image.open(colorbar_path) as colorbar_img:
        colorbar_img = colorbar_img.convert("RGBA")
        cb_width, cb_height = colorbar_img.size
        if cb_height == 0:
            raise ValueError("Colorbar height is zero, cannot resample.")
        new_width = int(cb_width * (target_height / cb_height))
        resized_cb = colorbar_img.resize(
            (new_width, target_height), Image.Resampling.LANCZOS
        )
        resized_cb.save(colorbar_path, format="PNG")


def make_transparent_png(width: int, height: int) -> bytes:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def placeholder_from_png(path: Path) -> bytes:
    with Image.open(path) as img:
        width, height = img.size
    return make_transparent_png(width, height)


def _run_stall_timeout_seconds() -> float:
    raw_value = os.getenv("SAR_PATTERN_VALIDATION_RUN_STALL_TIMEOUT_S", "").strip()
    if not raw_value:
        return _RUN_STALL_TIMEOUT_S
    try:
        return max(float(raw_value), 0.1)
    except ValueError:
        return _RUN_STALL_TIMEOUT_S


def _normalize_failure_message(message: str) -> str:
    normalized = message.strip() or "Workflow execution failed."
    lowered = normalized.lower()
    connection_markers = (
        "connection refused",
        "failed to establish a new connection",
        "temporary failure in name resolution",
        "name or service not known",
        "nodename nor servname provided",
        "network is unreachable",
        "no route to host",
        "connection reset by peer",
        "connection aborted",
        "server disconnected",
        "broken pipe",
        "could not connect",
    )
    if any(marker in lowered for marker in connection_markers):
        return _SERVER_UNREACHABLE_ERROR
    return normalized


def _server_connection_monitor_markup() -> str:
    return f"""
    <div id=\"{_SERVER_STATUS_BANNER_ID}\" role=\"alert\" aria-live=\"assertive\"></div>
    <img
        src=\"data:image/gif;base64,R0lGODlhAQABAAAAACw=\"
        alt=\"\"
        style=\"display:none\"
        onload=\"(function(element) {{
            if (window['{_SERVER_STATUS_MONITOR_KEY}']) {{
                element.remove();
                return;
            }}
            window['{_SERVER_STATUS_MONITOR_KEY}'] = true;
            const bannerId = '{_SERVER_STATUS_BANNER_ID}';
            const message = {_SERVER_UNREACHABLE_ERROR!r};
            const render = function(messageText) {{
                const banner = document.getElementById(bannerId);
                if (!banner) {{
                    return;
                }}
                banner.innerHTML = messageText
                    ? '<div style=&quot;background:#FDE8E8;border-left:4px solid {GuiColors.FAIL.value};padding:8px 12px;border-radius:6px;font-size:13px;color:#7B1515;font-family:Arial,sans-serif;&quot;><b>Error:</b> ' + messageText + '</div>'
                    : '';
            }};
            const ping = function() {{
                fetch(window.location.href, {{ method: 'HEAD', cache: 'no-store' }})
                    .then(function(response) {{
                        if (!response.ok) {{
                            throw new Error('request failed');
                        }}
                        render('');
                    }})
                    .catch(function() {{
                        render(message);
                    }});
            }};
            window.addEventListener('offline', function() {{
                render(message);
            }});
            window.addEventListener('online', function() {{
                ping();
            }});
            window.setInterval(ping, {_SERVER_STATUS_POLL_INTERVAL_MS});
            ping();
            element.remove();
        }})(this)\"
    />
    """


class FilterButtonGrid:
    FILTER_ATTR = {
        DatabaseSampleColumn.ANTENNA_TYPE: "antenna_type",
        DatabaseSampleColumn.FREQUENCY: "frequency",
        DatabaseSampleColumn.DISTANCE: "distance",
        DatabaseSampleColumn.MASS: "mass",
    }

    def __init__(
        self,
        catalog: DatabaseSampleCatalog,
        on_change: Callable[[], None] | None = None,
    ):
        self.catalog = catalog
        self.on_change = on_change
        self.filter_options = DatabaseSampleFilters()
        self.filtered_df = self.catalog.filter_dataframe(self.filter_options)
        self.button_groups: dict[DatabaseSampleColumn, list[widgets.ToggleButton]] = {}

    def create_radio_button_grid(self) -> widgets.HBox:
        all_button_columns: list[widgets.VBox] = []
        for (
            column_name,
            unique_values,
        ) in self.catalog.unique_entries_in_columns().items():
            column_enum = DatabaseSampleColumn(column_name)
            if column_enum == DatabaseSampleColumn.FILE_PATH:
                continue
            buttons: list[widgets.ToggleButton] = []
            for value in unique_values:
                btn = widgets.ToggleButton(
                    description=str(value),
                    disabled=False,
                    layout=widgets.Layout(width="100%"),
                )
                btn.observe(self._make_handler(btn, column_enum), "value")
                buttons.append(btn)
            self.button_groups[column_enum] = buttons
            all_button_columns.append(
                widgets.VBox(
                    [widgets.HTML(f"<b>{column_name}</b>"), *buttons],
                    layout=widgets.Layout(
                        flex="1 1 0%",
                        min_width="0",
                        overflow="hidden",
                    ),
                )
            )
        return widgets.HBox(all_button_columns, layout=widgets.Layout(width="100%"))

    def _coerce_value(
        self, column_enum: DatabaseSampleColumn, raw_value: str
    ) -> str | float:
        if column_enum == DatabaseSampleColumn.ANTENNA_TYPE:
            return raw_value
        return float(raw_value)

    def _make_handler(
        self, button: widgets.ToggleButton, column_enum: DatabaseSampleColumn
    ) -> Callable[[Bunch], None]:
        def handler(change: Bunch) -> None:
            if change["name"] != "value":
                return
            column_attr = self.FILTER_ATTR[column_enum]
            value = self._coerce_value(column_enum, change.owner.description)
            group_buttons = self.button_groups[column_enum]
            if change["new"]:
                for sibling in group_buttons:
                    if sibling is not button:
                        sibling.value = False
                setattr(
                    self.filter_options,
                    column_attr,
                    DatabaseSampleFilterOption(column_name=column_enum, value=value),
                )
            else:
                current_filter = getattr(self.filter_options, column_attr)
                if current_filter and current_filter.value == value:
                    setattr(self.filter_options, column_attr, None)
            self.filtered_df = self.catalog.filter_dataframe(self.filter_options)
            self.update_button_states(self.filtered_df)
            if self.on_change is not None:
                self.on_change()

        return handler

    def update_button_states(self, filtered_df: pd.DataFrame) -> None:
        for column_enum, buttons in self.button_groups.items():
            valid_values = set(filtered_df[column_enum.value].astype(str))
            for button in buttons:
                button.disabled = button.description not in valid_values

    def apply_filters(self, filters: DatabaseSampleFilters) -> None:
        self.filter_options = filters.model_copy(deep=True)
        self.filtered_df = self.catalog.filter_dataframe(self.filter_options)
        for column_enum, attr_name in self.FILTER_ATTR.items():
            selected = getattr(self.filter_options, attr_name)
            selected_description = None if selected is None else str(selected.value)
            for button in self.button_groups.get(column_enum, []):
                button.unobserve_all("value")
                button.value = button.description == selected_description
                button.observe(self._make_handler(button, column_enum), "value")
        self.update_button_states(self.filtered_df)
        if self.on_change is not None:
            self.on_change()

    @property
    def selected_reference_path(self) -> Path | None:
        if len(self.filtered_df) != 1:
            return None
        return Path(self.filtered_df[DatabaseSampleColumn.FILE_PATH.value].iloc[0])


class SarGammaComparisonUI:
    def __init__(self, paths: WorkspacePaths | None = None):
        ensure_notebook_prerequisites()
        self.paths = paths or default_workspace_paths()
        self.paths.ensure_runtime_dirs()

        self.logging_window = OutputWidgetHandler()
        self.logging_window.setFormatter(
            logging.Formatter("%(asctime)s  - [%(levelname)s] %(message)s")
        )
        # Attach handler to the package logger so runner.py logs reach the widget too
        _frontend_logger = logging.getLogger("sar_pattern_validation.voila_frontend")
        _frontend_logger.handlers.clear()
        _frontend_logger.addHandler(self.logging_window)
        _frontend_logger.setLevel(logging.INFO)
        _frontend_logger.propagate = False

        self.logger = logging.getLogger(__name__)

        self.catalog = DatabaseSampleCatalog.scan(self.paths.database_path)
        self.runner = SarPatternValidationRunner(self.paths)
        self.workflow_results: WorkflowResultPayload | None = None
        self._rerun_candidate_results: WorkflowResultPayload | None = None
        self._progress_thread = None
        self._workflow_thread = None
        self._stop_event = None
        self._workflow_run_id = 0
        self._active_run_id: int | None = None
        self._stall_watchdog_stop_event: threading.Event | None = None
        self._stall_watchdog_thread: threading.Thread | None = None
        self._last_run_activity_at: float | None = None

        self.radio_button_grid = FilterButtonGrid(self.catalog, self._on_filter_change)
        display(self.create_ui())
        self.restore_state()

    def _build_state(self) -> UiState:
        return UiState(
            measured_file_name=self.uploaded_file_name_label.value,
            power_level=float(self.power_level.value),
            active_filters=self.radio_button_grid.filter_options,
            last_result=self.workflow_results,
        )

    def _persist_state(self) -> None:
        save_ui_state(self.paths, self._build_state())

    def _on_filter_change(self) -> None:
        self._refresh_run_button_state()
        self._persist_state()

    def _on_power_level_change(self, change: Bunch) -> None:
        if change["name"] != "value":
            return
        self._persist_state()
        self._refresh_run_button_state()

    def _measured_file_sha256(self) -> str | None:
        if not self.paths.measured_file_path.exists():
            return None
        return hashlib.sha256(self.paths.measured_file_path.read_bytes()).hexdigest()

    def _same_dataset_as_current_inputs(
        self, results: WorkflowResultPayload | None, reference_path: Path
    ) -> bool:
        if results is None:
            return False
        if results.reference_file_path != str(reference_path):
            return False
        if results.measured_file_sha256 != self._measured_file_sha256():
            return False
        return self._restore_outputs_available()

    def _result_matches_current_inputs(
        self, results: WorkflowResultPayload | None, reference_path: Path
    ) -> bool:
        if not self._same_dataset_as_current_inputs(results, reference_path):
            return False
        if results is None or results.input_power_level_dbm is None:
            return False
        return abs(
            float(results.input_power_level_dbm) - float(self.power_level.value)
        ) < (1e-9)

    def _recalculate_results_for_power(
        self, results: WorkflowResultPayload, *, power_level_dbm: float
    ) -> WorkflowResultPayload | None:
        raw_measured_peak = results.measured_pssar_at_input_power
        if raw_measured_peak is None:
            return None
        measured_pssar_30dbm = raw_measured_peak * (
            10 ** ((30.0 - float(power_level_dbm)) / 10.0)
        )
        scaling_error = (measured_pssar_30dbm / results.reference_pssar) - 1.0
        return results.model_copy(
            update={
                "measured_pssar": measured_pssar_30dbm,
                "measured_pssar_at_input_power": raw_measured_peak,
                "scaling_error": scaling_error,
                "input_power_level_dbm": float(power_level_dbm),
            }
        )

    def _refresh_run_button_state(self) -> None:
        self.run_button.disabled = not (
            not self._is_workflow_running()
            and self.paths.measured_file_path.exists()
            and self.radio_button_grid.selected_reference_path is not None
        )

    def _is_workflow_running(self) -> bool:
        return self._workflow_thread is not None and self._workflow_thread.is_alive()

    def _set_feedback_banner(self, message: str, *, severity: str) -> None:
        palette = {
            "error": ("#FDE8E8", GuiColors.FAIL.value, "#7B1515", "Error"),
            "warning": ("#FFF3CD", "#B8860B", "#7B6015", "Warning"),
            "info": ("#E8F6FD", GuiColors.PRIMARY.value, "#005A8C", "Info"),
        }
        background, border, text_color, label = palette[severity]
        escaped_message = html.escape(message.strip())
        self.feedback_banner.value = (
            f'<div style="background:{background};border-left:4px solid {border};'
            "padding:8px 12px;border-radius:6px;font-size:13px;"
            f'color:{text_color};font-family:Arial,sans-serif;">'
            f"<b>{label}:</b> {escaped_message}</div>"
        )

    def _clear_feedback_banner(self) -> None:
        self.feedback_banner.value = ""

    def _prepare_for_new_run(self) -> None:
        self.workflow_results = None
        self.results_display.value = ""
        self._clear_feedback_banner()
        self.update_images(no_data=True)
        self._rerun_candidate_results = None
        self._persist_state()

    def _dispatch_ui_update(
        self, callback: Callable[..., None], /, *args, **kwargs
    ) -> None:
        ipython = get_ipython()
        kernel = getattr(ipython, "kernel", None)
        io_loop = getattr(kernel, "io_loop", None)

        if io_loop is None or threading.current_thread() is threading.main_thread():
            callback(*args, **kwargs)
            return

        finished = threading.Event()
        errors: list[Exception] = []

        def run_callback() -> None:
            try:
                callback(*args, **kwargs)
            except Exception as error:  # noqa: BLE001
                errors.append(error)
            finally:
                finished.set()

        io_loop.add_callback(run_callback)
        if not finished.wait(timeout=_UI_CALLBACK_TIMEOUT_S):
            self.logger.warning(
                "Timed out while waiting for a UI update callback; applying a fail-safe local update."
            )
            callback(*args, **kwargs)
            return
        if errors:
            raise errors[0]

    def _mark_run_activity(self, run_id: int | None = None) -> None:
        if run_id is not None and run_id != self._active_run_id:
            return
        self._last_run_activity_at = time.monotonic()

    def _cancel_stall_watchdog(self) -> None:
        if self._stall_watchdog_stop_event is not None:
            self._stall_watchdog_stop_event.set()
        if (
            self._stall_watchdog_thread is not None
            and self._stall_watchdog_thread.is_alive()
            and self._stall_watchdog_thread is not threading.current_thread()
        ):
            self._stall_watchdog_thread.join(timeout=1.0)
        self._stall_watchdog_stop_event = None
        self._stall_watchdog_thread = None
        self._last_run_activity_at = None

    def _begin_workflow_run(self) -> int:
        self._workflow_run_id += 1
        self._active_run_id = self._workflow_run_id
        self._mark_run_activity(self._active_run_id)
        return self._workflow_run_id

    def _start_stall_watchdog(self, *, button: widgets.Button, run_id: int) -> None:
        import contextvars

        self._cancel_stall_watchdog()
        self._stall_watchdog_stop_event = threading.Event()
        self._mark_run_activity(run_id)

        def watch_for_stall() -> None:
            while True:
                stop_event = self._stall_watchdog_stop_event
                if stop_event is None:
                    return
                if stop_event.wait(_RUN_STALL_POLL_INTERVAL_S):
                    return
                if run_id != self._active_run_id:
                    return
                if self._last_run_activity_at is None:
                    continue
                if (
                    time.monotonic() - self._last_run_activity_at
                    < _run_stall_timeout_seconds()
                ):
                    continue
                self._dispatch_ui_update(
                    self._handle_workflow_failure,
                    message=_RUN_STALLED_ERROR,
                    button=button,
                    run_id=run_id,
                )
                self._dispatch_ui_update(
                    self._finish_workflow_run,
                    button=button,
                    run_id=run_id,
                )
                return

        watchdog_ctx = contextvars.copy_context()
        self._stall_watchdog_thread = threading.Thread(
            target=watchdog_ctx.run,
            args=(watch_for_stall,),
            daemon=True,
        )
        self._stall_watchdog_thread.start()

    def _start_progress_updater(self) -> None:
        import contextvars

        self._stop_event = threading.Event()
        with self.progress_output:
            self.progress_output.clear_output()
            display(self.progress_bar)

        def update_progress() -> None:
            duration = 240
            interval = 0.1
            steps = int(duration / interval)
            for index in range(steps):
                if self._stop_event is not None and self._stop_event.is_set():
                    break
                self.progress_bar.value = min(0.9, index / steps * 0.9)
                time.sleep(interval)

        self.progress_bar.value = 0.0
        self.progress_bar.bar_style = "info"
        self.progress_bar.style = {"bar_color": GuiColors.PRIMARY.value}
        ctx = contextvars.copy_context()
        self._progress_thread = threading.Thread(
            target=ctx.run, args=(update_progress,), daemon=True
        )
        self._progress_thread.start()

    def _stop_progress_updater(self, *, completed: bool) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._progress_thread is not None and self._progress_thread.is_alive():
            self._progress_thread.join(timeout=1.0)
        if completed:
            for value in range(int(self.progress_bar.value * 100), 101, 5):
                self.progress_bar.value = value / 100
                time.sleep(0.05)
            self.progress_bar.value = self.progress_bar.max
        self.progress_bar.style = {
            "bar_color": (
                GuiColors.PRIMARY.value if completed else GuiColors.FAIL.value
            )
        }
        self.progress_output.clear_output()

    def _run_workflow_task(
        self,
        *,
        button: widgets.Button,
        reference_path: Path,
        power_level_dbm: float,
        measured_file_sha256: str | None,
        run_id: int,
    ) -> None:
        try:
            results = self.runner.run_workflow(
                reference_file_path=reference_path,
                power_level_dbm=power_level_dbm,
                on_log_activity=lambda: self._mark_run_activity(run_id),
            )
            results = results.model_copy(
                update={
                    "reference_file_path": str(reference_path),
                    "measured_file_sha256": measured_file_sha256,
                }
            )
            self._dispatch_ui_update(
                self._handle_workflow_success,
                results=results,
                run_id=run_id,
            )
        except Exception as error:  # noqa: BLE001
            issue = getattr(error, "validation_issue", None)
            message = str(error).strip() or "Workflow execution failed."
            self._dispatch_ui_update(
                self._handle_workflow_failure,
                message=message,
                button=button,
                run_id=run_id,
                validation_issue=issue,
            )
        finally:
            self._dispatch_ui_update(
                self._finish_workflow_run,
                button=button,
                run_id=run_id,
            )

    def _handle_workflow_success(
        self, *, results: WorkflowResultPayload, run_id: int
    ) -> None:
        if run_id != self._active_run_id:
            self.logger.info("Discarded stale workflow success for run %s.", run_id)
            return
        self.workflow_results = results
        self._mark_run_activity(run_id)
        self._stop_progress_updater(completed=True)
        self._persist_state()
        self.update_images()
        self._update_analytical_results(results)
        self.logger.info("SAR Pattern Validation done.")

    def _handle_reused_results(self, *, results: WorkflowResultPayload) -> None:
        self.workflow_results = results
        self._persist_state()
        self.update_images()
        self._update_analytical_results(results)
        power_dbm = (
            float(results.input_power_level_dbm)
            if results.input_power_level_dbm is not None
            else float(self.power_level.value)
        )
        self._set_feedback_banner(
            f"Results updated for {power_dbm:.1f} dBm input power. "
            "Registration and gamma map reused from the previous run.",
            severity="info",
        )
        self.logger.info(
            "Reused previous registration and gamma outputs for power-only rerun."
        )

    def _handle_workflow_failure(
        self,
        *,
        message: str,
        button: widgets.Button | None,
        run_id: int | None,
        validation_issue: dict | None = None,
    ) -> None:
        if run_id is not None and run_id != self._active_run_id:
            self.logger.info("Discarded stale workflow failure for run %s.", run_id)
            return
        if validation_issue is not None:
            issue_message = str(validation_issue.get("message") or "").strip()
            issue_code = str(validation_issue.get("code") or "").strip()
            severity = str(validation_issue.get("severity") or "error").strip().lower()
            if severity not in ("error", "warning", "info"):
                severity = "error"
            display_message = issue_message or _normalize_failure_message(message)
            if issue_code:
                display_message = f"[{issue_code}] {display_message}"
        else:
            display_message = _normalize_failure_message(message)
            severity = "error"
        self.workflow_results = None
        self._stop_progress_updater(completed=False)
        self._set_feedback_banner(display_message, severity=severity)
        self.logger.error(display_message)
        self._persist_state()
        if button is not None:
            button.style = {
                "button_color": GuiColors.PRIMARY.value,
                "text_color": GuiColors.TEXT_PRIMARY.value,
            }
        self._refresh_run_button_state()

    def _finish_workflow_run(
        self, *, button: widgets.Button, run_id: int | None
    ) -> None:
        if run_id is not None and run_id != self._active_run_id:
            return
        self._cancel_stall_watchdog()
        self._workflow_thread = None
        self._rerun_candidate_results = None
        self._active_run_id = None
        button.style = {
            "button_color": GuiColors.PRIMARY.value,
            "text_color": GuiColors.TEXT_PRIMARY.value,
        }
        self._refresh_run_button_state()

    def _reset_after_new_upload(self) -> None:
        self._active_run_id = None
        self._workflow_thread = None
        self._cancel_stall_watchdog()
        self._stop_progress_updater(completed=False)
        self.workflow_results = None
        self._rerun_candidate_results = None
        self.results_display.value = ""
        self._clear_feedback_banner()
        self.update_images(no_data=True)
        self.run_button.style = {
            "button_color": GuiColors.PRIMARY.value,
            "text_color": GuiColors.TEXT_PRIMARY.value,
        }
        self._persist_state()
        self._refresh_run_button_state()

    def handle_button_click(self, button: widgets.Button) -> None:
        try:
            reference_path = self.radio_button_grid.selected_reference_path
            if reference_path is None:
                raise RuntimeError("Select exactly one reference configuration.")
            rerun_candidate = self.workflow_results
            measured_file_sha256 = self._measured_file_sha256()
            if self._result_matches_current_inputs(rerun_candidate, reference_path):
                self._set_feedback_banner(_EXACT_REPEAT_WARNING, severity="warning")
                self.logger.warning(_EXACT_REPEAT_WARNING)
                self._refresh_run_button_state()
                return
            if self._same_dataset_as_current_inputs(rerun_candidate, reference_path):
                reused_results = self._recalculate_results_for_power(
                    rerun_candidate,
                    power_level_dbm=float(self.power_level.value),
                )
                if reused_results is not None:
                    self._handle_reused_results(results=reused_results)
                    self._refresh_run_button_state()
                    return

            self._rerun_candidate_results = rerun_candidate
            run_id = self._begin_workflow_run()
            self._prepare_for_new_run()
            self._start_progress_updater()
            self._start_stall_watchdog(button=button, run_id=run_id)
        except Exception as error:  # noqa: BLE001
            message = str(error).strip() or "Workflow execution failed."
            self._handle_workflow_failure(message=message, button=button, run_id=None)
            return

        button.style = {
            "button_color": GuiColors.LOADING.value,
            "text_color": GuiColors.TEXT_PRIMARY.value,
        }
        button.disabled = True
        ipython = get_ipython()
        kernel = getattr(ipython, "kernel", None)
        io_loop = getattr(kernel, "io_loop", None)

        if io_loop is None:
            self._start_workflow_run(
                button,
                reference_path=reference_path,
                measured_file_sha256=measured_file_sha256,
                power_level_dbm=float(self.power_level.value),
                run_id=run_id,
            )
            return

        # Let any in-flight widget value syncs land before we snapshot inputs for
        # the backend run. This is especially important after restoring state.
        io_loop.call_later(
            0.2,
            self._start_workflow_run,
            button,
            reference_path,
            measured_file_sha256,
            float(self.power_level.value),
            run_id,
        )

    def _start_workflow_run(
        self,
        button: widgets.Button,
        reference_path: Path,
        measured_file_sha256: str | None,
        power_level_dbm: float,
        run_id: int,
    ) -> None:
        import contextvars

        try:
            if run_id != self._active_run_id:
                self.logger.info("Skipped starting stale workflow run %s.", run_id)
                return
            ctx = contextvars.copy_context()
            self._workflow_thread = threading.Thread(
                target=ctx.run,
                args=(self._run_workflow_task,),
                kwargs={
                    "button": button,
                    "reference_path": reference_path,
                    "power_level_dbm": power_level_dbm,
                    "measured_file_sha256": measured_file_sha256,
                    "run_id": run_id,
                },
                daemon=True,
            )
            self._workflow_thread.start()
        except Exception as error:  # noqa: BLE001
            message = str(error).strip() or "Workflow execution failed."
            self._handle_workflow_failure(
                message=message,
                button=button,
                run_id=run_id,
            )
            self._finish_workflow_run(button=button, run_id=run_id)

    def _update_analytical_results(self, results: WorkflowResultPayload) -> None:
        run_power_level_dbm = (
            float(results.input_power_level_dbm)
            if results.input_power_level_dbm is not None
            else float(self.power_level.value)
        )
        measured_at_power = (
            float(results.measured_pssar_at_input_power)
            if results.measured_pssar_at_input_power is not None
            else results.measured_pssar * (10 ** ((run_power_level_dbm - 30.0) / 10.0))
        )
        pssar_pass = abs(results.scaling_error * 100) <= 10.0
        pattern_pass = results.pass_rate_percent >= 100.0

        def result_cell(passed: bool) -> str:
            bg = GuiColors.PRIMARY.value if passed else GuiColors.FAIL.value
            text = "Pass" if passed else "Fail"
            return f'<td style="{_TD}background:{bg};"><b style="color:#000">{text}</b></td>'

        self.results_display.value = f"""
        <div style="font-family:Arial,sans-serif;font-size:13px;">
          <p style="margin:0 0 8px 0;font-weight:bold;font-size:14px;">Peak spatial-average SAR (psSAR)</p>
          <table style="border-collapse:collapse;margin-bottom:16px;">
            <thead><tr>
              <th style="{_TH}">Result</th>
              <th style="{_TH}">Measured, {run_power_level_dbm:.1f} dBm</th>
              <th style="{_TH}">Measured, 30 dBm</th>
              <th style="{_TH}">Reference, 30 dBm</th>
              <th style="{_TH}">Scaling Error [%]</th>
              <th style="{_TH}">Criteria [%]</th>
            </tr></thead>
            <tbody><tr>
              {result_cell(pssar_pass)}
              <td style="{_TD}">{measured_at_power:.2f} W/kg</td>
              <td style="{_TD}">{results.measured_pssar:.2f} W/kg</td>
              <td style="{_TD}">{results.reference_pssar:.2f} W/kg</td>
              <td style="{_TD}">{results.scaling_error * 100:.2f}</td>
              <td style="{_TD}">&le; &plusmn; 10</td>
            </tr></tbody>
          </table>
          <p style="margin:0 0 8px 0;font-weight:bold;font-size:14px;">SAR pattern match</p>
          <table style="border-collapse:collapse;">
            <thead><tr>
              <th style="{_TH}">Result</th>
              <th style="{_TH}">Pass rate</th>
              <th style="{_TH}">Criteria</th>
            </tr></thead>
            <tbody><tr>
              {result_cell(pattern_pass)}
              <td style="{_TD}">{results.pass_rate_percent:.2f}%</td>
              <td style="{_TD}">100%</td>
            </tr></tbody>
          </table>
        </div>
        """

    def _on_file_upload_change(self, change: Bunch) -> None:
        value = change["new"]
        if not value:
            self.uploaded_file_name_label.value = ""
            if self.paths.measured_file_path.exists():
                self.paths.measured_file_path.unlink()
            self._reset_after_new_upload()
            return

        file_info = value[0]
        new_content = file_info["content"]
        new_sha256 = hashlib.sha256(new_content).hexdigest()
        prior_sha256 = (
            self.workflow_results.measured_file_sha256
            if self.workflow_results is not None
            else None
        )

        self.uploaded_file_name_label.value = str(file_info["name"])
        self.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.measured_file_path.write_bytes(new_content)

        if (
            prior_sha256 is not None
            and prior_sha256 == new_sha256
            and self._restore_outputs_available()
        ):
            self._persist_state()
            self._refresh_run_button_state()
            return

        self._reset_after_new_upload()

    def restore_state(self) -> None:
        state = load_or_migrate_ui_state(self.paths)
        if state is None:
            self.update_images(no_data=True)
            self._refresh_run_button_state()
            return

        self.uploaded_file_name_label.value = state.measured_file_name
        self.power_level.value = state.power_level
        self.radio_button_grid.apply_filters(state.active_filters)
        self.workflow_results = state.last_result
        if self.workflow_results is not None and self._restore_outputs_available():
            self.update_images(no_data=False)
            self._update_analytical_results(self.workflow_results)
            self.logger.info("Session state restored successfully")
        else:
            self.update_images(no_data=True)
        self._refresh_run_button_state()

    def _restore_outputs_available(self) -> bool:
        required_files = [
            self.paths.measured_file_path,
            self.paths.reference_image_path,
            self.paths.measured_image_path,
            self.paths.aligned_means_path,
            self.paths.registered_image_path,
            self.paths.gamma_comparison_path,
            self.paths.gamma_comparison_failures_path,
        ]
        return all(path.exists() for path in required_files)

    def update_images(self, *, no_data: bool = False) -> None:
        if no_data:
            image_files: list[Path] = [self.paths.no_data_image] * 8
        else:
            resample_colorbar_to_match_plot_inplace(
                self.paths.aligned_means_colorbar_path,
                self.paths.aligned_means_path,
            )
            resample_colorbar_to_match_plot_inplace(
                self.paths.gamma_comparison_colorbar_path,
                self.paths.gamma_comparison_path,
            )
            image_files = [
                self.paths.reference_image_path,
                self.paths.measured_image_path,
                self.paths.aligned_means_path,
                self.paths.aligned_means_colorbar_path,
                self.paths.registered_image_path,
                self.paths.gamma_comparison_path,
                self.paths.gamma_comparison_colorbar_path,
                self.paths.gamma_comparison_failures_path,
            ]

        widgets_list = [
            self.image_top_left,
            self.image_top_middle,
            self.image_top_right,
            self.colorbar_top,
            self.image_bottom_left,
            self.image_bottom_middle,
            self.colorbar_bottom,
            self.image_bottom_right,
        ]
        for img_widget, path in zip(widgets_list, image_files, strict=True):
            if no_data:
                img_widget.value = TRANSPARENT_PNG
            else:
                try:
                    img_widget.value = path.read_bytes()
                except FileNotFoundError:
                    img_widget.value = TRANSPARENT_PNG

        placeholder_png = (
            TRANSPARENT_PNG
            if no_data
            else placeholder_from_png(self.paths.aligned_means_colorbar_path)
        )
        for placeholder in (
            self.placeholder_1,
            self.placeholder_2,
            self.placeholder_3,
            self.placeholder_4,
        ):
            placeholder.value = placeholder_png

    def create_ui(
        self,
        *,
        left_ratio: float = 0.3,
        right_ratio: float = 0.7,
        side_gap: str = "100px",
    ) -> widgets.VBox:
        def row(children, gap: str = "10px", align: str = "center") -> widgets.HBox:
            return widgets.HBox(
                children=children,
                layout=widgets.Layout(
                    gap=gap,
                    width="100%",
                    align_items=align,
                    overflow="hidden",
                ),
            )

        def col(children, gap: str = "10px") -> widgets.VBox:
            return widgets.VBox(
                children=children,
                layout=widgets.Layout(
                    gap=gap,
                    width="100%",
                    align_items="stretch",
                    overflow="hidden",
                ),
            )

        def flex_item(widget, flex="1", min_width="0"):
            widget.layout = widgets.Layout(
                flex=flex, min_width=min_width, overflow="hidden"
            )
            return widget

        self.upload_1 = widgets.FileUpload(
            accept=".csv",
            multiple=False,
            description="Measured CSV",
            style={
                "button_color": GuiColors.PRIMARY.value,
                "text_color": GuiColors.TEXT_PRIMARY.value,
            },
        )
        self.upload_1.observe(self._on_file_upload_change, names="value")
        self.power_level = widgets.BoundedFloatText(
            value=23.0,
            min=-10,
            max=50,
            step=0.1,
            continuous_update=True,
            description="power level [dBm]:",
            style={"description_width": "initial"},
        )
        self.power_level.observe(self._on_power_level_change, names="value")
        self.uploaded_file_name_label = widgets.Label(value="")

        tooltip = widgets.HTML(
            value="""
            <div style="
                background-color: #E8F6FD;
                color: #005A8C;
                border-left: 4px solid #0090D0;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 14px;
            ">
                <b>ℹ️ Note:</b> The uploaded <code>.csv</code> files must be smaller than <b>10 MB</b>.
            </div>
            """
        )

        left_setup_section = col(
            [
                row(
                    [
                        flex_item(self.upload_1, "1"),
                        flex_item(self.power_level, "1", "150px"),
                    ]
                ),
                self.uploaded_file_name_label,
                tooltip,
                self.radio_button_grid.create_radio_button_grid(),
            ]
        )
        left_setup_section.layout.flex = str(left_ratio)

        self.run_button = widgets.Button(
            description="Compare Patterns",
            style={
                "button_color": GuiColors.PRIMARY.value,
                "text_color": GuiColors.TEXT_PRIMARY.value,
            },
            disabled=True,
        )
        self.run_button.on_click(self.handle_button_click)
        self.server_status_banner = widgets.HTML(
            value=_server_connection_monitor_markup()
        )
        self.feedback_banner = widgets.HTML(value="")
        self.results_display = widgets.HTML(value="")
        run_button_row = row(
            [flex_item(self.run_button, "0 0 auto")],
            gap="0px",
        )

        self.progress_bar = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=1.0,
            description="Progress:",
            bar_style="info",
            style={"bar_color": GuiColors.PRIMARY.value},
            layout=widgets.Layout(width="95%"),
        )
        self.progress_output = widgets.Output(layout=widgets.Layout(width="100%"))
        progress_bar_container = row(
            [self.progress_output], gap="0px", align="flex-start"
        )

        def wrap_image(img: widgets.Image, flex: str = "1") -> widgets.Box:
            return widgets.Box(
                [img],
                layout=widgets.Layout(
                    flex=flex,
                    height="100%",
                    overflow="hidden",
                    align_items="stretch",
                    justify_content="center",
                ),
            )

        def create_main_image() -> tuple[widgets.Image, widgets.Box]:
            image = widgets.Image(
                format="png",
                layout=widgets.Layout(
                    width="100%", height="100%", object_fit="contain"
                ),
            )
            return image, wrap_image(image, flex="4")

        def create_colorbar_image() -> tuple[widgets.Image, widgets.Box]:
            image = widgets.Image(
                format="png",
                layout=widgets.Layout(
                    width="100%", height="100%", object_fit="contain"
                ),
            )
            return image, wrap_image(image, flex="1")

        self.image_top_left, box_top_left = create_main_image()
        self.image_top_middle, box_top_middle = create_main_image()
        self.image_top_right, box_top_right = create_main_image()
        self.image_bottom_left, box_bottom_left = create_main_image()
        self.image_bottom_middle, box_bottom_middle = create_main_image()
        self.image_bottom_right, box_bottom_right = create_main_image()
        self.colorbar_top, box_cb_top = create_colorbar_image()
        self.colorbar_bottom, box_cb_bottom = create_colorbar_image()
        self.placeholder_1, box_ph_1 = create_colorbar_image()
        self.placeholder_2, box_ph_2 = create_colorbar_image()
        self.placeholder_3, box_ph_3 = create_colorbar_image()
        self.placeholder_4, box_ph_4 = create_colorbar_image()
        self.update_images(no_data=True)

        row_layout = widgets.Layout(
            width="100%", align_items="stretch", justify_content="space-between"
        )
        top_row = widgets.HBox(
            [
                widgets.HBox(
                    [box_top_left, box_ph_1],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
                widgets.HBox(
                    [box_top_middle, box_ph_2],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
                widgets.HBox(
                    [box_top_right, box_cb_top],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
            ],
            layout=row_layout,
        )
        bottom_row = widgets.HBox(
            [
                widgets.HBox(
                    [box_bottom_left, box_ph_3],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
                widgets.HBox(
                    [box_bottom_middle, box_cb_bottom],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
                widgets.HBox(
                    [box_bottom_right, box_ph_4],
                    layout=widgets.Layout(width="33%", align_items="stretch"),
                ),
            ],
            layout=row_layout,
        )
        images_section = widgets.Box(
            [widgets.VBox([top_row, bottom_row], layout=widgets.Layout(width="100%"))],
            layout=widgets.Layout(max_height="600px", overflow_y="auto", padding="5px"),
        )

        right_results_section = col(
            [
                run_button_row,
                progress_bar_container,
                self.server_status_banner,
                self.feedback_banner,
                images_section,
                self.results_display,
            ]
        )
        right_results_section.layout.flex = str(right_ratio)
        main_gui_section = widgets.HBox(
            [left_setup_section, right_results_section],
            layout=widgets.Layout(gap=side_gap, width="100%", align_items="flex-start"),
        )
        return widgets.VBox(
            [main_gui_section, self.logging_window.show_logs()],
            layout=widgets.Layout(gap="10px", width="100%", align_items="flex-start"),
        )


def bootstrap_voila_ui(paths: WorkspacePaths | None = None) -> SarGammaComparisonUI:
    return SarGammaComparisonUI(paths=paths)
