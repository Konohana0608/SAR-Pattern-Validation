import hashlib
import io
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pandas as pd
import pytest
from PIL import Image
from traitlets.utils.bunch import Bunch

from sar_pattern_validation.sample_catalog import (
    DatabaseSampleCatalog,
    DatabaseSampleColumn,
    DatabaseSampleFilterOption,
    DatabaseSampleFilters,
)
from sar_pattern_validation.voila_frontend.models import UiState, WorkflowResultPayload
from sar_pattern_validation.voila_frontend.runner import SarPatternValidationRunner
from sar_pattern_validation.voila_frontend.runtime import WorkspacePaths
from sar_pattern_validation.voila_frontend.state import (
    load_or_migrate_ui_state,
    load_ui_state,
    save_ui_state,
)
from sar_pattern_validation.voila_frontend.ui import (
    TRANSPARENT_PNG,
    FilterButtonGrid,
    GuiColors,
    SarGammaComparisonUI,
    make_transparent_png,
    placeholder_from_png,
    resample_colorbar_to_match_plot_inplace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> WorkflowResultPayload:
    defaults = dict(
        pass_rate_percent=95.0,
        evaluated_pixel_count=100,
        passed_pixel_count=95,
        failed_pixel_count=5,
        gamma_image_path=None,
        failure_image_path=None,
        registered_overlay_path=None,
        loaded_images_path=None,
        reference_image_path=None,
        measured_image_path=None,
        aligned_measured_path=None,
        measured_pssar=1.0,
        measured_pssar_at_input_power=0.2,
        reference_pssar=1.1,
        scaling_error=0.1,
        dose_to_agreement=5.0,
        distance_to_agreement=2.0,
        input_power_level_dbm=23.0,
    )
    return WorkflowResultPayload(**{**defaults, **overrides})


def _write_png(path: Path, width: int = 100, height: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGBA", (width, height), (128, 64, 32, 255)) as img:
        img.save(path, format="PNG")


def _completed_process(*, returncode: int, stdout: str, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["sar-pattern-validation"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self) -> bool:
        return False


class _DeferredThread:
    started = False

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        type(self).started = True

    def is_alive(self) -> bool:
        return True


class _ImmediateIOLoop:
    def __init__(self):
        self.calls: list[tuple[float, object, tuple[object, ...]]] = []

    def call_later(self, delay, callback, *args):
        self.calls.append((delay, callback, args))
        callback(*args)


class _NeverRunsIOLoop:
    def add_callback(self, callback):
        self.callback = callback


class _FakePopen:
    def __init__(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str = "",
        on_communicate=None,
    ):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._on_communicate = on_communicate

    def communicate(self):
        if self._on_communicate is not None:
            self._on_communicate()
        return self._stdout, self._stderr


def test_default_workspace_paths_uses_notebooks_dir_for_repo_checkout(
    tmp_path: Path,
) -> None:
    notebook_dir = tmp_path / "sar-pattern-validation" / "notebooks"
    (tmp_path / "sar-pattern-validation" / "src" / "sar_pattern_validation").mkdir(
        parents=True
    )
    (tmp_path / "sar-pattern-validation" / "data" / "database").mkdir(parents=True)
    notebook_dir.mkdir(parents=True)

    from sar_pattern_validation.voila_frontend.runtime import default_workspace_paths

    with patch(
        "sar_pattern_validation.voila_frontend.runtime.Path.cwd",
        return_value=notebook_dir,
    ):
        paths = default_workspace_paths()

    assert paths.workspace_root == notebook_dir
    assert paths.project_root == notebook_dir.parent


def test_default_workspace_paths_uses_notebooks_dir_from_repo_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "sar-pattern-validation"
    (repo_root / "src" / "sar_pattern_validation").mkdir(parents=True)
    (repo_root / "data" / "database").mkdir(parents=True)
    (repo_root / "notebooks").mkdir(parents=True)

    from sar_pattern_validation.voila_frontend.runtime import default_workspace_paths

    with patch(
        "sar_pattern_validation.voila_frontend.runtime.Path.cwd",
        return_value=repo_root,
    ):
        paths = default_workspace_paths()

    assert paths.workspace_root == repo_root / "notebooks"
    assert paths.project_root == repo_root
    assert paths.database_path == repo_root / "data" / "database"


def test_default_workspace_paths_uses_deployment_root_with_sibling_project(
    tmp_path: Path,
) -> None:
    deployment_root = tmp_path / "deployment-root"
    project_root = deployment_root / "sar-pattern-validation"
    (project_root / "src" / "sar_pattern_validation").mkdir(parents=True)
    (project_root / "data" / "database").mkdir(parents=True)
    deployment_root.mkdir(parents=True, exist_ok=True)

    from sar_pattern_validation.voila_frontend.runtime import default_workspace_paths

    with patch(
        "sar_pattern_validation.voila_frontend.runtime.Path.cwd",
        return_value=deployment_root,
    ):
        paths = default_workspace_paths()

    assert paths.workspace_root == deployment_root
    assert paths.project_root == project_root
    assert paths.database_path == project_root / "data" / "database"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_catalog(tmp_path: Path) -> DatabaseSampleCatalog:
    db = tmp_path / "db"
    db.mkdir()
    for name in (
        "dipole_900MHz_Flat_15mm_1g.csv",
        "dipole_900MHz_Flat_15mm_10g.csv",
        "patch_2450MHz_Flat_5mm_1g.csv",
    ):
        (db / name).write_text("x,y,sar\n0,0,1\n", encoding="utf-8")
    return DatabaseSampleCatalog.scan(db)


@pytest.fixture
def filter_grid(small_catalog: DatabaseSampleCatalog) -> FilterButtonGrid:
    grid = FilterButtonGrid(small_catalog)
    grid.create_radio_button_grid()  # populates button_groups
    return grid


# ---------------------------------------------------------------------------
# Image utility tests
# ---------------------------------------------------------------------------


def test_runner_raises_frontend_safe_error_for_backend_failure(
    workspace_with_database: WorkspacePaths,
) -> None:
    runner = SarPatternValidationRunner(workspace_with_database)
    failed_run = _FakePopen(
        returncode=1,
        stdout=json.dumps(
            {
                "status": "error",
                "error": {
                    "type": "ValueError",
                    "message": "Measured CSV could not be parsed.",
                },
            }
        ),
        stderr="Traceback (most recent call last):\nValueError: secret backend detail\n",
    )

    with (
        patch(
            "sar_pattern_validation.voila_frontend.runner.subprocess.Popen",
            return_value=failed_run,
        ),
        pytest.raises(RuntimeError, match="Measured CSV could not be parsed") as error,
    ):
        runner.run_workflow(
            reference_file_path=Path("reference.csv"),
            power_level_dbm=23.0,
        )

    assert "Traceback" not in str(error.value)


def test_runner_sanitizes_invalid_backend_output(
    workspace_with_database: WorkspacePaths,
) -> None:
    runner = SarPatternValidationRunner(workspace_with_database)
    failed_run = _FakePopen(
        returncode=1,
        stdout="not-json",
        stderr="Traceback (most recent call last):\nRuntimeError: secret backend detail\n",
    )

    with (
        patch(
            "sar_pattern_validation.voila_frontend.runner.subprocess.Popen",
            return_value=failed_run,
        ),
        pytest.raises(
            RuntimeError,
            match="Workflow backend returned an invalid response",
        ) as error,
    ):
        runner.run_workflow(
            reference_file_path=Path("reference.csv"),
            power_level_dbm=23.0,
        )

    assert "Traceback" not in str(error.value)
    assert "secret backend detail" not in str(error.value)


def test_runner_sets_timestamped_backend_log_file_env(
    workspace_with_database: WorkspacePaths,
) -> None:
    runner = SarPatternValidationRunner(workspace_with_database)
    success_run = _FakePopen(
        returncode=0,
        stdout=json.dumps(
            {
                "status": "success",
                "result": _make_result().model_dump(mode="json"),
            }
        ),
    )

    with patch(
        "sar_pattern_validation.voila_frontend.runner.subprocess.Popen",
        return_value=success_run,
    ) as popen:
        runner.run_workflow(
            reference_file_path=Path("reference.csv"),
            power_level_dbm=23.0,
        )

    env = popen.call_args.kwargs["env"]
    backend_log_file = Path(env["SAR_PATTERN_VALIDATION_BACKEND_LOG_FILE"])
    assert backend_log_file.parent == workspace_with_database.system_state_dir
    assert backend_log_file.name.startswith("backend-")
    assert backend_log_file.suffix == ".log"


def test_runner_streams_backend_log_file_lines(
    workspace_with_database: WorkspacePaths, caplog: pytest.LogCaptureFixture
) -> None:
    runner = SarPatternValidationRunner(workspace_with_database)
    backend_log_path = workspace_with_database.system_state_dir / "backend-test.log"

    def _write_backend_log() -> None:
        backend_log_path.write_text(
            "Backend started\nRegistration completed\n",
            encoding="utf-8",
        )

    success_run = _FakePopen(
        returncode=0,
        stdout=json.dumps(
            {
                "status": "success",
                "result": _make_result().model_dump(mode="json"),
            }
        ),
        on_communicate=_write_backend_log,
    )

    # Attach caplog directly to the runner logger so capture survives even if
    # a previous test (e.g. one using the sar_ui fixture) set propagate=False
    # on the parent voila_frontend logger.
    runner_logger = logging.getLogger("sar_pattern_validation.voila_frontend.runner")
    runner_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=runner_logger.name)
    try:
        with (
            patch.object(runner, "_backend_log_path", return_value=backend_log_path),
            patch(
                "sar_pattern_validation.voila_frontend.runner.subprocess.Popen",
                return_value=success_run,
            ),
        ):
            runner.run_workflow(
                reference_file_path=Path("reference.csv"),
                power_level_dbm=23.0,
            )
    finally:
        runner_logger.removeHandler(caplog.handler)

    assert "[backend] Backend started" in caplog.text
    assert "[backend] Registration completed" in caplog.text


def test_make_transparent_png_has_correct_dimensions() -> None:
    data = make_transparent_png(80, 120)
    with Image.open(io.BytesIO(data)) as img:
        assert img.size == (80, 120)


def test_make_transparent_png_all_pixels_transparent() -> None:
    data = make_transparent_png(4, 4)
    with Image.open(io.BytesIO(data)) as img:
        pixels = list(img.convert("RGBA").getdata())
        assert all(a == 0 for _, _, _, a in pixels)


def test_resample_colorbar_to_match_plot_height(tmp_path: Path) -> None:
    plot_path = tmp_path / "plot.png"
    colorbar_path = tmp_path / "colorbar.png"
    _write_png(plot_path, width=300, height=200)
    _write_png(colorbar_path, width=50, height=400)

    resample_colorbar_to_match_plot_inplace(colorbar_path, plot_path)

    with Image.open(colorbar_path) as img:
        assert img.size[1] == 200


def test_resample_colorbar_preserves_aspect_ratio(tmp_path: Path) -> None:
    plot_path = tmp_path / "plot.png"
    colorbar_path = tmp_path / "colorbar.png"
    _write_png(plot_path, width=300, height=100)
    _write_png(colorbar_path, width=50, height=200)

    resample_colorbar_to_match_plot_inplace(colorbar_path, plot_path)

    with Image.open(colorbar_path) as img:
        assert img.size == (25, 100)


def test_placeholder_from_png_is_transparent_and_same_size(tmp_path: Path) -> None:
    source = tmp_path / "src.png"
    _write_png(source, width=60, height=90)

    data = placeholder_from_png(source)

    with Image.open(io.BytesIO(data)) as img:
        assert img.size == (60, 90)
        pixels = list(img.convert("RGBA").getdata())
        assert all(a == 0 for _, _, _, a in pixels)


# ---------------------------------------------------------------------------
# FilterButtonGrid tests
# ---------------------------------------------------------------------------


def test_filter_button_grid_creates_buttons_for_each_unique_value(
    filter_grid: FilterButtonGrid, small_catalog: DatabaseSampleCatalog
) -> None:
    unique = small_catalog.unique_entries_in_columns()
    for col_enum, buttons in filter_grid.button_groups.items():
        assert len(buttons) == len(unique[col_enum.value])


def test_filter_button_grid_toggle_sets_filter(filter_grid: FilterButtonGrid) -> None:
    btn = next(
        b
        for b in filter_grid.button_groups[DatabaseSampleColumn.FREQUENCY]
        if b.description == "900.0"
    )
    btn.value = True

    assert filter_grid.filter_options.frequency is not None
    assert filter_grid.filter_options.frequency.value == 900.0


def test_filter_button_grid_toggle_deselects_sibling(
    filter_grid: FilterButtonGrid,
) -> None:
    freq_buttons = filter_grid.button_groups[DatabaseSampleColumn.FREQUENCY]
    btn_900 = next(b for b in freq_buttons if b.description == "900.0")
    btn_2450 = next(b for b in freq_buttons if b.description == "2450.0")

    btn_900.value = True
    btn_2450.value = True

    assert not btn_900.value
    assert filter_grid.filter_options.frequency is not None
    assert filter_grid.filter_options.frequency.value == 2450.0


def test_filter_button_grid_untoggle_clears_filter(
    filter_grid: FilterButtonGrid,
) -> None:
    btn = next(
        b
        for b in filter_grid.button_groups[DatabaseSampleColumn.FREQUENCY]
        if b.description == "900.0"
    )
    btn.value = True
    btn.value = False

    assert filter_grid.filter_options.frequency is None


def test_filter_button_grid_no_reference_path_without_filters(
    filter_grid: FilterButtonGrid,
) -> None:
    assert filter_grid.selected_reference_path is None


def test_filter_button_grid_no_reference_path_when_multiple_match(
    filter_grid: FilterButtonGrid,
) -> None:
    btn_900 = next(
        b
        for b in filter_grid.button_groups[DatabaseSampleColumn.FREQUENCY]
        if b.description == "900.0"
    )
    btn_900.value = True

    assert filter_grid.selected_reference_path is None


def test_filter_button_grid_reference_path_when_unique_match(
    filter_grid: FilterButtonGrid,
) -> None:
    btn_patch = next(
        b
        for b in filter_grid.button_groups[DatabaseSampleColumn.ANTENNA_TYPE]
        if b.description == "patch"
    )
    btn_patch.value = True

    path = filter_grid.selected_reference_path
    assert path is not None
    assert "patch" in path.name


def test_filter_button_grid_on_change_callback_fires(tmp_path: Path) -> None:
    db = tmp_path / "db"
    db.mkdir()
    (db / "dipole_900MHz_Flat_15mm_1g.csv").write_text(
        "x,y,sar\n0,0,1\n", encoding="utf-8"
    )
    catalog = DatabaseSampleCatalog.scan(db)

    calls: list[None] = []
    grid = FilterButtonGrid(catalog, on_change=lambda: calls.append(None))
    grid.create_radio_button_grid()
    grid.button_groups[DatabaseSampleColumn.ANTENNA_TYPE][0].value = True

    assert len(calls) == 1


def test_filter_button_grid_apply_filters_restores_toggle_state(
    filter_grid: FilterButtonGrid,
) -> None:
    filters = DatabaseSampleFilters(
        frequency=DatabaseSampleFilterOption(
            column_name=DatabaseSampleColumn.FREQUENCY,
            value=900.0,
        )
    )
    filter_grid.apply_filters(filters)

    btn_900 = next(
        b
        for b in filter_grid.button_groups[DatabaseSampleColumn.FREQUENCY]
        if b.description == "900.0"
    )
    assert btn_900.value is True


def test_filter_button_grid_apply_filters_clears_previous_selection(
    filter_grid: FilterButtonGrid,
) -> None:
    btn_dipole = next(
        b
        for b in filter_grid.button_groups[DatabaseSampleColumn.ANTENNA_TYPE]
        if b.description == "dipole"
    )
    btn_dipole.value = True

    filter_grid.apply_filters(DatabaseSampleFilters())

    assert not btn_dipole.value


def test_filter_button_grid_buttons_disabled_when_value_not_in_filtered_results(
    filter_grid: FilterButtonGrid,
) -> None:
    btn_dipole = next(
        b
        for b in filter_grid.button_groups[DatabaseSampleColumn.ANTENNA_TYPE]
        if b.description == "dipole"
    )
    btn_patch = next(
        b
        for b in filter_grid.button_groups[DatabaseSampleColumn.ANTENNA_TYPE]
        if b.description == "patch"
    )
    btn_dipole.value = True

    assert btn_patch.disabled


# ---------------------------------------------------------------------------
# SarGammaComparisonUI tests
# ---------------------------------------------------------------------------


class TestSarGammaComparisonUI:
    def test_run_button_is_disabled_on_init(self, sar_ui: SarGammaComparisonUI) -> None:
        assert sar_ui.run_button.disabled

    def test_expected_widget_types_are_created(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        import ipywidgets as widgets

        assert isinstance(sar_ui.run_button, widgets.Button)
        assert isinstance(sar_ui.upload_1, widgets.FileUpload)
        assert isinstance(sar_ui.progress_bar, widgets.FloatProgress)
        assert isinstance(sar_ui.feedback_banner, widgets.HTML)
        assert isinstance(sar_ui.results_display, widgets.HTML)

    def test_run_button_enables_when_file_and_unique_filter(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")

        with patch.object(
            type(sar_ui.radio_button_grid),
            "selected_reference_path",
            new_callable=PropertyMock,
            return_value=tmp_path / "ref.csv",
        ):
            sar_ui._refresh_run_button_state()

        assert not sar_ui.run_button.disabled

    def test_run_button_stays_disabled_without_measured_file(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        with patch.object(
            type(sar_ui.radio_button_grid),
            "selected_reference_path",
            new_callable=PropertyMock,
            return_value=tmp_path / "ref.csv",
        ):
            sar_ui._refresh_run_button_state()

        assert sar_ui.run_button.disabled

    def test_run_button_stays_disabled_without_reference_selection(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y\n0,0\n", encoding="utf-8")
        sar_ui._refresh_run_button_state()

        assert sar_ui.run_button.disabled

    def test_build_state_captures_power_level(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        sar_ui.power_level.value = 17.5
        assert sar_ui._build_state().power_level == 17.5

    def test_build_state_captures_file_name(self, sar_ui: SarGammaComparisonUI) -> None:
        sar_ui.uploaded_file_name_label.value = "meas.csv"
        assert sar_ui._build_state().measured_file_name == "meas.csv"

    def test_build_state_captures_workflow_results(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        sar_ui.workflow_results = _make_result(pass_rate_percent=88.0)
        last_result = sar_ui._build_state().last_result
        assert last_result is not None
        assert last_result.pass_rate_percent == 88.0

    def test_file_upload_change_writes_file_to_disk(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        content = b"x,y,sar\n0,0,1\n"
        bunch = Bunch(new=[{"name": "data.csv", "content": content}])  # type: ignore
        sar_ui._on_file_upload_change(bunch)

        assert sar_ui.paths.measured_file_path.exists()
        assert sar_ui.paths.measured_file_path.read_bytes() == content

    def test_file_upload_change_updates_label(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        bunch = Bunch(new=[{"name": "my_data.csv", "content": b"x\n"}])  # type: ignore
        sar_ui._on_file_upload_change(bunch)
        assert sar_ui.uploaded_file_name_label.value == "my_data.csv"

    def test_file_upload_empty_clears_label(self, sar_ui: SarGammaComparisonUI) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")
        sar_ui.uploaded_file_name_label.value = "old.csv"
        sar_ui._on_file_upload_change(Bunch(new=[]))
        assert sar_ui.uploaded_file_name_label.value == ""
        assert sar_ui.paths.measured_file_path.exists() is False

    def test_file_upload_change_clears_prior_results_and_feedback(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        sar_ui.workflow_results = _make_result(pass_rate_percent=88.0)
        sar_ui.results_display.value = "<div>Old Results</div>"
        sar_ui._set_feedback_banner("Previous warning", severity="warning")
        for attr_name in (
            "reference_image_path",
            "measured_image_path",
            "aligned_means_path",
            "aligned_means_colorbar_path",
            "registered_image_path",
            "gamma_comparison_path",
            "gamma_comparison_colorbar_path",
            "gamma_comparison_failures_path",
        ):
            _write_png(getattr(sar_ui.paths, attr_name))

        sar_ui._on_file_upload_change(
            Bunch(new=[{"name": "replacement.csv", "content": b"x,y,sar\n0,0,2\n"}])
        )

        assert sar_ui.workflow_results is None
        assert sar_ui.results_display.value == ""
        assert sar_ui.feedback_banner.value == ""
        assert sar_ui._build_state().last_result is None
        assert bytes(sar_ui.image_top_left.value) == TRANSPARENT_PNG

    def test_update_analytical_results_shows_pass(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        sar_ui._update_analytical_results(
            _make_result(
                evaluated_pixel_count=50, passed_pixel_count=50, failed_pixel_count=0
            )
        )
        assert "Pass" in sar_ui.results_display.value
        assert GuiColors.PRIMARY.value in sar_ui.results_display.value

    def test_update_analytical_results_shows_fail(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        sar_ui._update_analytical_results(
            _make_result(
                evaluated_pixel_count=100, passed_pixel_count=60, failed_pixel_count=40
            )
        )
        assert "Fail" in sar_ui.results_display.value
        assert GuiColors.FAIL.value in sar_ui.results_display.value

    def test_update_analytical_results_shows_pass_rate(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        sar_ui._update_analytical_results(_make_result(pass_rate_percent=87.3))
        assert "87.3" in sar_ui.results_display.value

    def test_update_analytical_results_uses_result_run_power_not_live_widget(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        sar_ui.power_level.value = 30.0
        sar_ui._update_analytical_results(
            _make_result(
                measured_pssar=1.0,
                measured_pssar_at_input_power=0.2,
                input_power_level_dbm=23.0,
            )
        )

        assert "Measured, 23.0 dBm" in sar_ui.results_display.value
        assert ">0.20 W/kg<" in sar_ui.results_display.value
        assert ">1.00 W/kg<" in sar_ui.results_display.value

    def test_update_images_no_data_sets_transparent_pixels(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        from sar_pattern_validation.voila_frontend.ui import TRANSPARENT_PNG

        sar_ui.update_images(no_data=True)
        assert sar_ui.image_top_left.value == TRANSPARENT_PNG
        assert sar_ui.image_bottom_right.value == TRANSPARENT_PNG

    def test_update_images_with_data_reads_files(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        for attr_name in (
            "reference_image_path",
            "measured_image_path",
            "aligned_means_path",
            "aligned_means_colorbar_path",
            "registered_image_path",
            "gamma_comparison_path",
            "gamma_comparison_colorbar_path",
            "gamma_comparison_failures_path",
        ):
            _write_png(getattr(sar_ui.paths, attr_name))

        sar_ui.update_images(no_data=False)

        assert sar_ui.image_top_left.value != b""

    def test_restore_state_applies_persisted_power_level(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        state = UiState(measured_file_name="restored.csv", power_level=12.0)
        save_ui_state(sar_ui.paths, state)
        sar_ui.restore_state()

        assert sar_ui.uploaded_file_name_label.value == "restored.csv"
        assert sar_ui.power_level.value == 12.0

    def test_power_level_widget_syncs_continuously(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        assert sar_ui.power_level.continuous_update is True

    def test_ui_includes_server_connection_monitor_banner(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        assert (
            "sar-pattern-validation-server-status" in sar_ui.server_status_banner.value
        )
        assert "Could not reach the Voila server" in sar_ui.server_status_banner.value

    def test_power_level_change_refreshes_run_button_state(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")
        sar_ui.run_button.disabled = True

        with patch.object(
            type(sar_ui.radio_button_grid),
            "selected_reference_path",
            new_callable=PropertyMock,
            return_value=tmp_path / "reference.csv",
        ):
            sar_ui._on_power_level_change(Bunch(name="value", new=11.0, old=23.0))

        assert sar_ui.run_button.disabled is False

    def test_restore_state_with_no_state_file_does_not_crash(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        assert not sar_ui.paths.ui_state_path.exists()
        sar_ui.restore_state()

    def test_handle_button_click_shows_sanitized_error_banner(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")

        with (
            patch.object(sar_ui, "_start_progress_updater"),
            patch.object(sar_ui, "_start_stall_watchdog"),
            patch.object(sar_ui, "_stop_progress_updater"),
            patch(
                "sar_pattern_validation.voila_frontend.ui.threading.Thread",
                _ImmediateThread,
            ),
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=tmp_path / "reference.csv",
            ),
            patch.object(
                sar_ui.runner,
                "run_workflow",
                side_effect=RuntimeError("Measured CSV could not be parsed."),
            ),
        ):
            sar_ui.handle_button_click(sar_ui.run_button)

        assert "Measured CSV could not be parsed." in sar_ui.feedback_banner.value
        assert "Traceback" not in sar_ui.feedback_banner.value
        assert sar_ui.run_button.disabled is False

    def test_handle_button_click_connection_failure_shows_server_error(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")

        with (
            patch.object(sar_ui, "_start_progress_updater"),
            patch.object(sar_ui, "_start_stall_watchdog"),
            patch.object(sar_ui, "_stop_progress_updater"),
            patch(
                "sar_pattern_validation.voila_frontend.ui.threading.Thread",
                _ImmediateThread,
            ),
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=tmp_path / "reference.csv",
            ),
            patch.object(
                sar_ui.runner,
                "run_workflow",
                side_effect=ConnectionError("[Errno 111] Connection refused"),
            ),
        ):
            sar_ui.handle_button_click(sar_ui.run_button)

        assert "Could not reach the Voila server" in sar_ui.feedback_banner.value
        assert sar_ui.run_button.disabled is False

    def test_handle_button_click_warns_and_skips_exact_repeat(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        measured_bytes = b"x,y,sar\n0,0,1\n"
        sar_ui.paths.measured_file_path.write_bytes(measured_bytes)
        reference_path = tmp_path / "reference.csv"
        measured_sha256 = hashlib.sha256(measured_bytes).hexdigest()
        sar_ui.workflow_results = _make_result(
            pass_rate_percent=88.0,
            input_power_level_dbm=23.0,
            reference_file_path=str(reference_path),
            measured_file_sha256=measured_sha256,
        )
        sar_ui.results_display.value = "<div>Existing Results</div>"

        for attr_name in (
            "reference_image_path",
            "measured_image_path",
            "aligned_means_path",
            "aligned_means_colorbar_path",
            "registered_image_path",
            "gamma_comparison_path",
            "gamma_comparison_colorbar_path",
            "gamma_comparison_failures_path",
        ):
            _write_png(getattr(sar_ui.paths, attr_name))

        with (
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=reference_path,
            ),
            patch.object(sar_ui.runner, "run_workflow") as run_workflow,
        ):
            sar_ui.handle_button_click(sar_ui.run_button)

        run_workflow.assert_not_called()
        assert "already match the current results" in sar_ui.feedback_banner.value
        assert "Existing Results" in sar_ui.results_display.value
        assert sar_ui.run_button.disabled is False

    def test_handle_button_click_clears_prior_results_before_rerun(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")
        sar_ui.workflow_results = _make_result(pass_rate_percent=88.0)
        sar_ui.results_display.value = "<div>Old Results</div>"

        for attr_name in (
            "reference_image_path",
            "measured_image_path",
            "aligned_means_path",
            "aligned_means_colorbar_path",
            "registered_image_path",
            "gamma_comparison_path",
            "gamma_comparison_colorbar_path",
            "gamma_comparison_failures_path",
        ):
            _write_png(getattr(sar_ui.paths, attr_name))

        def _assert_cleared_then_return(
            *,
            reference_file_path: Path,
            power_level_dbm: float,
            on_log_activity=None,
        ):
            assert sar_ui.workflow_results is None
            assert sar_ui.results_display.value == ""
            return _make_result(pass_rate_percent=91.0)

        with (
            patch.object(sar_ui, "_start_progress_updater"),
            patch.object(sar_ui, "_start_stall_watchdog"),
            patch.object(sar_ui, "_stop_progress_updater"),
            patch(
                "sar_pattern_validation.voila_frontend.ui.threading.Thread",
                _ImmediateThread,
            ),
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=tmp_path / "reference.csv",
            ),
            patch.object(
                sar_ui.runner,
                "run_workflow",
                side_effect=_assert_cleared_then_return,
            ),
            patch.object(sar_ui, "_persist_state"),
        ):
            sar_ui.handle_button_click(sar_ui.run_button)

        assert "Old Results" not in sar_ui.results_display.value
        assert "91.0" in sar_ui.results_display.value
        assert (
            sar_ui.image_top_left.value
            == sar_ui.paths.reference_image_path.read_bytes()
        )

    def test_handle_button_click_persists_cleared_state_before_rerun(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")
        sar_ui.workflow_results = _make_result(pass_rate_percent=88.0)
        sar_ui.results_display.value = "<div>Old Results</div>"

        persisted_results: list[WorkflowResultPayload | None] = []

        def _capture_persisted_state() -> None:
            persisted_results.append(sar_ui._build_state().last_result)

        with (
            patch.object(sar_ui, "_start_progress_updater"),
            patch.object(sar_ui, "_start_stall_watchdog"),
            patch.object(sar_ui, "_stop_progress_updater"),
            patch(
                "sar_pattern_validation.voila_frontend.ui.threading.Thread",
                _ImmediateThread,
            ),
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=tmp_path / "reference.csv",
            ),
            patch.object(
                sar_ui.runner,
                "run_workflow",
                return_value=_make_result(pass_rate_percent=91.0),
            ),
            patch.object(sar_ui, "update_images"),
            patch.object(
                sar_ui, "_persist_state", side_effect=_capture_persisted_state
            ),
        ):
            sar_ui.handle_button_click(sar_ui.run_button)

        assert len(persisted_results) >= 2
        assert persisted_results[0] is None
        assert persisted_results[-1] is not None
        assert persisted_results[-1].pass_rate_percent == 91.0

    def test_handle_button_click_reuses_existing_results_for_power_only_rerun(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        measured_bytes = b"x,y,sar\n0,0,1\n"
        sar_ui.paths.measured_file_path.write_bytes(measured_bytes)
        reference_path = tmp_path / "reference.csv"
        measured_sha256 = hashlib.sha256(measured_bytes).hexdigest()

        sar_ui.workflow_results = _make_result(
            pass_rate_percent=88.0,
            measured_pssar=1.0,
            measured_pssar_at_input_power=0.2,
            reference_pssar=1.1,
            scaling_error=-0.1,
            input_power_level_dbm=23.0,
            reference_file_path=str(reference_path),
            measured_file_sha256=measured_sha256,
        )

        for attr_name in (
            "reference_image_path",
            "measured_image_path",
            "aligned_means_path",
            "aligned_means_colorbar_path",
            "registered_image_path",
            "gamma_comparison_path",
            "gamma_comparison_colorbar_path",
            "gamma_comparison_failures_path",
        ):
            _write_png(getattr(sar_ui.paths, attr_name))

        sar_ui.power_level.value = 10.0

        with (
            patch.object(sar_ui, "_start_progress_updater"),
            patch.object(sar_ui, "_start_stall_watchdog"),
            patch.object(sar_ui, "_stop_progress_updater"),
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=reference_path,
            ),
            patch.object(sar_ui.runner, "run_workflow") as run_workflow,
        ):
            sar_ui.handle_button_click(sar_ui.run_button)

        run_workflow.assert_not_called()
        assert sar_ui.workflow_results is not None
        assert sar_ui.workflow_results.input_power_level_dbm == 10.0
        assert sar_ui.workflow_results.measured_pssar_at_input_power == pytest.approx(
            0.2
        )
        assert sar_ui.workflow_results.measured_pssar == pytest.approx(20.0)
        assert sar_ui.run_button.disabled is False

    def test_dispatch_ui_update_timeout_uses_fail_safe_local_callback(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        calls: list[str] = []
        fake_ipython = type(
            "FakeIPython",
            (),
            {"kernel": type("FakeKernel", (), {"io_loop": _NeverRunsIOLoop()})()},
        )()

        with (
            patch(
                "sar_pattern_validation.voila_frontend.ui._UI_CALLBACK_TIMEOUT_S",
                0.01,
            ),
            patch(
                "sar_pattern_validation.voila_frontend.ui.get_ipython",
                return_value=fake_ipython,
            ),
        ):
            sar_ui._dispatch_ui_update(calls.append, "applied")

        assert calls == ["applied"]

    def test_stall_watchdog_surfaces_error_and_reenables_button(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")
        sar_ui._workflow_thread = _DeferredThread()
        run_id = sar_ui._begin_workflow_run()
        sar_ui.run_button.disabled = True

        with (
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=tmp_path / "reference.csv",
            ),
            patch(
                "sar_pattern_validation.voila_frontend.ui._RUN_STALL_TIMEOUT_S",
                0.01,
            ),
            patch(
                "sar_pattern_validation.voila_frontend.ui._RUN_STALL_POLL_INTERVAL_S",
                0.01,
            ),
        ):
            sar_ui._start_stall_watchdog(button=sar_ui.run_button, run_id=run_id)
            sar_ui._last_run_activity_at = time.monotonic() - 1.0
            # Watchdog dispatches failure directly via _dispatch_ui_update; wait for it.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and sar_ui._active_run_id is not None:
                time.sleep(0.02)

        assert "stopped making progress" in sar_ui.feedback_banner.value
        assert sar_ui.run_button.disabled is False
        assert sar_ui._active_run_id is None

    def test_runner_backend_logs_reach_output_widget(
        self, sar_ui: SarGammaComparisonUI
    ) -> None:
        backend_log_path = sar_ui.paths.system_state_dir / "backend-ui.log"

        def _write_backend_log() -> None:
            backend_log_path.write_text(
                "Backend line one\nBackend line two\n", encoding="utf-8"
            )

        success_run = _FakePopen(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "success",
                    "result": _make_result().model_dump(mode="json"),
                }
            ),
            on_communicate=_write_backend_log,
        )

        with (
            patch.object(
                sar_ui.runner, "_backend_log_path", return_value=backend_log_path
            ),
            patch(
                "sar_pattern_validation.voila_frontend.runner.subprocess.Popen",
                return_value=success_run,
            ),
        ):
            sar_ui.runner.run_workflow(
                reference_file_path=Path("reference.csv"),
                power_level_dbm=23.0,
            )

        assert any("Backend line one" in line for line in sar_ui.logging_window._lines)
        assert any("Backend line two" in line for line in sar_ui.logging_window._lines)

    def test_handle_button_click_starts_background_workflow_thread(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")
        _DeferredThread.started = False

        with (
            patch.object(sar_ui, "_start_progress_updater"),
            patch.object(sar_ui, "_start_stall_watchdog"),
            patch(
                "sar_pattern_validation.voila_frontend.ui.threading.Thread",
                _DeferredThread,
            ),
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=tmp_path / "reference.csv",
            ),
            patch.object(sar_ui.runner, "run_workflow") as run_workflow,
        ):
            sar_ui.handle_button_click(sar_ui.run_button)

        run_workflow.assert_not_called()
        assert _DeferredThread.started is True
        assert sar_ui.run_button.disabled is True

    def test_handle_button_click_defers_start_until_kernel_io_loop_turn(
        self, sar_ui: SarGammaComparisonUI, tmp_path: Path
    ) -> None:
        sar_ui.paths.measured_file_path.parent.mkdir(parents=True, exist_ok=True)
        sar_ui.paths.measured_file_path.write_text("x,y,sar\n0,0,1\n", encoding="utf-8")
        _DeferredThread.started = False
        io_loop = _ImmediateIOLoop()
        fake_ipython = type(
            "FakeIPython",
            (),
            {"kernel": type("FakeKernel", (), {"io_loop": io_loop})()},
        )()

        with (
            patch.object(sar_ui, "_start_progress_updater"),
            patch.object(sar_ui, "_start_stall_watchdog"),
            patch(
                "sar_pattern_validation.voila_frontend.ui.threading.Thread",
                _DeferredThread,
            ),
            patch(
                "sar_pattern_validation.voila_frontend.ui.get_ipython",
                return_value=fake_ipython,
            ),
            patch.object(
                type(sar_ui.radio_button_grid),
                "selected_reference_path",
                new_callable=PropertyMock,
                return_value=tmp_path / "reference.csv",
            ),
            patch.object(sar_ui.runner, "run_workflow") as run_workflow,
        ):
            sar_ui.handle_button_click(sar_ui.run_button)

        run_workflow.assert_not_called()
        assert len(io_loop.calls) == 1
        delay, _, _ = io_loop.calls[0]
        assert delay == pytest.approx(0.2)
        assert _DeferredThread.started is True


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------


def test_save_and_load_ui_state_roundtrip(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_workspace(tmp_path)
    state = UiState(
        measured_file_name="test.csv",
        power_level=30.0,
        last_result=_make_result(),
    )
    save_ui_state(paths, state)

    loaded = load_ui_state(paths)

    assert loaded is not None
    assert loaded.measured_file_name == "test.csv"
    assert loaded.power_level == 30.0
    assert loaded.last_result is not None
    assert loaded.last_result.pass_rate_percent == 95.0


def test_save_ui_state_sets_updated_at_timestamp(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_workspace(tmp_path)
    save_ui_state(paths, UiState())

    loaded = load_ui_state(paths)
    assert loaded is not None
    assert loaded.updated_at_epoch_s is not None
    assert loaded.updated_at_epoch_s > 0


def test_load_ui_state_returns_none_when_file_missing(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_workspace(tmp_path)
    assert load_ui_state(paths) is None


def test_save_ui_state_overwrites_previous(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_workspace(tmp_path)
    save_ui_state(paths, UiState(power_level=10.0))
    save_ui_state(paths, UiState(power_level=20.0))

    loaded = load_ui_state(paths)
    assert loaded is not None
    assert loaded.power_level == 20.0


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


def test_database_sample_catalog_scans_and_filters(tmp_path: Path) -> None:
    database_path = tmp_path / "database"
    database_path.mkdir()
    for name in (
        "dipole_900MHz_Flat_15mm_1g.csv",
        "dipole_900MHz_Flat_15mm_10g.csv",
        "patch_2450MHz_Flat_5mm_1g.csv",
    ):
        (database_path / name).write_text("x,y,sar\n0,0,1\n", encoding="utf-8")

    catalog = DatabaseSampleCatalog.scan(database_path)

    assert len(catalog.samples) == 3
    unique_values = catalog.unique_entries_in_columns()
    assert unique_values[DatabaseSampleColumn.ANTENNA_TYPE.value] == ["dipole", "patch"]

    filtered = catalog.filter_dataframe(
        DatabaseSampleFilters(
            frequency=DatabaseSampleFilterOption(
                column_name=DatabaseSampleColumn.FREQUENCY,
                value=900.0,
            )
        )
    )
    assert len(filtered) == 2


def test_runner_prefers_local_checkout_only_for_repo_notebooks(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    project_root = workspace_root / "sar-pattern-validation"
    (project_root / "src" / "sar_pattern_validation").mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='sar-pattern-validation'\n", encoding="utf-8"
    )
    paths = WorkspacePaths.from_workspace(workspace_root)
    runner = SarPatternValidationRunner(paths)

    notebook_paths = WorkspacePaths.from_repo_notebook_dir(project_root / "notebooks")
    (project_root / "notebooks").mkdir(parents=True, exist_ok=True)
    notebook_runner = SarPatternValidationRunner(notebook_paths)

    original_mode = os.environ.get("SAR_PATTERN_VALIDATION_BACKEND_MODE")
    original_local = os.environ.get("SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE")
    original_url = os.environ.get("GITHUB_PACKAGE_URL")
    original_branch = os.environ.get("BRANCH")
    try:
        os.environ.pop("SAR_PATTERN_VALIDATION_BACKEND_MODE", None)
        os.environ["GITHUB_PACKAGE_URL"] = "https://example.invalid/repo"
        os.environ["BRANCH"] = "feature"
        assert (
            runner.backend_source_spec() == "git+https://example.invalid/repo@feature"
        )
        assert notebook_runner.backend_source_spec() == str(project_root)

        os.environ["SAR_PATTERN_VALIDATION_BACKEND_MODE"] = "local"
        os.environ["SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE"] = "/tmp/local-repo"
        assert runner.backend_source_spec() == "/tmp/local-repo"
        assert notebook_runner.backend_source_spec() == "/tmp/local-repo"
        assert runner.build_command("--help")[:5] == [
            "uvx",
            "--no-cache",
            "--from",
            "/tmp/local-repo",
            "sar-pattern-validation",
        ]

        os.environ["SAR_PATTERN_VALIDATION_BACKEND_MODE"] = "remote"
        assert (
            runner.backend_source_spec() == "git+https://example.invalid/repo@feature"
        )
        assert (
            notebook_runner.backend_source_spec()
            == "git+https://example.invalid/repo@feature"
        )
    finally:
        _restore_env("SAR_PATTERN_VALIDATION_BACKEND_MODE", original_mode)
        _restore_env("SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE", original_local)
        _restore_env("GITHUB_PACKAGE_URL", original_url)
        _restore_env("BRANCH", original_branch)


def test_load_or_migrate_ui_state_from_legacy_files(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    paths = WorkspacePaths.from_workspace(workspace_root)
    paths.system_state_dir.mkdir(parents=True, exist_ok=True)

    paths.legacy_state_path.write_text(
        json.dumps(
            {
                "measured_file_name": "measured.csv",
                "power_level": 17.5,
                "timestamp": 123.0,
            }
        ),
        encoding="utf-8",
    )
    paths.legacy_workflow_results_path.write_text(
        json.dumps(
            {
                "pass_rate_percent": 98.0,
                "evaluated_pixel_count": 100,
                "passed_pixel_count": 98,
                "failed_pixel_count": 2,
                "gamma_image_path": str(workspace_root / "images" / "gamma.png"),
                "failure_image_path": None,
                "registered_overlay_path": None,
                "loaded_images_path": None,
                "reference_image_path": None,
                "measured_image_path": None,
                "aligned_measured_path": None,
                "measured_pssar": 1.0,
                "reference_pssar": 1.1,
                "scaling_error": 0.1,
                "dose_to_agreement": 5.0,
                "distance_to_agreement": 2.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                DatabaseSampleColumn.ANTENNA_TYPE.value: "dipole",
                DatabaseSampleColumn.FREQUENCY.value: 900.0,
                DatabaseSampleColumn.DISTANCE.value: 15.0,
                DatabaseSampleColumn.MASS.value: 1.0,
                DatabaseSampleColumn.FILE_PATH.value: "/tmp/reference.csv",
            }
        ]
    ).to_csv(paths.legacy_filtered_db_csv_path, index=False)

    state = load_or_migrate_ui_state(paths)

    assert state is not None
    assert state.measured_file_name == "measured.csv"
    assert state.power_level == 17.5
    assert state.last_result is not None
    assert state.active_filters.antenna_type is not None
    assert state.active_filters.antenna_type.value == "dipole"
    assert paths.ui_state_path.exists()


def test_notebook_bootstrap_is_thin_and_v2_only() -> None:
    notebook = json.loads(Path("notebooks/voila.ipynb").read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "root_validator" not in sources
    assert "validator(" not in sources
    assert ".dict(" not in sources
    assert "class Config:" not in sources
    assert "bootstrap_voila_ui" in sources


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
