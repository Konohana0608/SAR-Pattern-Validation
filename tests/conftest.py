import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from .helpers import gaussian_2d, make_rect_grid, write_sar_csv

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SAVE_MEASUREMENT_VALIDATION_PLOTS", "0")
os.environ.setdefault("SAVE_TUTORIAL_VALIDATION_PLOTS", "0")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow during broad test runs.",
    )
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end Playwright tests (requires playwright browsers installed).",
    )


def _is_explicit_selection(config: pytest.Config) -> bool:
    # VS Code "Run All" and common bulk invocations target a top-level directory.
    # Keep those fast by default, but allow explicit file/test runs to include slow tests.
    args = [str(arg) for arg in config.args]
    if not args:
        return False

    for arg in args:
        if "::" in arg:
            # Test node ID or specific test function
            return True
        if arg.endswith(".py"):
            # Specific file
            return True
        # VS Code may pass test IDs in other formats
        if "test_measurement_workflow_cases_" in arg:
            return True

    return False


def _should_skip_slow(config: pytest.Config) -> bool:
    if bool(config.getoption("--run-slow")):
        return False
    if _is_explicit_selection(config):
        return False

    markexpr = str(getattr(config.option, "markexpr", "") or "").strip()
    return not (markexpr and "slow" in markexpr and "not slow" not in markexpr)


def _should_skip_e2e(config: pytest.Config) -> bool:
    if bool(config.getoption("--run-e2e")):
        return False
    else:
        skip_e2e = not _is_explicit_selection(config)
        return skip_e2e


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("slow") and _should_skip_slow(item.config):
        pytest.skip(
            "slow test skipped by default for bulk runs; select it directly or pass --run-slow"
        )
    if item.get_closest_marker("e2e") and _should_skip_e2e(item.config):
        pytest.skip(
            "e2e test skipped by default for bulk runs; select it directly or pass --run-e2e and run with -p no:xdist"
        )


@pytest.fixture(autouse=True)
def disable_interactive_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    def fail_show(*args, **kwargs) -> None:
        raise AssertionError("Interactive matplotlib windows are disabled in tests")

    monkeypatch.setattr(plt, "show", fail_show)


@pytest.fixture
def tmp_csv_pair(tmp_path: Path):
    """
    Two small, identical rectangular-lattice CSVs.

    Convention in the pipeline:
      - measured CSV -> fixed image (target space)
      - reference CSV -> moving image (warped onto measured)
    """
    x, y = make_rect_grid(step=0.006)
    _, _, Z = gaussian_2d(x, y, x0=0.01, y0=-0.01, sx=0.02, sy=0.03, peak=1.0)

    measured = tmp_path / "measured.csv"
    reference = tmp_path / "reference.csv"

    write_sar_csv(measured, x, y, Z)
    write_sar_csv(reference, x, y, Z)

    return str(measured), str(reference)


_DEMO_RUN_DIR = Path(__file__).resolve().parent / "fixtures" / "demo_run"
_DEMO_REFERENCE_FILE = "dipole_1450MHz_Flat_10mm_10g.csv"


@pytest.fixture
def workspace_with_database(tmp_path: Path):
    """Workspace whose project_root symlinks to the repo, giving access to real DB."""
    from sar_pattern_validation.voila_frontend.runtime import WorkspacePaths

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "sar-pattern-validation").symlink_to(_REPO_ROOT)
    paths = WorkspacePaths.from_workspace(workspace_root)
    paths.ensure_runtime_dirs()
    return paths


@pytest.fixture
def completed_workspace(tmp_path: Path):
    """Workspace pre-populated with a real completed run (images + result payload)."""
    import json

    from sar_pattern_validation.voila_frontend.models import WorkflowResultPayload
    from sar_pattern_validation.voila_frontend.runtime import WorkspacePaths

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "sar-pattern-validation").symlink_to(_REPO_ROOT)
    paths = WorkspacePaths.from_workspace(workspace_root)
    paths.ensure_runtime_dirs()

    shutil.copytree(_DEMO_RUN_DIR / "images", paths.images_dir, dirs_exist_ok=True)
    shutil.copy2(
        _DEMO_RUN_DIR / "uploaded_data" / "measured_data.csv",
        paths.measured_file_path,
    )

    raw = json.loads((_DEMO_RUN_DIR / "result.json").read_text())
    raw.pop("reference_file", None)
    raw.pop("measured_sha256", None)
    for path_field in (
        "gamma_image_path",
        "failure_image_path",
        "registered_overlay_path",
        "reference_image_path",
        "measured_image_path",
        "aligned_measured_path",
    ):
        raw[path_field] = None

    import hashlib

    reference_path = paths.project_root / "data" / "database" / _DEMO_REFERENCE_FILE
    sha = hashlib.sha256(paths.measured_file_path.read_bytes()).hexdigest()
    payload = WorkflowResultPayload.model_validate(raw)
    payload = payload.model_copy(
        update={
            "reference_file_path": str(reference_path),
            "measured_file_sha256": sha,
        }
    )
    return paths, payload


@pytest.fixture
def sar_ui(workspace_with_database):
    """SarGammaComparisonUI with display + prerequisites mocked out."""
    from sar_pattern_validation.voila_frontend.ui import SarGammaComparisonUI

    with (
        patch("sar_pattern_validation.voila_frontend.ui.ensure_notebook_prerequisites"),
        patch("sar_pattern_validation.voila_frontend.ui.display"),
    ):
        yield SarGammaComparisonUI(paths=workspace_with_database)


# ---------------------------------------------------------------------------
# Negative-path fixtures for Task 6.6 (one fixture per failure mode)
#
# Each fixture is paired with the expected ValidationIssue tuple
#     (code, severity, message_substring)
# via the `_EXPECTED_*` module-level dicts below, so tests can assert against
# a single source of truth.
# ---------------------------------------------------------------------------

EXPECTED_CSV_FORMAT_INVALID: tuple[str, str, str] = (
    "CSV_FORMAT_INVALID",
    "error",
    "CSV format invalid",
)
EXPECTED_MEASUREMENT_AREA_OUT_OF_BOUNDS: tuple[str, str, str] = (
    "MEASUREMENT_AREA_OUT_OF_BOUNDS",
    "error",
    "Measurement area is out of bounds",
)
EXPECTED_NOISE_FLOOR_OUT_OF_BOUNDS: tuple[str, str, str] = (
    "NOISE_FLOOR_OUT_OF_BOUNDS",
    "error",
    "Noise floor is out of bounds",
)
EXPECTED_MASK_TOO_SMALL: tuple[str, str, str] = (
    "MASK_TOO_SMALL",
    "error",
    "Gamma evaluation mask is too small",
)


@pytest.fixture
def malformed_csv_path(tmp_path: Path) -> tuple[Path, tuple[str, str, str]]:
    """A CSV with no recognisable x/y coordinate columns.

    Returns ``(path, expected)`` where ``expected`` is the
    ``(code, severity, message_substring)`` tuple associated with the
    failure mode this fixture provokes.
    """
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text("col_a,col_b,col_c\n1,2,3\n4,5,6\n", encoding="utf-8")
    return csv_path, EXPECTED_CSV_FORMAT_INVALID


@pytest.fixture
def valid_csv_path(tmp_path: Path) -> Path:
    """A small valid CSV with x, y, sar columns."""
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text(
        "x,y,sar\n0.0,0.0,1.0\n0.001,0.0,1.2\n0.0,0.001,0.9\n0.001,0.001,1.1\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def out_of_bounds_measurement_area() -> tuple[dict[str, float], tuple[str, str, str]]:
    """Measurement area kwargs that violate the bounds policy.

    Returns ``(kwargs, expected)`` where ``expected`` is the
    ``(code, severity, message_substring)`` tuple.
    """
    return (
        {"measurement_area_x_mm": 700.0, "measurement_area_y_mm": 200.0},
        EXPECTED_MEASUREMENT_AREA_OUT_OF_BOUNDS,
    )


@pytest.fixture
def out_of_bounds_noise_floor() -> tuple[dict[str, float], tuple[str, str, str]]:
    """Noise-floor kwargs that violate the bounds policy.

    Returns ``(kwargs, expected)`` where ``expected`` is the
    ``(code, severity, message_substring)`` tuple.
    """
    return (
        {"noise_floor_wkg": -0.1},
        EXPECTED_NOISE_FLOOR_OUT_OF_BOUNDS,
    )


@pytest.fixture
def mask_too_small_workflow_config(
    tmp_path: Path,
) -> tuple[dict[str, object], tuple[str, str, str]]:
    """Build kwargs for a workflow whose effective evaluation mask is < 22 mm.

    Writes a tiny synthetic measured/reference CSV pair (10 mm × 10 mm),
    well below the 22 mm 10 g cube-face threshold. The kwargs are returned
    as a ``dict`` rather than a fully-validated ``WorkflowConfig`` so that
    callers can splat them directly into ``complete_workflow(...)``.

    Returns ``(kwargs, expected)`` where ``expected`` is the
    ``(code, severity, message_substring)`` tuple.
    """
    # A tiny 10x10 mm grid -> well below 22 mm, but still big enough that
    # earlier pipeline stages (registration, etc.) succeed.
    x, y = make_rect_grid(xmin=-0.005, xmax=0.005, ymin=-0.005, ymax=0.005, step=0.001)
    _, _, Z = gaussian_2d(x, y, x0=0.0, y0=0.0, sx=0.002, sy=0.002, peak=1.0)
    measured = tmp_path / "tiny_measured.csv"
    reference = tmp_path / "tiny_reference.csv"
    write_sar_csv(measured, x, y, Z)
    write_sar_csv(reference, x, y, Z)

    kwargs: dict[str, object] = {
        "measured_file_path": str(measured),
        "reference_file_path": str(reference),
        "render_plots": False,
        "show_plot": False,
        "log_level": "WARNING",
    }
    return kwargs, EXPECTED_MASK_TOO_SMALL


# ---------------------------------------------------------------------------
# Shared fixtures for voila frontend tests
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEMO_RUN_DIR = Path(__file__).resolve().parent / "fixtures" / "demo_run"
_DEMO_REFERENCE_FILE = "dipole_1450MHz_Flat_10mm_10g.csv"


@pytest.fixture
def workspace_with_database(tmp_path: Path):
    """Workspace whose project_root symlinks to the repo, giving access to real DB."""
    from sar_pattern_validation.voila_frontend.runtime import WorkspacePaths

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "sar-pattern-validation").symlink_to(_REPO_ROOT)
    paths = WorkspacePaths.from_workspace(workspace_root)
    paths.ensure_runtime_dirs()
    return paths


@pytest.fixture
def completed_workspace(tmp_path: Path):
    """Workspace pre-populated with a real completed run (images + result payload).

    Copies pre-baked output images and measured CSV from tests/fixtures/demo_run/,
    then returns (WorkspacePaths, WorkflowResultPayload) ready for use without
    running the backend subprocess.  Completes in <1s.
    """
    import json
    import shutil

    from sar_pattern_validation.voila_frontend.models import WorkflowResultPayload
    from sar_pattern_validation.voila_frontend.runtime import WorkspacePaths

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "sar-pattern-validation").symlink_to(_REPO_ROOT)
    paths = WorkspacePaths.from_workspace(workspace_root)
    paths.ensure_runtime_dirs()

    shutil.copytree(_DEMO_RUN_DIR / "images", paths.images_dir, dirs_exist_ok=True)
    shutil.copy2(
        _DEMO_RUN_DIR / "uploaded_data" / "measured_data.csv",
        paths.measured_file_path,
    )

    raw = json.loads((_DEMO_RUN_DIR / "result.json").read_text())
    raw.pop("reference_file", None)
    raw.pop("measured_sha256", None)
    for path_field in (
        "gamma_image_path",
        "failure_image_path",
        "registered_overlay_path",
        "reference_image_path",
        "measured_image_path",
        "aligned_measured_path",
    ):
        raw[path_field] = None

    import hashlib

    reference_path = paths.project_root / "data" / "database" / _DEMO_REFERENCE_FILE
    sha = hashlib.sha256(paths.measured_file_path.read_bytes()).hexdigest()
    payload = WorkflowResultPayload.model_validate(raw)
    payload = payload.model_copy(
        update={
            "reference_file_path": str(reference_path),
            "measured_file_sha256": sha,
        }
    )
    return paths, payload


@pytest.fixture
def sar_ui(workspace_with_database):
    """SarGammaComparisonUI with display + prerequisites mocked out."""
    from sar_pattern_validation.voila_frontend.ui import SarGammaComparisonUI

    with (
        patch("sar_pattern_validation.voila_frontend.ui.ensure_notebook_prerequisites"),
        patch("sar_pattern_validation.voila_frontend.ui.display"),
    ):
        yield SarGammaComparisonUI(paths=workspace_with_database)


# ---------------------------------------------------------------------------
# Voila server fixture for Playwright e2e tests
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def voila_server(tmp_path_factory: pytest.TempPathFactory):
    """Starts a real voila server for e2e tests. Yields the render URL."""
    notebook_source = _REPO_ROOT / "notebooks" / "voila.ipynb"
    port = _free_port()
    workspace_root = tmp_path_factory.mktemp("voila-e2e-workspace")
    (workspace_root / "sar-pattern-validation").symlink_to(_REPO_ROOT)
    shutil.copy2(notebook_source, workspace_root / "voila.ipynb")

    env = os.environ.copy()
    env["SAR_PATTERN_VALIDATION_BACKEND_MODE"] = "local"
    env["SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE"] = str(_REPO_ROOT)
    env["SAR_PATTERN_VALIDATION_RUN_STALL_TIMEOUT_S"] = "60"
    env["MPLBACKEND"] = "Agg"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "voila",
            "voila.ipynb",
            "--no-browser",
            f"--port={port}",
            "--Voila.ip=127.0.0.1",
        ],
        cwd=workspace_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # In voila 0.5 single-notebook mode the notebook is served at root, not /voila/render/...
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"Voila exited early (code {proc.returncode}):\n{out}")
        try:
            with urllib.request.urlopen(base_url + "/", timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        pytest.fail("Timed out waiting for Voila server to start")

    yield base_url, workspace_root

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
