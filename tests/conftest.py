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


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("slow") and _should_skip_slow(item.config):
        pytest.skip(
            "slow test skipped by default for bulk runs; select it directly or pass --run-slow"
        )
    if item.get_closest_marker("e2e") and not bool(item.config.getoption("--run-e2e")):
        pytest.skip(
            "e2e test skipped by default; pass --run-e2e and run with -p no:xdist"
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


# ---------------------------------------------------------------------------
# Shared fixtures for voila frontend tests
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


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
                body = resp.read().decode("utf-8", errors="replace")
            if "voila" in body.lower() or "jupyter" in body.lower():
                ready = True
                break
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)

    if not ready:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("Timed out waiting for Voila server to start")

    yield base_url, workspace_root

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
