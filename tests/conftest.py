import os
from pathlib import Path

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
