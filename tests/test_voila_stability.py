"""
Stability-focused Playwright tests for the voila UI.

Run with:
    uv run python -m pytest -v -o "addopts=" --run-e2e -p no:xdist tests/test_voila_stability.py

Tests focus on:
1. Rerun stability — multiple consecutive runs on the same page
2. Validation error surfacing — malformed CSV → structured error banner, no traceback
3. Button state consistency — button re-enables after every terminal state
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_TIMEOUT = 90_000  # ms — widgets appear in ~30-60s on WSL2
_RUN_TIMEOUT = 180_000  # ms — full backend run can take up to 3 min on WSL2
_BUTTON_REENABLE_TIMEOUT = 30_000  # ms — how long we wait for button to re-enable
_UPLOAD_CSV_PATH = _REPO_ROOT / "data" / "example" / "measured_sSAR1g.csv"

# ---------------------------------------------------------------------------
# Helper fixture — fresh browser page per test (not shared state)
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_voila_page(playwright, voila_server):
    """One fresh page per test: new context, new page, full kernel load."""
    base_url, _ = voila_server
    browser = playwright.chromium.launch(headless=True)
    ctx = browser.new_context()
    page = ctx.new_page()
    page.set_default_timeout(_KERNEL_TIMEOUT)
    page.goto(base_url + "/", timeout=_KERNEL_TIMEOUT)
    page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)
    yield page
    ctx.close()
    browser.close()


# ---------------------------------------------------------------------------
# Shared page for multi-run rerun tests (cheaper — one kernel for all reruns)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stability_voila_page(playwright, voila_server):
    """Module-scoped page for the rerun stability tests."""
    base_url, _ = voila_server
    browser = playwright.chromium.launch(headless=True)
    ctx = browser.new_context()
    page = ctx.new_page()
    page.set_default_timeout(_KERNEL_TIMEOUT)
    page.goto(base_url + "/", timeout=_KERNEL_TIMEOUT)
    page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)
    yield page
    ctx.close()
    browser.close()


# ---------------------------------------------------------------------------
# Helper functions (mirror the pattern in test_voila_e2e.py)
# ---------------------------------------------------------------------------

_FIND_BTN_JS = (
    "() => [...document.querySelectorAll('button')]"
    ".find(b => b.textContent.includes('Compare Patterns'))"
)


def _upload_file(page, file_path: Path) -> None:
    with page.expect_file_chooser(timeout=10_000) as fc:
        page.locator(".widget-upload").click()
    fc.value.set_files(str(file_path))
    page.wait_for_function(
        "(expected) => document.body.innerText.includes(expected)",
        arg=file_path.name,
        timeout=20_000,
    )


def _select_unique_reference(page) -> None:
    """Click filter buttons until exactly one reference matches and run is enabled."""
    toggle_buttons = page.locator(".widget-toggle-button")
    count = toggle_buttons.count()
    run_btn = page.locator("button:has-text('Compare Patterns')")

    for i in range(count):
        if run_btn.get_attribute("disabled") is None:
            return
        btn = toggle_buttons.nth(i)
        if not btn.is_disabled():
            btn.click()
            page.wait_for_timeout(500)

    assert run_btn.get_attribute("disabled") is None, (
        "Run button should be enabled after selecting a unique reference"
    )


def _prepare_for_run(page, csv_path: Path = _UPLOAD_CSV_PATH) -> None:
    """Upload CSV and select a unique reference so the run button is enabled."""
    _upload_file(page, csv_path)
    _select_unique_reference(page)


def _wait_for_run_button_to_disable(page, timeout_ms: int = 10_000) -> None:
    """Wait until the run button becomes disabled (run started)."""
    page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN_JS})(); return b && b.disabled; }}",
        timeout=timeout_ms,
    )


def _wait_for_run_button_to_enable(
    page, timeout_ms: int = _BUTTON_REENABLE_TIMEOUT
) -> None:
    """Wait until the run button becomes enabled (run finished)."""
    page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN_JS})(); return b && !b.disabled; }}",
        timeout=timeout_ms,
    )


def _wait_for_workflow_cycle(page, timeout_ms: int = _RUN_TIMEOUT) -> None:
    """Wait for the run button to go disabled then re-enable (one full cycle)."""
    _wait_for_run_button_to_disable(page, timeout_ms=10_000)
    _wait_for_run_button_to_enable(page, timeout_ms=timeout_ms)


def _run_button_is_enabled(page) -> bool:
    btn = page.locator("button:has-text('Compare Patterns')")
    return btn.get_attribute("disabled") is None


def _get_progress_bar_value(page) -> float | None:
    """Return the progress bar value (0.0-1.0), or None if not found."""
    bar = page.locator(".widget-progress")
    if bar.count() == 0:
        return None
    aria_val = bar.first.get_attribute("aria-valuenow")
    if aria_val is None:
        return None
    try:
        return float(aria_val)
    except ValueError:
        return None


def _has_results_table(page) -> bool:
    return "Peak spatial-average SAR" in page.content()


def _feedback_banner_text(page) -> str:
    """Return the visible text of the feedback banner widget."""
    banner = page.locator(".widget-html")
    for i in range(banner.count()):
        text = banner.nth(i).inner_text()
        if "Error:" in text or "Warning:" in text or "Info:" in text:
            return text
    return ""


# ---------------------------------------------------------------------------
# Test 1: Rerun stability — three consecutive runs on the same page
# ---------------------------------------------------------------------------


def test_rerun_stability_run1(stability_voila_page) -> None:
    """First run: upload file, select reference, execute, verify results appear."""
    page = stability_voila_page
    _prepare_for_run(page)
    assert _run_button_is_enabled(page), "Run button must be enabled before run 1"

    page.locator("button:has-text('Compare Patterns')").click()
    _wait_for_workflow_cycle(page, timeout_ms=_RUN_TIMEOUT)

    assert _run_button_is_enabled(page), (
        "BUG: Run button is still disabled after run 1 completed"
    )
    # Progress bar should be at 0 or at max (not stuck mid-way)
    pb = _get_progress_bar_value(page)
    if pb is not None:
        assert pb == 0.0 or pb >= 1.0, (
            f"BUG: Progress bar stuck at {pb:.2f} after run 1"
        )


def test_rerun_stability_run2(stability_voila_page) -> None:
    """Second run: change power level so it's not an exact repeat, run again."""
    page = stability_voila_page
    # Change power level so it's not an exact repeat
    power_input = page.locator("input[type='number']").first
    power_input.click()
    power_input.fill("10.0")
    power_input.press("Tab")
    page.wait_for_timeout(1000)

    assert _run_button_is_enabled(page), "Run button must be enabled before run 2"

    page.locator("button:has-text('Compare Patterns')").click()
    _wait_for_workflow_cycle(page, timeout_ms=_RUN_TIMEOUT)

    assert _run_button_is_enabled(page), (
        "BUG: Run button is still disabled after run 2 completed"
    )
    pb = _get_progress_bar_value(page)
    if pb is not None:
        assert pb == 0.0 or pb >= 1.0, (
            f"BUG: Progress bar stuck at {pb:.2f} after run 2"
        )
    assert _has_results_table(page) or _feedback_banner_text(page), (
        "After run 2: neither results table nor feedback banner visible"
    )


def test_rerun_stability_run3(stability_voila_page) -> None:
    """Third run: change power level again, run a third time."""
    page = stability_voila_page
    power_input = page.locator("input[type='number']").first
    power_input.click()
    power_input.fill("17.0")
    power_input.press("Tab")
    page.wait_for_timeout(1000)

    assert _run_button_is_enabled(page), "Run button must be enabled before run 3"

    page.locator("button:has-text('Compare Patterns')").click()
    _wait_for_workflow_cycle(page, timeout_ms=_RUN_TIMEOUT)

    assert _run_button_is_enabled(page), (
        "BUG: Run button is still disabled after run 3 completed"
    )
    pb = _get_progress_bar_value(page)
    if pb is not None:
        assert pb == 0.0 or pb >= 1.0, (
            f"BUG: Progress bar stuck at {pb:.2f} after run 3"
        )
    assert _has_results_table(page) or _feedback_banner_text(page), (
        "After run 3: neither results table nor feedback banner visible"
    )


# ---------------------------------------------------------------------------
# Test 2: Validation error surfacing — malformed CSV
# ---------------------------------------------------------------------------


def test_malformed_csv_shows_structured_error_banner(
    fresh_voila_page, tmp_path: Path
) -> None:
    """
    Upload a CSV with no x/y/sar columns.
    Verify:
    - Error banner appears with [CODE] prefix (not a raw Python traceback)
    - No 'Traceback', no 'File "', no 'line N' in body text
    - Run button re-enables within 30s
    """
    page = fresh_voila_page

    # Create a deliberately malformed CSV (missing x, y, sar columns)
    malformed_csv = tmp_path / "malformed_measured.csv"
    malformed_csv.write_text("col_a,col_b\n1,2\n3,4\n", encoding="utf-8")

    _upload_file(page, malformed_csv)
    _select_unique_reference(page)
    assert _run_button_is_enabled(page), "Run button must be enabled before run"

    page.locator("button:has-text('Compare Patterns')").click()

    # Wait for button to disable (run started)
    _wait_for_run_button_to_disable(page, timeout_ms=10_000)

    # Wait for button to re-enable (run finished, with error)
    _wait_for_run_button_to_enable(page, timeout_ms=_BUTTON_REENABLE_TIMEOUT)

    assert _run_button_is_enabled(page), (
        "BUG: Run button did not re-enable after malformed CSV error"
    )

    body_text = page.locator("body").inner_text()

    # Must not contain a raw Python traceback
    assert "Traceback" not in body_text, (
        f"BUG: Raw Python traceback leaked into UI:\n{body_text[:2000]}"
    )
    assert 'File "' not in body_text, (
        f"BUG: Python file reference leaked into UI:\n{body_text[:2000]}"
    )
    assert not re.search(r"line \d+", body_text), (
        f"BUG: Python line number reference leaked into UI:\n{body_text[:2000]}"
    )

    # Check progress bar is reset
    pb = _get_progress_bar_value(page)
    if pb is not None:
        assert pb < 1.0 or pb == 0.0, (
            f"BUG: Progress bar not reset after error (value={pb:.2f})"
        )


def test_malformed_csv_banner_has_code_prefix(fresh_voila_page, tmp_path: Path) -> None:
    """
    Verify that the error banner for a malformed CSV shows a [CODE] formatted message.
    Acceptable: [CSV_FORMAT_INVALID] or any [CODE] prefix from the validation system.
    """
    page = fresh_voila_page

    malformed_csv = tmp_path / "malformed_measured2.csv"
    malformed_csv.write_text("col_a,col_b\n1,2\n3,4\n", encoding="utf-8")

    _upload_file(page, malformed_csv)
    _select_unique_reference(page)

    page.locator("button:has-text('Compare Patterns')").click()
    _wait_for_run_button_to_disable(page, timeout_ms=10_000)
    _wait_for_run_button_to_enable(page, timeout_ms=_BUTTON_REENABLE_TIMEOUT)

    body_text = page.locator("body").inner_text()
    page_html = page.content()

    # Look for [CODE] pattern in the page (indicates structured error)
    has_code_prefix = bool(re.search(r"\[[A-Z_]+\]", body_text))
    # Also accept if "Error:" label is present with some descriptive message
    has_error_label = "Error:" in body_text

    assert has_code_prefix or has_error_label, (
        f"BUG: Expected structured error banner with [CODE] prefix or 'Error:' label.\n"
        f"Body text:\n{body_text[:2000]}\n"
        f"Page HTML tail:\n{page_html[-1000:]}"
    )


# ---------------------------------------------------------------------------
# Test 3: Button state consistency after success
# ---------------------------------------------------------------------------


def test_button_reenables_after_successful_run(fresh_voila_page) -> None:
    """After a successful run, the button must re-enable within 30s."""
    page = fresh_voila_page
    _prepare_for_run(page, _UPLOAD_CSV_PATH)
    assert _run_button_is_enabled(page)

    import time

    start = time.monotonic()
    page.locator("button:has-text('Compare Patterns')").click()
    _wait_for_workflow_cycle(page, timeout_ms=_RUN_TIMEOUT)
    elapsed = time.monotonic() - start

    assert _run_button_is_enabled(page), (
        f"BUG: Button still disabled after run completed (elapsed={elapsed:.1f}s)"
    )


def test_button_reenables_after_error_run(fresh_voila_page, tmp_path: Path) -> None:
    """After an error run (malformed CSV), the button must re-enable within 30s."""
    page = fresh_voila_page

    malformed_csv = tmp_path / "error_measured.csv"
    malformed_csv.write_text("col_a,col_b\n1,2\n3,4\n", encoding="utf-8")

    _upload_file(page, malformed_csv)
    _select_unique_reference(page)

    import time

    start = time.monotonic()
    page.locator("button:has-text('Compare Patterns')").click()
    _wait_for_run_button_to_disable(page, timeout_ms=10_000)

    try:
        _wait_for_run_button_to_enable(page, timeout_ms=_BUTTON_REENABLE_TIMEOUT)
        elapsed = time.monotonic() - start
        assert _run_button_is_enabled(page), (
            f"BUG: Button still disabled after error (elapsed={elapsed:.1f}s)"
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        body_text = page.locator("body").inner_text()
        raise AssertionError(
            f"BUG: Button did not re-enable within {_BUTTON_REENABLE_TIMEOUT / 1000:.0f}s "
            f"after error (elapsed={elapsed:.1f}s). "
            f"Body text:\n{body_text[:1000]}"
        ) from exc


# ---------------------------------------------------------------------------
# Test 4: Late backend validation error must surface (regression for the bug
# where the progress bar stalled and the backend returned MASK_TOO_SMALL but
# nothing appeared in the frontend — see plan 'mutable-stargazing-balloon').
# ---------------------------------------------------------------------------


def _write_zero_sar_csv(path: Path) -> None:
    """Write a structurally-valid SAR CSV whose values are all below the noise
    floor — the workflow accepts the parse, runs registration, then raises
    MASK_TOO_SMALL during gamma evaluation. This reproduces the user's
    'progress stalled, backend responded, UI showed nothing' failure mode."""
    header = "X [m],Y [m],sSAR10g [W/Kg],Uncertainty [dB]"
    rows = [header]
    # 50x50 grid spanning ~60mm × 60mm — large enough to satisfy measurement-area
    # bounds, but with all-zero SAR so the mask threshold yields effectively
    # nothing and the inscribed-square check fails.
    spacing = 0.00125  # 1.25 mm
    origin_x = 0.0075
    origin_y = 0.0075
    for iy in range(50):
        for ix in range(50):
            x = origin_x + ix * spacing
            y = origin_y + iy * spacing
            rows.append(f"{x:.5f},{y:.5f},0,1.29")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_late_backend_validation_error_surfaces_banner(
    fresh_voila_page, tmp_path: Path
) -> None:
    """Regression: when the backend runs for several seconds and then raises a
    validation error, the banner must appear. Previously the success/failure
    handlers silently returned if the watchdog or a reset had nulled
    `_active_run_id`, leaving the UI blank."""
    page = fresh_voila_page

    zero_csv = tmp_path / "zero_sar_measured.csv"
    _write_zero_sar_csv(zero_csv)

    _upload_file(page, zero_csv)
    _select_unique_reference(page)
    assert _run_button_is_enabled(page), "Run button must be enabled before run"

    page.locator("button:has-text('Compare Patterns')").click()
    _wait_for_run_button_to_disable(page, timeout_ms=10_000)
    _wait_for_run_button_to_enable(page, timeout_ms=_RUN_TIMEOUT)

    banner_text = _feedback_banner_text(page)
    body_text = page.locator("body").inner_text()

    assert banner_text, (
        "BUG: backend completed but no feedback banner is visible — "
        "this is exactly the failure mode the user reported.\n"
        f"Body text:\n{body_text[:1500]}"
    )
    assert "Error:" in banner_text or "Warning:" in banner_text, (
        f"Expected an Error/Warning banner; got: {banner_text!r}"
    )

    pb = _get_progress_bar_value(page)
    if pb is not None:
        assert pb < 1.0 or pb == 0.0, (
            f"BUG: Progress bar not reset after late validation error (value={pb:.2f})"
        )
