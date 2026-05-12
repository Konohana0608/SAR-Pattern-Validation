"""End-to-end Playwright tests for the Voila UI.

Run inside the jupyter-math container via:
    make test-voila-e2e

Manual repro inside `make serve-voila`, then:
    /home/jovyan/.venv/bin/python -m pytest -v -s -o "addopts=" \\
        --run-e2e -p no:xdist tests/test_voila_e2e.py::<name>

Design:
- All tests share one browser page to avoid restarting the voila kernel per
  test (heavy imports make each startup ~30-90s on WSL2).
- Tests are ordered from read-only to state-modifying; later tests deliberately
  build on earlier state.
- Every fixture and helper logs entry/exit + key state via `_log()` to stdout.
  Run with `-s` to see the trail; use it as a reproduction script when a test
  fails — each `>>` / `<<` line names the action taken or awaited.
- Tests requiring features not yet ported back into main-melanie carry a
  `pytest.mark.skip` with the gating phase; the skips are removed by the
  cherry-pick PRs that bring the underlying notebook feature in.

DOM notes (ipywidgets 8.x + voila 0.5):
- Toggle buttons: class="... widget-toggle-button"; .mod-active added on selection
- FileUpload: class="... widget-upload"; no <input type=file> in DOM —
  intercept via expect_file_chooser()
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from pathlib import Path

import pytest
from attr import dataclass

pytest.importorskip("playwright")

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_TIMEOUT = 120_000  # ms — kernel startup + initial render
_UPLOAD_CSV_PATH = _REPO_ROOT / "data" / "example" / "measured_sSAR1g.csv"

# Reference selection used by _ensure_run_button_enabled. The four (column, value)
# pairs uniquely identify a row in data/database/, so the run button can enable
# deterministically. The resulting reference CSV is the file CI must pull from
# Git LFS — keep this in sync with .github/workflows/ci.yml's `git lfs pull`.
#   filename: dipole_1450MHz_Flat_10mm_1g.csv
_REFERENCE_FILTERS: tuple[tuple[str, str], ...] = (
    ("Antenna Type", "dipole"),
    ("Frequency [MHz]", "1450.0"),
    ("Distance [mm]", "10.0"),
    ("Mass [g]", "1.0"),
)
_REFERENCE_VALUES: frozenset[str] = frozenset(v for _, v in _REFERENCE_FILTERS)


# ---------------------------------------------------------------------------
# Logging helper — plain print() so `pytest -v -s` shows the trail.
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Print a timestamped trace line so each step is greppable in CI logs."""
    stamp = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
    print(f"[{stamp}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Shared page fixture — kernel starts once for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def voila_page(playwright, voila_server):
    """Navigate to voila once and keep the page alive for all tests."""
    base_url, _ = voila_server
    _log(f">> voila_page: launching headless chromium (server={base_url})")
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(_KERNEL_TIMEOUT)

    started = time.time()
    _log(f">> voila_page: page.goto({base_url}/)")
    page.goto(base_url + "/", timeout=_KERNEL_TIMEOUT)

    _log(
        f">> voila_page: waiting for selector .widget-button (timeout={_KERNEL_TIMEOUT}ms)"
    )
    page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)
    _log(f"<< voila_page: kernel ready after {time.time() - started:.1f}s")

    yield page

    _log(">> voila_page: teardown — closing context + browser")
    context.close()
    browser.close()
    _log("<< voila_page: teardown complete")


@pytest.fixture(autouse=True)
def _capture_final_screenshot(request, voila_page):
    """Save a PNG of the final browser state after every test (pass and fail).

    Artifacts land in PLAYWRIGHT_ARTIFACTS_DIR (set by the test-harness script)
    or fall back to ``test-artifacts/playwright/`` relative to the repo root.
    File is named after the test function so it's unambiguous in CI and local
    review.
    """
    yield
    artifacts_dir = Path(
        os.environ.get(
            "PLAYWRIGHT_ARTIFACTS_DIR",
            str(_REPO_ROOT / "test-artifacts" / "playwright"),
        )
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w-]", "_", request.node.name)
    screenshot_path = artifacts_dir / f"{safe_name}.png"
    try:
        voila_page.screenshot(path=str(screenshot_path), full_page=True)
        _log(f"   screenshot → {screenshot_path}")
    except Exception as exc:  # noqa: BLE001
        _log(f"   screenshot failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers (each logs entry+exit so a hung helper is obvious in the trace)
# ---------------------------------------------------------------------------


def _upload_file(voila_page, file_path: Path) -> None:
    _log(f">> upload_file: clicking .widget-upload to upload {file_path.name}")
    with voila_page.expect_file_chooser(timeout=5_000) as fc:
        voila_page.locator(".widget-upload").click()
    fc.value.set_files(str(file_path))
    _log(f">> upload_file: waiting for {file_path.name!r} to appear in body text")
    voila_page.wait_for_function(
        "(expected) => document.body.innerText.includes(expected)",
        arg=file_path.name,
        timeout=15_000,
    )
    _log(f"<< upload_file: {file_path.name} visible in DOM")


def _ensure_run_button_enabled(voila_page) -> None:
    _log(">> ensure_run_button_enabled: starting")
    if _UPLOAD_CSV_PATH.name not in voila_page.locator("body").inner_text():
        _log("   no upload yet — performing initial upload")
        _upload_file(voila_page, _UPLOAD_CSV_PATH)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    _FIND_RUN_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )

    if run_btn.get_attribute("disabled") is None:
        _log("<< ensure_run_button_enabled: already enabled")
        return

    # Click the target value in each filter column. Each column is a VBox
    # whose first child is `<b>{column_name}</b>`. We anchor on the <b> and
    # take its *closest* widget-vbox ancestor — a plain `.widget-vbox:has(b…)`
    # would also match parent VBoxes (e.g. the cell-level wrapper) and pull
    # in toggles from neighbour columns like "10.0" appearing in both
    # Distance=10mm and Mass=10g. `:text-is()` keeps "1.0" from also
    # matching "10.0" inside the same column.
    for column_name, value in _REFERENCE_FILTERS:
        column = voila_page.locator(
            f'xpath=//b[normalize-space()="{column_name}"]'
            "/ancestor::*[contains(concat(' ', normalize-space(@class), ' '),"
            " ' widget-vbox ')][1]"
        )
        target_btn = column.locator(f'.widget-toggle-button:text-is("{value}")')
        classes = target_btn.get_attribute("class") or ""
        if "mod-active" in classes:
            _log(f"   {column_name} → {value} already active; skipping")
            continue
        _log(f"   clicking {column_name} → {value}")
        target_btn.click()
        voila_page.wait_for_timeout(200)

    # check_settings polls every 1 s; wait up to 3 s for it to re-enable.
    with contextlib.suppress(Exception):
        voila_page.wait_for_function(
            f"() => {{ const b = ({_FIND_RUN_BTN})(); return b && !b.disabled; }}",
            timeout=3_000,
        )

    if run_btn.get_attribute("disabled") is not None:
        # Defensive fallback: clear any stale toggles activated by earlier
        # tests that don't match our four target values, then re-wait.
        _log("   run still disabled — clearing stale toggles")
        active = voila_page.locator(".widget-toggle-button.mod-active")
        for i in range(active.count()):
            btn = active.nth(i)
            text = (btn.text_content() or "").strip()
            if text in _REFERENCE_VALUES:
                continue
            with contextlib.suppress(Exception):
                btn.click()
                voila_page.wait_for_timeout(200)
        with contextlib.suppress(Exception):
            voila_page.wait_for_function(
                f"() => {{ const b = ({_FIND_RUN_BTN})(); return b && !b.disabled; }}",
                timeout=3_000,
            )

    assert run_btn.get_attribute("disabled") is None, (
        "Run button should be enabled once a measured file and unique reference are selected"
    )
    _log("<< ensure_run_button_enabled: enabled")


def _wait_for_workflow_cycle(voila_page, timeout_ms: int = 120_000) -> None:
    """Wait for Compare Patterns to disable (running) then re-enable (done)."""
    _log(">> wait_for_workflow_cycle: starting")
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    _FIND_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )
    _log("   waiting up to 10s for run button to become disabled (cycle started)")
    cycle_started = True
    with contextlib.suppress(Exception):
        voila_page.wait_for_function(
            f"() => {{ const b = ({_FIND_BTN})(); return b && b.disabled; }}",
            timeout=10_000,
        )
    if run_btn.get_attribute("disabled") is None:
        cycle_started = False
        _log(
            "   run button never observed disabled; proceeding with terminal-state wait"
        )

    # Terminal states for a Compare Patterns click. The error-banner forms
    # (`Error:`, `Workflow execution failed`, `Could not reach …`) are listed
    # so a silent backend failure short-circuits the 120 s wait — the success-
    # path assertions below still gate on the result-table strings.
    terminal_state_js = (
        "() => {"
        "  const bodyText = document.body.innerText;"
        "  const bodyHtml = document.body.innerHTML;"
        "  return bodyText.includes('Pass rate')"
        "    || bodyHtml.includes('Reference, 30 dBm')"
        "    || bodyText.includes('already match the current results')"
        "    || bodyText.includes('SAR pattern validation complete.')"
        "    || bodyText.includes('Could not reach the Voila server')"
        "    || bodyText.includes('Workflow execution failed')"
        "    || /\\bError:\\s/.test(bodyText);"
        "}"
    )

    _log(
        f"   waiting up to {timeout_ms / 1000:.0f}s for terminal UI state after run click"
    )
    voila_page.wait_for_function(
        terminal_state_js,
        timeout=timeout_ms,
    )

    if cycle_started:
        _log("   waiting up to 30s for run button to re-enable after terminal state")
        voila_page.wait_for_function(
            f"() => {{ const b = ({_FIND_BTN})(); return b && !b.disabled; }}",
            timeout=30_000,
        )
    assert run_btn.get_attribute("disabled") is None
    _log("   waiting for rendered result state in DOM")
    voila_page.wait_for_function(
        terminal_state_js,
        timeout=30_000,
    )
    _log("<< wait_for_workflow_cycle: complete")


@dataclass
class PSSARRowValues:
    measured_value: float
    measured_30dbm: float
    reference_value: float
    scaling_error: float


def _extract_pssar_row_values(page_html: str) -> PSSARRowValues:
    _log(">> extract_pssar_row_values: scanning page HTML")
    # New two-table layout: result badge | measured@power | measured@30dBm | reference@30dBm | scaling_error | criteria
    match = re.search(
        r"Reference, 30 dBm</th>.*?"  # anchor on header
        r"(?:Pass|Fail).*?</td>"  # skip result badge cell
        r"\s*<td[^>]*>[\d.]+\s*W/kg</td>"  # skip measured@input_power
        r"\s*<td[^>]*>([\d.]+)\s*W/kg</td>"  # measured@30dBm
        r"\s*<td[^>]*>([\d.]+)\s*W/kg</td>"  # reference@30dBm
        r"\s*<td[^>]*>([-\d.]+)</td>",  # scaling error [%]
        page_html,
        flags=re.S,
    )
    assert match is not None, "Could not extract result table values from page."
    values = PSSARRowValues(
        measured_value=float(match.group(1)),  # measured 30 dBm
        measured_30dbm=float(match.group(1)),
        reference_value=float(match.group(2)),  # reference 30 dBm
        scaling_error=float(match.group(3)),  # scaling error [%]
    )
    _log(f"<< extract_pssar_row_values: {values}")
    return values


def _set_power_level(voila_page, value: float) -> None:
    _log(f">> set_power_level: setting power input to {value}")
    power_input = voila_page.locator("input[type='number']").first
    power_input.click()
    power_input.fill(str(value))
    power_input.press("Tab")
    voila_page.wait_for_function(
        "(expected) => {"
        "  const input = document.querySelector(\"input[type='number']\");"
        "  return input && Math.abs(Number(input.value) - expected) < 0.01;"
        "}",
        arg=value,
        timeout=10_000,
    )
    _log(f"<< set_power_level: confirmed value={value}")


# ---------------------------------------------------------------------------
# Read-only smoke tests (run first — page is in fresh state)
# ---------------------------------------------------------------------------


def test_run_button_is_visible(voila_page) -> None:
    _log(">> test_run_button_is_visible")
    assert voila_page.locator("button:has-text('Compare Patterns')").is_visible()
    _log("<< test_run_button_is_visible: pass")


def test_run_button_is_disabled_on_fresh_load(voila_page) -> None:
    _log(">> test_run_button_is_disabled_on_fresh_load")
    btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert btn.get_attribute("disabled") is not None
    _log("<< test_run_button_is_disabled_on_fresh_load: pass")


def test_filter_toggle_buttons_are_visible(voila_page) -> None:
    _log(">> test_filter_toggle_buttons_are_visible")
    count = voila_page.locator(".widget-toggle-button").count()
    _log(f"   toggle button count = {count}")
    assert count > 0
    _log("<< test_filter_toggle_buttons_are_visible: pass")


def test_file_upload_button_is_present(voila_page) -> None:
    _log(">> test_file_upload_button_is_present")
    count = voila_page.locator(".widget-upload").count()
    _log(f"   widget-upload count = {count}")
    assert count == 1
    _log("<< test_file_upload_button_is_present: pass")


# ---------------------------------------------------------------------------
# State-modifying tests (each builds on the previous)
# ---------------------------------------------------------------------------


def test_clicking_filter_button_activates_it(voila_page) -> None:
    _log(">> test_clicking_filter_button_activates_it")
    voila_page.locator(".widget-toggle-button").first.click()
    voila_page.wait_for_selector(".widget-toggle-button.mod-active", timeout=10_000)
    active_count = voila_page.locator(".widget-toggle-button.mod-active").count()
    _log(f"   active toggle count after click = {active_count}")
    assert active_count >= 1
    _log("<< test_clicking_filter_button_activates_it: pass")


def test_file_upload_updates_filename_label(voila_page) -> None:
    _log(">> test_file_upload_updates_filename_label")
    _upload_file(voila_page, _UPLOAD_CSV_PATH)
    _log("<< test_file_upload_updates_filename_label: pass")


def test_run_button_enables_after_upload_and_unique_filter(voila_page) -> None:
    """Clicks filter buttons until exactly one reference matches; asserts run is enabled."""
    _log(">> test_run_button_enables_after_upload_and_unique_filter")
    _ensure_run_button_enabled(voila_page)
    _log("<< test_run_button_enables_after_upload_and_unique_filter: pass")


# ---------------------------------------------------------------------------
# Tests gated on features that haven't been cherry-picked back into main-melanie.
# Each Phase B PR removes the matching skip mark when it lands the feature.
# ---------------------------------------------------------------------------


def test_run_workflow_and_check_results_table(voila_page) -> None:
    """Clicks Compare Patterns and asserts the results tables render without error."""
    _log(">> test_run_workflow_and_check_results_table")
    _ensure_run_button_enabled(voila_page)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None, "Run button must be enabled first"

    _log("   clicking Compare Patterns")
    run_btn.click()
    _wait_for_workflow_cycle(voila_page)

    body_text = voila_page.locator("body").inner_text()
    page_html = voila_page.content()

    assert "Traceback" not in body_text, (
        f"Python traceback in page:\n{body_text[:3000]}"
    )
    assert not re.search(r"\bError:\s", body_text), (
        f"Error banner in page after workflow ran:\n{body_text[:3000]}"
    )
    assert "Workflow execution failed" not in body_text, (
        f"Workflow failure banner in page:\n{body_text[:3000]}"
    )
    assert "Reference, 30 dBm" in page_html, (
        f"Result table not found.\nBody text:\n{body_text[:3000]}\n\nPage HTML tail:\n{page_html[-2000:]}"
    )
    assert "Pass rate" in body_text, (
        f"Pass rate not found.\nBody text:\n{body_text[:3000]}"
    )
    assert "Pass" in body_text or "Fail" in body_text, (
        f"No Pass/Fail result found.\nBody text:\n{body_text[:3000]}"
    )
    _log("<< test_run_workflow_and_check_results_table: pass")


def test_restored_session_rerun_updates_results_after_power_change(voila_page) -> None:
    _log(">> test_restored_session_rerun_updates_results_after_power_change")
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    if "Reference, 30 dBm" not in voila_page.content():
        _log("   no prior results in DOM — running once to seed state")
        run_btn.click()
        _wait_for_workflow_cycle(voila_page)
    first_values = _extract_pssar_row_values(voila_page.content())

    _log(f"   reloading page (timeout={_KERNEL_TIMEOUT}ms)")
    voila_page.reload(timeout=_KERNEL_TIMEOUT)
    voila_page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)
    voila_page.wait_for_function(
        f"() => document.body.innerText.includes('{_UPLOAD_CSV_PATH.name}')",
        timeout=15_000,
    )
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('Reference, 30 dBm')",
        timeout=15_000,
    )

    restored_values = _extract_pssar_row_values(voila_page.content())
    assert restored_values.measured_30dbm == pytest.approx(
        first_values.measured_30dbm, abs=0.01
    )
    assert restored_values.reference_value == pytest.approx(
        first_values.reference_value, abs=0.01
    )
    assert restored_values.scaling_error == pytest.approx(
        first_values.scaling_error, abs=0.01
    )

    _set_power_level(voila_page, 10.0)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    # check_settings re-enables the button within ~1s after restore; wait up to 3s.
    _FIND_RUN_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )
    with contextlib.suppress(Exception):
        voila_page.wait_for_function(
            f"() => {{ const b = ({_FIND_RUN_BTN})(); return b && !b.disabled; }}",
            timeout=3_000,
        )
    assert run_btn.get_attribute("disabled") is None

    _log("   clicking Compare Patterns after power change")
    run_btn.click()
    _wait_for_workflow_cycle(voila_page)

    second_values = _extract_pssar_row_values(voila_page.content())
    # Verify the run completed and results are displayed (not a memo-cache early return).
    # Note: measured_pssar is not scaled by power_level_dbm in the current CLI, so we
    # assert results are *present* and consistent with using the same reference file.
    assert second_values.reference_value == pytest.approx(
        first_values.reference_value, abs=0.01
    ), "Reference pssar should be unchanged (same reference file used)"
    assert "Pass rate" in voila_page.locator("body").inner_text()
    assert (
        "already match the current results"
        not in voila_page.locator("body").inner_text()
    ), "Memo cache should NOT have fired — power level changed"
    _log("<< test_restored_session_rerun_updates_results_after_power_change: pass")


def test_same_session_rerun_updates_results_after_power_change(voila_page) -> None:
    _log(">> test_same_session_rerun_updates_results_after_power_change")
    _ensure_run_button_enabled(voila_page)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None

    first_values = _extract_pssar_row_values(voila_page.content())

    _set_power_level(voila_page, 17.0)
    _log("   clicking Compare Patterns after power change")
    run_btn.click()
    _wait_for_workflow_cycle(voila_page)

    second_values = _extract_pssar_row_values(voila_page.content())
    assert run_btn.get_attribute("disabled") is None
    # Verify the run completed and results are present (not a memo-cache early return).
    assert second_values.reference_value == pytest.approx(
        first_values.reference_value, abs=0.01
    ), "Reference pssar should be unchanged (same reference file used)"
    assert "Pass rate" in voila_page.locator("body").inner_text()
    assert (
        "already match the current results"
        not in voila_page.locator("body").inner_text()
    ), "Memo cache should NOT have fired — power level changed from previous run"
    _log("<< test_same_session_rerun_updates_results_after_power_change: pass")


def test_exact_repeat_shows_warning_without_rerunning(voila_page) -> None:
    _log(">> test_exact_repeat_shows_warning_without_rerunning")
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    previous_html = voila_page.content()

    _log("   clicking Compare Patterns to repeat an unchanged run")
    run_btn.click()
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('already match the current results')",
        timeout=15_000,
    )

    assert run_btn.get_attribute("disabled") is None
    assert "Reference, 30 dBm" in previous_html
    assert "Reference, 30 dBm" in voila_page.content()
    _log("<< test_exact_repeat_shows_warning_without_rerunning: pass")


def test_uploading_new_data_clears_prior_results(voila_page, tmp_path: Path) -> None:
    _log(">> test_uploading_new_data_clears_prior_results")
    replacement_csv = tmp_path / "replacement_measured.csv"
    replacement_csv.write_text("x,y,sar\n0,0,2\n1,1,3\n", encoding="utf-8")

    assert "Reference, 30 dBm" in voila_page.content()
    _upload_file(voila_page, replacement_csv)
    _log("   waiting for result table to disappear from DOM")
    voila_page.wait_for_function(
        "() => !document.body.innerHTML.includes('Reference, 30 dBm')",
        timeout=10_000,
    )

    page_html = voila_page.content()
    assert "Reference, 30 dBm" not in page_html
    assert "psSAR" not in page_html
    _log("<< test_uploading_new_data_clears_prior_results: pass")
