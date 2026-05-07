"""End-to-end Playwright tests for the Voila UI.

Run inside the jupyter-math container via:
    make voila-test-docker

Manual repro inside `make voila-shell-docker`:
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

import re
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright")

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_TIMEOUT = 120_000  # ms — kernel startup + initial render
_UPLOAD_CSV_PATH = _REPO_ROOT / "data" / "example" / "measured_sSAR1g.csv"


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

    toggle_buttons = voila_page.locator(".widget-toggle-button")
    count = toggle_buttons.count()
    _log(f"   found {count} toggle buttons; will click until run button enables")
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")

    for i in range(count):
        if run_btn.get_attribute("disabled") is None:
            _log(f"<< ensure_run_button_enabled: enabled after {i} toggle clicks")
            return
        btn = toggle_buttons.nth(i)
        if not btn.is_disabled():
            _log(f"   clicking toggle {i}/{count}")
            btn.click()
            voila_page.wait_for_timeout(500)

    assert run_btn.get_attribute("disabled") is None, (
        "Run button should be enabled once a measured file and unique reference are selected"
    )
    _log("<< ensure_run_button_enabled: enabled after exhausting toggles")


def _wait_for_workflow_cycle(voila_page, timeout_ms: int = 120_000) -> None:
    """Wait for Compare Patterns to disable (running) then re-enable (done)."""
    _log(">> wait_for_workflow_cycle: starting")
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    _FIND_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )
    _log("   waiting up to 10s for run button to become disabled (cycle started)")
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && b.disabled; }}",
        timeout=10_000,
    )
    _log(
        f"   waiting up to {timeout_ms / 1000:.0f}s for run button to re-enable (cycle complete)"
    )
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && !b.disabled; }}",
        timeout=timeout_ms,
    )
    assert run_btn.get_attribute("disabled") is None
    _log("<< wait_for_workflow_cycle: complete")


def _extract_pssar_row_values(page_html: str) -> tuple[float, float, float, float]:
    _log(">> extract_pssar_row_values: scanning page HTML")
    match = re.search(
        (
            r"Peak spatial-average SAR \(psSAR\).*?"
            r"<tbody><tr>.*?"
            r"<td style=\"[^\"]*\"><b style=\"color:[^\"]*\">(?:Pass|Fail)</b></td>.*?"
            r"<td style=\"[^\"]*\">([0-9.]+) W/kg</td>.*?"
            r"<td style=\"[^\"]*\">([0-9.]+) W/kg</td>.*?"
            r"<td style=\"[^\"]*\">([0-9.]+) W/kg</td>.*?"
            r"<td style=\"[^\"]*\">([-0-9.]+)</td>"
        ),
        page_html,
        flags=re.S,
    )
    assert match is not None, "Could not extract the measured-value cell from the page."
    values = tuple(float(match.group(index)) for index in range(1, 5))
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


@pytest.mark.skip(
    reason="requires Phase B port: filter ToggleButton grid (button_groups) "
    "is constructed in the notebook but never displayed at startup on main-melanie"
)
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


@pytest.mark.skip(
    reason="requires Phase B port: filter ToggleButton grid not displayed on main-melanie "
    "(see test_filter_toggle_buttons_are_visible)"
)
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


@pytest.mark.skip(
    reason="requires Phase B port: filter ToggleButton grid is what enables the "
    "run button; without the grid being displayed, run button never enables"
)
def test_run_button_enables_after_upload_and_unique_filter(voila_page) -> None:
    """Clicks filter buttons until exactly one reference matches; asserts run is enabled."""
    _log(">> test_run_button_enables_after_upload_and_unique_filter")
    _ensure_run_button_enabled(voila_page)
    _log("<< test_run_button_enables_after_upload_and_unique_filter: pass")


# ---------------------------------------------------------------------------
# Tests gated on features that haven't been cherry-picked back into main-melanie.
# Each Phase B PR removes the matching skip mark when it lands the feature.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="requires Phase B1: results table + JSON manifest")
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
    assert "Peak spatial-average SAR" in page_html, (
        f"psSAR table not found.\nBody text:\n{body_text[:3000]}\n\nPage HTML tail:\n{page_html[-2000:]}"
    )
    assert "SAR pattern match" in page_html, (
        f"Pattern match table not found.\nBody text:\n{body_text[:3000]}"
    )
    assert "Pass" in page_html or "Fail" in page_html, (
        f"No Pass/Fail result found.\nPage HTML tail:\n{page_html[-2000:]}"
    )
    _log("<< test_run_workflow_and_check_results_table: pass")


@pytest.mark.skip(reason="requires Phase B1 + B4: results table + power-rerun state")
def test_restored_session_rerun_updates_results_after_power_change(voila_page) -> None:
    _log(">> test_restored_session_rerun_updates_results_after_power_change")
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    if "Peak spatial-average SAR" not in voila_page.content():
        _log("   no prior results in DOM — running once to seed state")
        run_btn.click()
        _wait_for_workflow_cycle(voila_page)
    first_measured_value, first_measured_30dbm, _, first_scaling_error = (
        _extract_pssar_row_values(voila_page.content())
    )

    _log(f"   reloading page (timeout={_KERNEL_TIMEOUT}ms)")
    voila_page.reload(timeout=_KERNEL_TIMEOUT)
    voila_page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)
    voila_page.wait_for_function(
        f"() => document.body.innerText.includes('{_UPLOAD_CSV_PATH.name}')",
        timeout=15_000,
    )

    restored_measured_value, restored_measured_30dbm, _, restored_scaling_error = (
        _extract_pssar_row_values(voila_page.content())
    )
    assert restored_measured_value == pytest.approx(first_measured_value, abs=0.01)
    assert restored_measured_30dbm == pytest.approx(first_measured_30dbm, abs=0.01)
    assert restored_scaling_error == pytest.approx(first_scaling_error, abs=0.01)

    _set_power_level(voila_page, 10.0)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None

    _log("   clicking Compare Patterns after power change")
    run_btn.click()
    voila_page.wait_for_function(
        "(expected) => !document.body.innerText.includes(expected)",
        arg=f"{first_measured_30dbm:.2f} W/kg",
        timeout=10_000,
    )

    second_measured_value, second_measured_30dbm, _, second_scaling_error = (
        _extract_pssar_row_values(voila_page.content())
    )
    assert run_btn.get_attribute("disabled") is None
    assert second_measured_value == pytest.approx(first_measured_value, abs=0.01)
    assert second_measured_30dbm != pytest.approx(first_measured_30dbm, abs=0.01)
    assert second_scaling_error != pytest.approx(first_scaling_error, abs=0.01)
    _log("<< test_restored_session_rerun_updates_results_after_power_change: pass")


@pytest.mark.skip(reason="requires Phase B1 + B4: results table + power-rerun state")
def test_same_session_rerun_updates_results_after_power_change(voila_page) -> None:
    _log(">> test_same_session_rerun_updates_results_after_power_change")
    _ensure_run_button_enabled(voila_page)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None

    first_measured_value, first_measured_30dbm, _, first_scaling_error = (
        _extract_pssar_row_values(voila_page.content())
    )

    _set_power_level(voila_page, 17.0)
    _log("   clicking Compare Patterns after power change")
    run_btn.click()
    voila_page.wait_for_function(
        "(expected) => !document.body.innerText.includes(expected)",
        arg=f"{first_measured_30dbm:.2f} W/kg",
        timeout=10_000,
    )

    second_measured_value, second_measured_30dbm, _, second_scaling_error = (
        _extract_pssar_row_values(voila_page.content())
    )
    assert run_btn.get_attribute("disabled") is None
    assert second_measured_value == pytest.approx(first_measured_value, abs=0.01)
    assert second_measured_30dbm != pytest.approx(first_measured_30dbm, abs=0.01)
    assert second_scaling_error != pytest.approx(first_scaling_error, abs=0.01)
    _log("<< test_same_session_rerun_updates_results_after_power_change: pass")


@pytest.mark.skip(reason="requires Phase B3: memo cache 'already match' warning")
def test_exact_repeat_shows_warning_without_rerunning(voila_page) -> None:
    _log(">> test_exact_repeat_shows_warning_without_rerunning")
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    previous_html = voila_page.content()

    _log("   clicking Compare Patterns to repeat an unchanged run")
    run_btn.click()
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('already match the current results')",
        timeout=10_000,
    )

    assert run_btn.get_attribute("disabled") is None
    assert "Peak spatial-average SAR" in previous_html
    assert "Peak spatial-average SAR" in voila_page.content()
    _log("<< test_exact_repeat_shows_warning_without_rerunning: pass")


@pytest.mark.skip(
    reason="requires Phase B1 + state-clear: results table + clear-on-new-upload"
)
def test_uploading_new_data_clears_prior_results(voila_page, tmp_path: Path) -> None:
    _log(">> test_uploading_new_data_clears_prior_results")
    replacement_csv = tmp_path / "replacement_measured.csv"
    replacement_csv.write_text("x,y,sar\n0,0,2\n1,1,3\n", encoding="utf-8")

    assert "Peak spatial-average SAR" in voila_page.content()
    _upload_file(voila_page, replacement_csv)
    _log("   waiting for Peak spatial-average SAR to disappear from DOM")
    voila_page.wait_for_function(
        "() => !document.body.innerHTML.includes('Peak spatial-average SAR')",
        timeout=10_000,
    )

    page_html = voila_page.content()
    assert "Peak spatial-average SAR" not in page_html
    assert "SAR pattern match" not in page_html
    _log("<< test_uploading_new_data_clears_prior_results: pass")
