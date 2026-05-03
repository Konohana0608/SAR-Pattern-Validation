"""
End-to-end Playwright tests for the voila UI.

Run with:
    .venv/bin/python -m pytest tests/test_voila_e2e.py --run-e2e -p no:xdist --override-ini="addopts="

Prerequisites:
    .venv/bin/playwright install chromium

Design: all tests share one browser page to avoid restarting the voila kernel
per test (heavy imports make each startup ~90s on WSL2). Tests are ordered from
read-only to state-modifying; later tests deliberately build on earlier state.

DOM notes (ipywidgets 8.x + voila 0.5):
- Toggle buttons: class="... widget-toggle-button"; .mod-active added on selection
- FileUpload: class="... widget-upload"; no <input type=file> in DOM — use expect_file_chooser()
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_TIMEOUT = 60_000  # ms — widgets appear in ~30s after page load
_UPLOAD_CSV_PATH = _REPO_ROOT / "data" / "example" / "measured_sSAR1g.csv"


# ---------------------------------------------------------------------------
# Shared page fixture — kernel starts once for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def voila_page(playwright, voila_server):
    """Navigate to voila once and keep the page alive for all tests in this module."""
    base_url, _ = voila_server
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(_KERNEL_TIMEOUT)
    page.goto(base_url + "/", timeout=_KERNEL_TIMEOUT)
    # Block until any widget button appears — kernel has finished executing
    page.wait_for_selector(".widget-button", timeout=_KERNEL_TIMEOUT)
    yield page
    context.close()
    browser.close()


# ---------------------------------------------------------------------------
# Read-only smoke tests (run first — page is in fresh state)
# ---------------------------------------------------------------------------


def test_run_button_is_visible(voila_page) -> None:
    assert voila_page.locator("button:has-text('Compare Patterns')").is_visible()


def test_run_button_is_disabled_on_fresh_load(voila_page) -> None:
    btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert btn.get_attribute("disabled") is not None


def test_filter_toggle_buttons_are_visible(voila_page) -> None:
    assert voila_page.locator(".widget-toggle-button").count() > 0


def test_file_upload_button_is_present(voila_page) -> None:
    # ipywidgets FileUpload renders as a button with class widget-upload (no <input> in DOM)
    assert voila_page.locator(".widget-upload").count() == 1


# ---------------------------------------------------------------------------
# State-modifying tests (each builds on the previous)
# ---------------------------------------------------------------------------


def _ensure_run_button_enabled(voila_page) -> None:
    if _UPLOAD_CSV_PATH.name not in voila_page.locator("body").inner_text():
        _upload_file(voila_page, _UPLOAD_CSV_PATH)

    toggle_buttons = voila_page.locator(".widget-toggle-button")
    count = toggle_buttons.count()
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")

    for i in range(count):
        if run_btn.get_attribute("disabled") is None:
            return
        btn = toggle_buttons.nth(i)
        if not btn.is_disabled():
            btn.click()
            voila_page.wait_for_timeout(500)

    assert run_btn.get_attribute("disabled") is None, (
        "Run button should be enabled once a measured file and unique reference are selected"
    )


def _wait_for_workflow_cycle(voila_page, timeout_ms: int = 120_000) -> None:
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    _FIND_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && b.disabled; }}",
        timeout=10_000,
    )
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && !b.disabled; }}",
        timeout=timeout_ms,
    )
    assert run_btn.get_attribute("disabled") is None


def _upload_file(voila_page, file_path: Path) -> None:
    with voila_page.expect_file_chooser(timeout=5_000) as fc:
        voila_page.locator(".widget-upload").click()
    fc.value.set_files(str(file_path))
    voila_page.wait_for_function(
        "(expected) => document.body.innerText.includes(expected)",
        arg=file_path.name,
        timeout=15_000,
    )


def _extract_pssar_row_values(page_html: str) -> tuple[float, float, float, float]:
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
    return tuple(float(match.group(index)) for index in range(1, 5))


def _set_power_level(voila_page, value: float) -> None:
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


def test_clicking_filter_button_activates_it(voila_page) -> None:
    # Toggle buttons ARE the .widget-toggle-button elements (not nested inside them)
    voila_page.locator(".widget-toggle-button").first.click()
    voila_page.wait_for_selector(".widget-toggle-button.mod-active", timeout=10_000)
    assert voila_page.locator(".widget-toggle-button.mod-active").count() >= 1


def test_file_upload_updates_filename_label(voila_page) -> None:
    # FileUpload triggers a native file chooser on click — intercept with expect_file_chooser
    _upload_file(voila_page, _UPLOAD_CSV_PATH)


def test_run_button_enables_after_upload_and_unique_filter(voila_page) -> None:
    """Clicks filter buttons until exactly one reference matches, then asserts run is enabled."""
    _ensure_run_button_enabled(voila_page)


def test_run_workflow_and_check_results_table(voila_page) -> None:
    """Clicks Compare Patterns and asserts the results tables render without error."""
    _ensure_run_button_enabled(voila_page)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None, "Run button must be enabled first"

    run_btn.click()
    _wait_for_workflow_cycle(voila_page)

    body_text = voila_page.locator("body").inner_text()
    page_html = voila_page.content()

    # Check for error markers — fail fast with a helpful message
    assert "Traceback" not in body_text, (
        f"Python traceback in page:\n{body_text[:3000]}"
    )

    # Both result tables must appear (HTML widget text is in DOM)
    assert "Peak spatial-average SAR" in page_html, (
        f"psSAR table not found.\nBody text:\n{body_text[:3000]}\n\nPage HTML tail:\n{page_html[-2000:]}"
    )
    assert "SAR pattern match" in page_html, (
        f"Pattern match table not found.\nBody text:\n{body_text[:3000]}"
    )
    assert "Pass" in page_html or "Fail" in page_html, (
        f"No Pass/Fail result found.\nPage HTML tail:\n{page_html[-2000:]}"
    )


def test_restored_session_rerun_updates_results_after_power_change(voila_page) -> None:
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    if "Peak spatial-average SAR" not in voila_page.content():
        run_btn.click()
        _wait_for_workflow_cycle(voila_page)
    first_measured_value, first_measured_30dbm, _, first_scaling_error = (
        _extract_pssar_row_values(voila_page.content())
    )

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


def test_same_session_rerun_updates_results_after_power_change(voila_page) -> None:
    _ensure_run_button_enabled(voila_page)

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None

    first_measured_value, first_measured_30dbm, _, first_scaling_error = (
        _extract_pssar_row_values(voila_page.content())
    )

    _set_power_level(voila_page, 17.0)
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


def test_exact_repeat_shows_warning_without_rerunning(voila_page) -> None:
    _ensure_run_button_enabled(voila_page)
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    previous_html = voila_page.content()

    run_btn.click()
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('already match the current results')",
        timeout=10_000,
    )

    assert run_btn.get_attribute("disabled") is None
    assert "Peak spatial-average SAR" in previous_html
    assert "Peak spatial-average SAR" in voila_page.content()


def test_uploading_new_data_clears_prior_results(voila_page, tmp_path: Path) -> None:
    replacement_csv = tmp_path / "replacement_measured.csv"
    replacement_csv.write_text("x,y,sar\n0,0,2\n1,1,3\n", encoding="utf-8")

    assert "Peak spatial-average SAR" in voila_page.content()
    _upload_file(voila_page, replacement_csv)
    voila_page.wait_for_function(
        "() => !document.body.innerHTML.includes('Peak spatial-average SAR')",
        timeout=10_000,
    )

    page_html = voila_page.content()
    assert "Peak spatial-average SAR" not in page_html
    assert "SAR pattern match" not in page_html
