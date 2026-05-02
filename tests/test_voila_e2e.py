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

from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KERNEL_TIMEOUT = 60_000  # ms — widgets appear in ~30s after page load


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


def test_clicking_filter_button_activates_it(voila_page) -> None:
    # Toggle buttons ARE the .widget-toggle-button elements (not nested inside them)
    voila_page.locator(".widget-toggle-button").first.click()
    voila_page.wait_for_selector(".widget-toggle-button.mod-active", timeout=10_000)
    assert voila_page.locator(".widget-toggle-button.mod-active").count() >= 1


def test_file_upload_updates_filename_label(voila_page) -> None:
    csv_path = _REPO_ROOT / "notebooks" / "uploaded_data" / "measured_data.csv"
    # FileUpload triggers a native file chooser on click — intercept with expect_file_chooser
    with voila_page.expect_file_chooser(timeout=5_000) as fc:
        voila_page.locator(".widget-upload").click()
    fc.value.set_files(str(csv_path))
    voila_page.wait_for_function(
        "() => document.body.innerText.includes('measured_data.csv')",
        timeout=15_000,
    )


def test_run_button_enables_after_upload_and_unique_filter(voila_page) -> None:
    """Clicks filter buttons until exactly one reference matches, then asserts run is enabled."""
    csv_path = _REPO_ROOT / "notebooks" / "uploaded_data" / "measured_data.csv"

    # Ensure file is uploaded (previous test may have done it already)
    if "measured_data.csv" not in voila_page.locator("body").inner_text():
        with voila_page.expect_file_chooser(timeout=5_000) as fc:
            voila_page.locator(".widget-upload").click()
        fc.value.set_files(str(csv_path))
        voila_page.wait_for_function(
            "() => document.body.innerText.includes('measured_data.csv')",
            timeout=15_000,
        )

    # Click one button per filter group until the run button enables
    toggle_buttons = voila_page.locator(".widget-toggle-button")
    count = toggle_buttons.count()
    run_btn = voila_page.locator("button:has-text('Compare Patterns')")

    for i in range(count):
        if run_btn.get_attribute("disabled") is None:
            break
        btn = toggle_buttons.nth(i)
        if not btn.is_disabled():
            btn.click()
            voila_page.wait_for_timeout(500)

    assert run_btn.get_attribute("disabled") is None, (
        "Run button should be enabled once a measured file and unique reference are selected"
    )


def test_run_workflow_and_check_results_table(voila_page) -> None:
    """Clicks Compare Patterns and asserts the results tables render without error."""
    _WORKFLOW_TIMEOUT = 120_000  # ms — workflow can take a while on first uvx run

    run_btn = voila_page.locator("button:has-text('Compare Patterns')")
    assert run_btn.get_attribute("disabled") is None, "Run button must be enabled first"

    run_btn.click()

    _FIND_BTN = (
        "() => [...document.querySelectorAll('button')]"
        ".find(b => b.textContent.includes('Compare Patterns'))"
    )
    # Wait for button to go disabled (workflow started)
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && b.disabled; }}",
        timeout=10_000,
    )
    # Wait for button to re-enable (workflow finished)
    voila_page.wait_for_function(
        f"() => {{ const b = ({_FIND_BTN})(); return b && !b.disabled; }}",
        timeout=_WORKFLOW_TIMEOUT,
    )

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
