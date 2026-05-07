"""Minimal end-to-end Playwright smoke against a live Voila server.

Phase A of the container-first harness: prove that bringing up
``itisfoundation/jupyter-math:3.0.5`` with the repo bind-mounted, then
serving ``notebooks/voila.ipynb``, yields a page that:

  1. Returns HTTP 200.
  2. Renders the main UI ("Compare Patterns" run button is present).
  3. Does not surface a Python traceback.

Widget-level assertions are deferred to Phase B cherry-picks (results table,
popups, etc.) — once features are ported back onto main-melanie one-by-one,
each PR layers its own e2e expectation here.

Run inside the jupyter-math container via:
    make voila-test-docker
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import expect  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

_KERNEL_TIMEOUT = 120_000  # ms — kernel startup + initial render


@pytest.fixture(scope="module")
def voila_page(playwright, voila_server):
    base_url, _ = voila_server
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(_KERNEL_TIMEOUT)
    page.goto(base_url + "/", timeout=_KERNEL_TIMEOUT)
    expect(page.locator("body")).to_contain_text(
        "Compare Patterns", timeout=_KERNEL_TIMEOUT
    )
    yield page
    context.close()
    browser.close()


def test_voila_page_loads(voila_page) -> None:
    body = voila_page.locator("body").inner_text(timeout=_KERNEL_TIMEOUT)
    assert "Compare Patterns" in body, (
        f"Expected 'Compare Patterns' in page body. First 2KB:\n{body[:2000]}"
    )


def test_voila_page_has_no_python_traceback(voila_page) -> None:
    body = voila_page.locator("body").inner_text(timeout=_KERNEL_TIMEOUT)
    assert "Traceback" not in body, f"Python traceback in page body:\n{body[:3000]}"
