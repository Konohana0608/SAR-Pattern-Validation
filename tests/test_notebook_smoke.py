"""Notebook kernel smoke test — §V3.

Executes notebooks/voila.ipynb in a real Jupyter kernel and asserts the run
completes without error.  This test is intentionally NOT marked @pytest.mark.e2e
so it runs before Playwright is started, giving a fast, readable failure
instead of an opaque timeout when a Python-level bug (TraitError, ImportError,
SyntaxError, widget misconfiguration) prevents the notebook from serving.

Run inside the itisfoundation/jupyter-math container:
    /home/jovyan/.venv/bin/python -m pytest -v -o "addopts=" \
        -m notebook_smoke tests/test_notebook_smoke.py

Requires:
    - jupyter nbconvert (available in the jupyter-math image)
    - git-lfs data pulled (data/database/ must contain at least one CSV)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NOTEBOOK = _REPO_ROOT / "notebooks" / "voila.ipynb"
_KERNEL_TIMEOUT_S = 90


@pytest.mark.notebook_smoke
def test_v3_notebook_kernel_executes_without_error() -> None:
    """§V3: voila notebook must execute in a Jupyter kernel without exception."""
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
        output_path = tmp.name

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={_KERNEL_TIMEOUT_S}",
            "--output",
            output_path,
            str(_NOTEBOOK),
        ],
        capture_output=True,
        text=True,
        timeout=_KERNEL_TIMEOUT_S + 30,
    )

    assert result.returncode == 0, (
        "Notebook kernel raised an exception — fix this before running Playwright.\n\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
