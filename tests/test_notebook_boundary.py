from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


def test_notebook_bootstrap_does_not_install_app_into_kernel() -> None:
    source = Path("notebook_bootstrap.py").read_text(encoding="utf-8")

    assert "uv pip install" not in source
    assert "eval_type_backport" not in source
    assert "runtime deps" not in source.lower()


def test_notebook_uses_local_shim_not_package_frontend() -> None:
    notebook = json.loads(Path("notebooks/voila.ipynb").read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "from notebook_support import" in sources
    assert "sar_pattern_validation.voila_frontend" not in sources


def test_py39_can_compile_and_import_kernel_shim() -> None:
    py39 = os.environ.get("SAR_PATTERN_VALIDATION_PY39_PYTHON")
    if not py39:
        pytest.skip("SAR_PATTERN_VALIDATION_PY39_PYTHON is not set")

    repo_root = Path.cwd()
    files = [
        "notebook_bootstrap.py",
        "notebook_support/__init__.py",
        "notebook_support/catalog.py",
        "notebook_support/runtime.py",
        "notebook_support/runner.py",
        "notebook_support/state.py",
        "notebook_support/ui.py",
    ]
    subprocess.run([py39, "-m", "py_compile", *files], check=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(
        [
            py39,
            "-c",
            (
                "import notebook_bootstrap;"
                "import notebook_support;"
                "import notebook_support.runtime;"
                "import notebook_support.runner;"
                "import notebook_support.state;"
                "import notebook_support.ui"
            ),
        ],
        check=True,
        env=env,
    )
