"""Stdlib-only bootstrap for the Voila notebook running on jupyter-math.

Production uses `itisfoundation/jupyter-math:3.0.5`, whose stock kernel is
Python 3.9. The application package itself remains Python 3.10+, so the
notebook must NOT try to install or import that package directly.

This bootstrap keeps the kernel-side concerns intentionally small:
- ensure `uv` is available for subprocess calls
- make the uv bin directory visible on `PATH`

The notebook UI then talks to the real package only through `uvx`.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def _install_uv() -> None:
    if shutil.which("uv") is not None:
        return
    subprocess.run(
        ["bash", "-c", "wget -qO- https://astral.sh/uv/install.sh | sh"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    user_bin = os.path.expanduser("~/.local/bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if user_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([user_bin, *path_parts])


def ensure_runtime_environment(*, target_python: str | None = None) -> None:
    """Ensure kernel-safe prerequisites for the notebook runtime.

    `target_python` is accepted for backward compatibility with existing test
    and docker harness call sites, but is intentionally unused now that the
    notebook no longer installs the application into the kernel environment.
    """
    del target_python
    _install_uv()
