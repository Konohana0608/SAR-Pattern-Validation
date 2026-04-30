from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    notebook_source = repo_root / "notebooks" / "voila.ipynb"
    port = _free_port()
    env = os.environ.copy()
    env["SAR_PATTERN_VALIDATION_BACKEND_MODE"] = "local"
    env["SAR_PATTERN_VALIDATION_LOCAL_PACKAGE_SOURCE"] = str(repo_root)

    with tempfile.TemporaryDirectory(prefix="sar-voila-smoke-") as temp_dir:
        workspace_root = Path(temp_dir) / "workspace"
        workspace_root.mkdir()
        (workspace_root / "sar-pattern-validation").symlink_to(repo_root)
        shutil.copy2(notebook_source, workspace_root / "voila.ipynb")

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "voila",
                "voila.ipynb",
                "--no-browser",
                f"--port={port}",
                "--Voila.ip=127.0.0.1",
            ],
            cwd=workspace_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            target_urls = [
                f"http://127.0.0.1:{port}/voila/render/voila.ipynb",
                f"http://127.0.0.1:{port}/",
            ]
            deadline = time.time() + 120
            while time.time() < deadline:
                if proc.poll() is not None:
                    output = proc.stdout.read() if proc.stdout is not None else ""
                    raise RuntimeError(
                        f"Voila exited early with code {proc.returncode}.\n{output}"
                    )
                for target_url in target_urls:
                    try:
                        with urllib.request.urlopen(target_url, timeout=2) as response:
                            body = response.read().decode("utf-8", errors="replace")
                        if "SAR Pattern Validation" in body:
                            print(f"Voila smoke passed at {target_url}")
                            return 0
                    except (urllib.error.URLError, TimeoutError):
                        continue
                time.sleep(1)

            output = proc.stdout.read() if proc.stdout is not None else ""
            raise TimeoutError(
                "Timed out waiting for Voila.\n"
                f"Checked URLs: {target_urls}\n"
                f"Process output so far:\n{output}"
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
