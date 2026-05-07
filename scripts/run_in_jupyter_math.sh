#!/usr/bin/env bash
# Runs INSIDE itisfoundation/jupyter-math:3.0.5.
# Mirrors the cloud workspace layout from osparc_makefile/Makefile:
#   /home/jovyan/work/workspace/sar-pattern-validation/   <- repo (bind-mounted)
#   /home/jovyan/work/workspace/voila.ipynb               <- copy of notebooks/voila.ipynb
#
# Production parity:
#   - The notebook bootstraps its own runtime deps via `notebook_bootstrap.py`
#     (uv only). The stock python-maths kernel stays on Python 3.9 and the
#     actual app runs through `uvx` in an isolated env.
#   - We DO install testing-layer deps (pytest, pytest-playwright, playwright,
#     pytest-xdist, pytest-cov) and the Chromium browser. The user never does
#     these in production — they're test-harness only.
#   - Tests run against /home/jovyan/.venv/bin/python so they exercise the same
#     interpreter the production kernel uses (Python 3.9.10).
#
# Subcommands: test [extra pytest args] | test-e2e | smoke | shell | exec <cmd...>
set -euo pipefail

WORKSPACE=/home/jovyan/work/workspace
REPO="$WORKSPACE/sar-pattern-validation"
KERNEL_VENV=/home/jovyan/.venv
KERNEL_PY="$KERNEL_VENV/bin/python"

# --------------------------------------------------------------------------
# 1. System deps for headless chromium (Playwright) + curl/git-lfs. Re-run
#    every container start because --rm discards the container layer; the
#    volume-cached /var/lib/apt/lists keeps `apt-get update` fast (~5s) and
#    `apt-get install` is idempotent (~5-10s when packages already in cache).
# --------------------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
    echo ">> apt-get install playwright deps + tooling"
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        curl ca-certificates git-lfs procps tree \
        libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libxkbcommon0 libatspi2.0-0 libx11-6 libxcomposite1 \
        libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libasound2 \
        libpangocairo-1.0-0 libpango-1.0-0 libcairo2 fonts-liberation \
        >/dev/null
    git lfs install --force >/dev/null 2>&1 || true
fi

# --------------------------------------------------------------------------
# 2. Install uv (matches notebook_bootstrap._install_uv() / production
#    notebook bootstrap). The backend also calls `uvx --no-cache --from <spec>`
#    at workflow time, so uv must be on PATH.
# --------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo ">> install uv"
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh
fi

# --------------------------------------------------------------------------
# 3. Notebook-side bootstrap only: ensure uv is available to the stock kernel.
#    The production notebook must not install the application into the kernel
#    environment anymore.
# --------------------------------------------------------------------------
cd "$REPO"
echo ">> notebook_bootstrap: ensure uv is available to the kernel"
PYTHONPATH="$REPO" "$KERNEL_PY" -c "
import notebook_bootstrap
notebook_bootstrap.ensure_runtime_environment(target_python='$KERNEL_PY')
"

# --------------------------------------------------------------------------
# 4. Testing-layer deps into the kernel venv. The user never installs these
#    in production — they're test-harness only. Installing them into
#    /home/jovyan/.venv (rather than /opt/conda) keeps tests on the same
#    interpreter the production kernel uses.
# --------------------------------------------------------------------------
echo ">> uv pip install testing-layer deps into $KERNEL_VENV"
uv pip install --python "$KERNEL_PY" --quiet \
    pytest pytest-playwright pytest-xdist pytest-cov playwright

echo ">> playwright install chromium"
"$KERNEL_VENV/bin/playwright" install chromium

# --------------------------------------------------------------------------
# 5. git-lfs pull + stage the notebook into the workspace root (mirrors
#    copy-notebook in osparc_makefile/Makefile).
# --------------------------------------------------------------------------
git lfs pull >/dev/null 2>&1 || true

mkdir -p "$WORKSPACE"
cp "$REPO/notebooks/voila.ipynb" "$WORKSPACE/voila.ipynb"

# --------------------------------------------------------------------------
# 6. Chown back to the host UID/GID on exit so bind-mounted files don't come
#    out root-owned. Only set when HOST_UID/HOST_GID are passed in.
# --------------------------------------------------------------------------
if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    trap 'find "$REPO" -uid 0 -exec chown "$HOST_UID:$HOST_GID" {} + 2>/dev/null || true' EXIT
fi

# --------------------------------------------------------------------------
# 7. Production parity: tests import `sar_pattern_validation` via sys.path
#    only when the test itself needs it. The notebook path relies on the
#    repo-root shim, not on importing the package inside the kernel.
# --------------------------------------------------------------------------
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Make the kernel venv's bin/ first on PATH so subprocesses (voila, jupyter,
# uvx) resolve to the production interpreter.
export PATH="$KERNEL_VENV/bin:$PATH"
export SAR_PATTERN_VALIDATION_PY39_PYTHON="$KERNEL_PY"

# --------------------------------------------------------------------------
# 8. Dispatch.
# --------------------------------------------------------------------------
CMD="${1:-test}"
shift || true
case "$CMD" in
    test)
        echo ">> running stability suite (extra args: $*)"
        exec "$KERNEL_PY" -m pytest -v -o "addopts=" --run-e2e -p no:xdist \
            tests/test_voila_stability.py "$@"
        ;;
    test-e2e)
        echo ">> running full e2e suite (extra args: $*)"
        exec "$KERNEL_PY" -m pytest -v -o "addopts=" --run-e2e -p no:xdist \
            tests/test_voila_stability.py tests/test_voila_e2e.py "$@"
        ;;
    smoke)
        echo ">> run_voila_smoke.py"
        exec "$KERNEL_PY" run_voila_smoke.py
        ;;
    shell)
        echo ">> workspace ready at $WORKSPACE"
        echo "   repo:        $REPO"
        echo "   notebook:    $WORKSPACE/voila.ipynb"
        echo "   kernel py:   $KERNEL_PY"
        echo "   PYTHONPATH:  $PYTHONPATH"
        cd "$WORKSPACE"
        exec bash
        ;;
    exec)
        exec "$@"
        ;;
    *)
        echo "unknown subcommand: $CMD" >&2
        echo "usage: $0 {test|test-e2e|smoke|shell|exec <cmd...>}" >&2
        exit 2
        ;;
esac
