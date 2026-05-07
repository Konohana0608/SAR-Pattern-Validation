#!/usr/bin/env bash
# Host-side wrapper. Spins up itisfoundation/jupyter-math:3.0.5 with the repo
# bind-mounted at the same path the cloud uses, then delegates to
# scripts/run_in_jupyter_math.sh.
#
# Why: a recurring "kernel just died" / silent UI failure on the host venv
# (WSL2) is impossible to disambiguate from a real product bug — Playwright
# only sees the symptom. Running inside the same image production uses
# eliminates the host-vs-container drift dimension. If the bug reproduces
# here, it's a product bug. If it disappears, the host venv is the problem.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${JUPYTER_MATH_IMAGE:-itisfoundation/jupyter-math:3.0.5}"
WORKSPACE_IN_CONTAINER="/home/jovyan/work/workspace"
REPO_IN_CONTAINER="$WORKSPACE_IN_CONTAINER/sar-pattern-validation"

# Persistent caches so reruns aren't paying the 200MB+ chromium download,
# the uvx resolve cost, or repeated pip downloads every time.
docker volume create sar-jupyter-math-uv-cache  >/dev/null
docker volume create sar-jupyter-math-pw-cache  >/dev/null
docker volume create sar-jupyter-math-apt-lists >/dev/null
docker volume create sar-jupyter-math-pip-cache >/dev/null

INTERACTIVE_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then INTERACTIVE_FLAGS=(-it); fi

# --shm-size: headless chromium uses /dev/shm; default 64MB causes flaky
# crashes that look like kernel deaths. Bump to 2g to match Playwright's
# recommendation.
# --entrypoint=bash: jupyter-math's default entrypoint expects oSPARC
# dynamic-sidecar env vars and dies under `set -u` when they're unset. We
# don't need its jupyter-server startup logic — the inner script runs Voila
# directly.
exec docker run --rm "${INTERACTIVE_FLAGS[@]}" \
    --name "sar-voila-jm-$$" \
    --user 0 \
    --shm-size=2g \
    --network bridge \
    --entrypoint=bash \
    -e HOME=/root \
    -e UV_CACHE_DIR=/root/.cache/uv \
    -e PIP_CACHE_DIR=/root/.cache/pip \
    -e PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
    -e SAR_PATTERN_VALIDATION_BACKEND_MODE=local \
    -e MPLBACKEND=Agg \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -v "$REPO_ROOT:$REPO_IN_CONTAINER" \
    -v sar-jupyter-math-uv-cache:/root/.cache/uv \
    -v sar-jupyter-math-pip-cache:/root/.cache/pip \
    -v sar-jupyter-math-pw-cache:/root/.cache/ms-playwright \
    -v sar-jupyter-math-apt-lists:/var/lib/apt/lists \
    -w "$REPO_IN_CONTAINER" \
    "$IMAGE" \
    "$REPO_IN_CONTAINER/scripts/run_in_jupyter_math.sh" "$@"
