# SAR Pattern Validation — Project Overview

## Purpose
Python toolkit for SAR (Specific Absorption Rate) pattern validation. Loads SAR maps from CSV, registers measured vs reference, evaluates 2D gamma. Exposed as a Voila interactive dashboard.

## Tech Stack
- Python ≥3.10, managed with `uv` (lockfile: `uv.lock`)
- Core: numpy, pandas, scipy, scikit-image, SimpleITK, matplotlib
- UI: ipywidgets, Voila (≥0.5), IPython
- Data validation: pydantic v2
- Testing: pytest, pytest-playwright (e2e), pytest-xdist (parallel)
- Linting/format: ruff (line-length=88, target py310)
- Type checking: ty
- Pre-commit hooks: ruff-check, ruff-format, standard file checks

## Package Layout
```
src/sar_pattern_validation/
  __init__.py
  errors.py
  gamma_eval.py
  image_loader.py
  measurement_validation_report.py
  plotting.py
  registration2d.py
  sample_catalog.py
  utils.py
  workflow_cli.py          # CLI entrypoint: sar-pattern-validation
  workflow_config.py
  workflow_schema.py
  workflows.py
  voila_frontend/
    __init__.py
    models.py              # Pydantic models: UiState, WorkflowResultPayload
    runner.py              # Subprocess backend execution
    runtime.py             # Workspace/runtime path discovery
    state.py               # UI state persistence (load/save ui_state.json)
    ui.py                  # Main ipywidgets UI: SarGammaComparisonUI
notebooks/
  voila.ipynb              # Thin Voila bootstrap (imports voila_frontend)
  system_state/ui_state.json
  uploaded_data/
tests/
  conftest.py
  test_voila_frontend.py   # Unit/mock tests for UI lifecycle
  test_voila_e2e.py        # Playwright browser tests
  test_voila_*             # Other voila tests
  test_*.py                # Core algorithm tests
```

## Entrypoints
- CLI: `uv run sar-pattern-validation` (workflow_cli.py)
- Voila dashboard: `uv run voila notebooks/voila.ipynb`
- Smoke test: `uv run python run_voila_smoke.py`
