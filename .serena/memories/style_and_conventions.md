# Code Style & Conventions

## Formatting
- ruff, line-length=88, target-version=py310
- `from __future__ import annotations` at top of all source files
- Imports: stdlib → third-party → local (ruff I enforces)

## Type hints
- Full type hints on all public functions and class attributes
- Pydantic v2 models for data contracts (UiState, WorkflowResultPayload in models.py)
- `collections.abc.Callable` preferred over `typing.Callable`

## Naming
- snake_case for functions, variables, modules
- PascalCase for classes
- UPPER_CASE for module-level constants
- Private helpers prefixed with `_`

## Docstrings
- Minimal: only where the WHY is non-obvious
- No multi-paragraph docstrings or multi-line comment blocks

## Error handling
- Domain errors in errors.py
- Frontend-safe wrapper: `WorkflowExecutionError(RuntimeError)` in runner.py
- Only validate at system boundaries (CSV input, subprocess output)

## Threading rules (voila_frontend)
- Worker threads: produce result/failure state only — no widget mutation
- Widget mutation: only on the UI/kernel event loop (via `_dispatch_ui_update` or equivalent)
- Progress bar and button state: owned by the single completion path, not independent threads

## Testing
- pytest with markers: `slow`, `integration`, `e2e`
- e2e tests require running voila server + playwright
- Default pytest: parallel with `-n auto --dist worksteal`
- Voila frontend tests: `tests/test_voila_frontend.py` (mock/unit)
- Browser tests: `tests/test_voila_e2e.py` (playwright)
