# Validation Rules & Test Infrastructure (from AGENTS.md)

## Required checks on every voila_frontend change

- `tests/test_voila_frontend.py` — unit/mock layer; must pass before any PR
- `tests/test_voila_e2e.py` — browser layer via Playwright; must pass before any PR
- When changing lifecycle, completion flow, rerun, upload/reset, or warnings/errors: validate BOTH layers
- If a change appears to fix backend logs but not visible UI state: reproduce in browser and verify with Playwright

## Correct test invocation (important)
`make tests` uses `uv run pytest` which can resolve to system Python 3.10 (wrong).
Always use `uv run python -m pytest` directly:
```
uv run python -m pytest -v -o "addopts=" tests/test_voila_frontend.py
uv run python -m pytest -v -o "addopts=" --run-e2e -p no:xdist tests/test_voila_e2e.py
```
e2e requires: `--run-e2e -p no:xdist` flags + playwright chromium installed.

## Test status (2026-05-03)
77 tests pass: 65 unit + 12 e2e. All previously listed failing tests resolved.

## Fast completed-state fixture
`completed_workspace` in `tests/conftest.py`:
- Returns `(WorkspacePaths, WorkflowResultPayload)` in <50ms (no backend subprocess)
- Copies pre-baked demo images from `tests/fixtures/demo_run/` into tmp workspace
- Use when `_restore_outputs_available()` must be True or image widgets must render
- Demo data: measured=`data/example/measured_sSAR1g.csv`, reference=`dipole_1450MHz_Flat_10mm_10g.csv`, power=23.0 dBm
- SHA256 of measured CSV computed live from the copied file (always accurate)

## Verification gates before PR merge
1. Unit tests prove chosen completion-path abstraction
2. Backend JSON/error/log behavior confirmed against staged baseline
3. Fresh `make voila-smoke`: backend completes → visible tables/images
4. One compare-patterns cycle + one rerun cycle via e2e
5. Manual check: session restore and stale-server restart cannot leave UI silently hung
