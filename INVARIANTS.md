# INVARIANTS â€” sar-pattern-validation

<!-- Auto-injected into SPEC Â§C on every dispatch. Agents must not modify. -->
<!-- Source: Serena memories + ConvInt extracts 2026-04-27 to 2026-05-04 -->
<!-- To update: add rows to Fix-Regression History, never remove existing ones -->

## Coupling Invariants

- **[INV-001]** All widget mutations must occur on the kernel thread via `_dispatch_ui_update`.
  Background threads must be started with `contextvars.copy_context()`.
  Verified by: `uv run python -m pytest -v -o "addopts=" tests/test_voila_frontend.py`

- **[INV-002]** Log output must use `widgets.Output` â€” never `widgets.HTML.value=` from a
  logging handler thread (comm message requires `shell_parent` context, silently fails).
  Verified by: browser check after any logging-related change

- **[INV-003]** `WorkflowResult` must be used (not replaced by dataclass `field` vs pydantic
  `Field`); `workflow_config.py` and `workflow_schema.py` must include measurement area fields.
  Verified by: `uv run python -m pytest -v -o "addopts=" tests/test_voila_frontend.py`

- **[INV-004]** `GIT_LFS_SKIP_SMUDGE=1` must be set in the subprocess environment in
  `runner.py`. Removing it causes exit 128 on git operations in install environments.
  Verified by: `grep -c "GIT_LFS_SKIP_SMUDGE" src/sar_pattern_validation/voila_frontend/runner.py`
  (must be â‰¥1)

- **[INV-005]** `ipywidgets` must NOT be imported at module level in `src/__init__.py`.
  It is a dev dependency only; module-level import breaks CLI usage with `JSONDecodeError`.
  Verified by: `grep -c "ipywidgets" src/sar_pattern_validation/__init__.py` (must be 0)

- **[INV-006]** `threading.Timer` must NEVER be used for widget dispatch or completion
  polling. Timer threads lack kernel context; widget assignments silently drop.
  Verified by: `grep -rn "threading.Timer" src/sar_pattern_validation/voila_frontend/`
  (must return empty)

- **[INV-007]** `_dispatch_ui_update` is the ONLY valid widget mutation path from worker
  threads. Direct `.value =` assignment from worker threads silently fails.
  Verified by: code review of any new threading path + e2e test

## Regression Test Suite

```bash
# Unit layer (no subprocess) â€” always run before PR
uv run python -m pytest -v -o "addopts=" tests/test_voila_frontend.py

# E2E layer (Playwright + real Voila) â€” run before PR when lifecycle changes
uv run python -m pytest -v -o "addopts=" --run-e2e -p no:xdist tests/test_voila_e2e.py

# Smoke test â€” backend subprocess must complete
make voila-smoke

# Never use bare `make tests` â€” resolves to system Python 3.10, not project venv
```

## Fix-Regression History

| Date | Fixed | What it broke | Lesson |
|------|-------|---------------|--------|
| 2026-04-?? | Async backend execution | Voila kernel hung; logs went empty | Logs to widgets.HTML.value= silently fail from logging thread; use widgets.Output |
| 2026-04-?? | Progress bar threading | Progress bar froze (Exception in Thread-4) | Progress threads need contextvars.copy_context() |
| 2026-04-?? | HTML widget for log display | Log output went empty | widgets.HTML.value= needs shell_parent; back to widgets.Output + display_data entries |
| 2026-04-?? | Moved voila_frontend re-export to src/__init__.py | CLI dead (JSONDecodeError) | ipywidgets is dev-only; never import at module level in __init__.py |
| 2026-04-30 | Merge conflict resolution | WorkflowResult removed; pydantic Fieldâ†’field; measurement area fields dropped | Merge resolution must preserve all pydantic models; run tests immediately after merge |
| 2026-05-03 | threading.Timer-based completion polling (_schedule_ui_callback) | Button never re-enables after workflow | Timer threads lack kernel context; use _dispatch_ui_update from worker |

## Constraints for Teammates

1. **Before any widget update path change**: verify kernel-thread-only rule (INV-001, INV-006, INV-007). Run unit + e2e tests.
2. **Before any logging change**: confirm `widgets.Output` is still used, not `widgets.HTML` (INV-002).
3. **Before any merge conflict resolution**: manually verify WorkflowResult, Field (not field), measurement area fields (INV-003).
4. **Before touching runner.py**: grep for GIT_LFS_SKIP_SMUDGE (INV-004).
5. **Before touching src/__init__.py**: grep for ipywidgets (INV-005).
6. **If a fix seems to work in unit tests but not in the browser**: reproduce in browser â€” mock tests do NOT prove real Voila callback delivery.
