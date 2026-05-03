# AGENTS - Voila Frontend

## Validation Requirements

- Treat `tests/test_voila_frontend.py` as a required check for mocked and unit-level Voila frontend behavior.
- Treat `tests/test_voila_e2e.py` as a required check for browser-visible Voila behavior.
- For full user-workflow validation, use Playwright-controlled UI coverage on top of the Python test suite; unit tests alone are not sufficient.
- When changing the Voila lifecycle, completion flow, rerun behavior, upload/reset behavior, or visible warnings/errors, validate both the unit/frontend and e2e/browser layers.

## Current TODO: Known Failing Checks

- `tests/test_voila_frontend.py::TestSarGammaComparisonUI::test_stall_watchdog_surfaces_error_and_reenables_button`
- `tests/test_voila_frontend.py::TestSarGammaComparisonUI::test_handle_button_click_defers_start_until_kernel_io_loop_turn`
- `tests/test_voila_e2e.py::test_same_session_rerun_updates_results_after_power_change`
- `tests/test_voila_e2e.py::test_exact_repeat_shows_warning_without_rerunning`
- `tests/test_voila_e2e.py::test_uploading_new_data_clears_prior_results`

## Voila Workflow Guidance

- Keep backend execution off the UI thread, but keep widget mutation and final state transitions on the UI loop.
- Prefer one deterministic completion path from backend outcome to rendered UI state.
- If a change appears to fix backend logs but not visible UI state, reproduce it in the browser and verify the final widget state with Playwright before treating the change as complete.

## Architecture Recovery Plan (2026-05-03)

### Root cause
Backend can finish + logs complete, yet frontend never transitions. Worker thread leaks widget mutations instead of routing through one UI-loop-owned completion path.

### Baseline to preserve (staged)
- Thin notebook bootstrap in `voila.ipynb`
- Workspace/runtime discovery in `runtime.py`
- Backend JSON/error/log contract in `runner.py` + `workflow_cli.py`
- Visible frontend feedback on start and error

### Recovery sequence
1. Worker thread → produces result/failure only, no widget mutation
2. All widget mutation → one explicit UI-loop-owned path (`_dispatch_ui_update` or equivalent)
3. Progress-bar completion/failure → same lifecycle path, not separate background thread
4. Temporarily simplify run-IDs, stall watchdog, power-only reuse, server-reachability until base transition proven
5. Re-add each higher-level behavior on top of verified completion path
6. Tighten tests around exact "backend completed → frontend rendered → progress stopped → button re-enabled" seam
7. Add `voila_frontend/ARCHITECTURE.md`: bootstrap, runtime dirs, backend contract, frontend state machine, threading rules, persistence model, anti-patterns

### Anti-patterns (do not reintroduce)
- Widget mutation from worker thread as fallback when `_dispatch_ui_update` fails to land
- Progress bar state owned by a thread independent of the completion callback
- Tests that only verify mocked callback paths, not real Voila callback delivery

### Verification gates before PR merge
1. Narrow unit tests prove the chosen completion-path abstraction
2. Backend JSON/error/log behavior confirmed against staged baseline
3. Fresh Voila smoke flow: backend completes → visible tables/images
4. One compare-patterns cycle + one rerun cycle via e2e
5. Manual check: session restore and stale-server restart cannot leave UI silently hung

## Voila UI: What Broke vs What Fixed It
BREAKS
1. LFS download failure → git reset --hard exit 128 (sessions 77d270cb, d6052fa0)

Two separate LFS assets blocked uvx/osparc's git+https install:

assets/no-data-transparent.png (1.6KB) — tiny file, shouldn't be in LFS at all
data/database/dipole_1450MHz_Flat_10mm_10g.csv (1.1MB) — bigger, still in LFS
Root: uvx runs git reset --hard <commit> which triggers LFS smudge filters. No LFS creds available in the install environment → fatal.

2. Logs window broke after HTML widget swap (session d6052fa0)

Steps:

widgets.Output with overflow_y='auto' was silently ignored (traitlets deprecation) → visual overflow bug
Fix attempt: replaced with widgets.HTML + inline CSS → logs went empty
Root: HTML.value = ... sends a comm message that requires ipykernel's shell_parent context variable. Background threads don't have it → silent failure (caught by logging.Handler.handleError)
3. __init__.py importing voila_frontend (session d9c96ab8)

__init__.py re-exported from .voila_frontend which imports ipywidgets at module level. ipywidgets is only a dev dep — not in [project.dependencies]. When uvx installed the package, import chain failed → empty stdout → json.loads('') → JSONDecodeError. CLI was dead silently.

4. Merge dropped WorkflowResult + pydantic breakage (session 77d270cb)

Merge conflict resolution accidentally:

Removed WorkflowResult from workflow_config.py
Used field (dataclass) instead of Field (pydantic)
Dropped measurement_area_x_mm/y_mm fields from WorkflowConfig
Voila ran but workflows silently broke at completion/state stages.

5. ValidationError crash on notebook restore (session d6052fa0)

ui_state.json had stale image paths. state.py didn't guard against it → notebook threw unhandled ValidationError on startup.

FIXES (what made it work)
1a	PNG in LFS	Migrated out of LFS (git rm --cached, removed .gitattributes rule)	.gitattributes, assets/no-data-transparent.png
1b	CSV in LFS	GIT_LFS_SKIP_SMUDGE=1 injected into subprocess env	runner.py
2	Logs empty/broken	Back to widgets.Output but emit display_data HTML entries (not stream text) — Output has special thread-safe comm path	ui.py / OutputWidgetHandler
3	CLI dead (ipywidgets import)	Removed voila_frontend re-exports from __init__.py	src/__init__.py
4	WorkflowResult/pydantic	Restored WorkflowResult, field → Field, re-added measurement area fields	workflow_config.py, workflow_schema.py
5	Notebook crash on restore	state.py catches ValidationError + JSONDecodeError, returns None (fresh start)	state.py
Key insight
Most breaks were silent — LFS fails with exit 128 (no log in notebook), HTML.value= fails without exception, __init__.py import error swallowed before CLI could print. Diagnosing required running the E2E path directly, not just unit tests.

Hanging Progress Bar + Results Never Showing
Both stem from the same root cause: background threads lack ipykernel's shell_parent context variable.

## Progress bar hung / threw exception
Error (session d6052fa0, line 380):


Exception in thread Thread-4 (update_progress):
  File "ipykernel/ipkernel.py", line 788, in run_closure
    _threading_Thread_run(self)
  File ".../threading.py", in run
    self._target(*self._args, **self._kwargs)
update_progress ran in a background thread. Any widget.value = ... or widget.description = ... sends a comm message to the browser — ipykernel requires shell_parent context variable for that. Background thread doesn't have it → exception → thread dies → progress bar freezes.

Fix: capture context at thread creation, run inside it:


ctx = contextvars.copy_context()
thread = Thread(target=lambda: ctx.run(update_progress, ...))
Results never showing
Upstream of the progress bar: the workflow was failing before producing results because runner.py subprocess triggered LFS smudge → git reset --hard exit 128 → WorkflowResult never emitted → _on_run_complete never called → results widget never updated.

Fix was GIT_LFS_SKIP_SMUDGE=1 in runner's subprocess env. Once that landed, the workflow completed and _on_run_complete fired.

Same root cause, different manifestation
Symptom	Widget	Thread	Fix
Logs empty	widgets.HTML.value=	logging handler thread	Back to widgets.Output (has special comm path)
Progress bar hangs + exception	widgets.IntProgress.value=	Thread-4 (update_progress)	contextvars.copy_context()
Results never show	N/A — workflow never completes	subprocess	GIT_LFS_SKIP_SMUDGE=1 in runner.py
Today's session (d2a44556) is analyzing the staged vs unstaged state to understand which of these fixes landed in the clean branch vs which are still in flux — that's the "frontend/backend lifecycle diverged" framing.

Additional Fixes/Breaks Found
No logs showing in Voila (session 77d270cb — after LFS was fixed)
Once the LFS was fixed and the workflow actually ran, logs were still absent. Two-part root cause:

OutputWidgetHandler was only attached to the sar_pattern_validation.voila_frontend.ui module logger — runner.py used a different logger, so its output never reached the widget
result.stderr (the backend subprocess's workflow logs) was captured but silently dropped on success — only used in error messages
Fix (ui.py + runner.py):

Moved handler attachment from ui module logger → sar_pattern_validation.voila_frontend package logger. Both ui and runner are children, so logs propagate up automatically. propagate=False set to prevent double-printing to root handler
Added LOGGER to runner.py, forwarded every non-empty line of result.stderr through it after subprocess completes
runner.py reading wrong stream on subprocess failure (session d9c96ab8)
When the subprocess exited with an error, runner.py was reading from the wrong stream (stdout instead of stderr or vice versa) to get the error message. Fixed alongside the __init__.py import fix.