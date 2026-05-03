# Voila Frontend Architecture

## Files
- `runtime.py`: WorkspacePaths + workspace/runtime directory discovery
- `runner.py`: SarPatternValidationRunner — subprocess backend execution, real-time log streaming (threading.Thread + file tail), JSON result parsing, GIT_LFS_SKIP_SMUDGE=1 in env
- `models.py`: WorkflowResultPayload (extends WorkflowResult) — adds reference_file_path, measured_file_sha256 for power-only reuse detection
- `state.py`: load_or_migrate_ui_state / save_ui_state — persists to notebooks/system_state/ui_state.json; catches ValidationError + JSONDecodeError (returns None = fresh start)
- `ui.py`: SarGammaComparisonUI — main ipywidgets UI class
- `notebooks/voila.ipynb`: thin bootstrap, imports voila_frontend and calls bootstrap_voila_ui()

## Backend contract
- CLI: `sar-pattern-validation` (workflow_cli.py)
- Output: JSON to stdout: `{"status": "success", "result": {...}}` or `{"status": "error", "error": {...}}`
- Logs: streamed from backend log file to frontend logger in real-time via _stream_backend_log_file thread
- Error details in `result.stderr` are logged on both success and failure

## Key UI features (as of 2026-05-03)
- **Exact repeat detection**: if same reference + measured file + power level submitted again, shows warning, no rerun
- **Power-only reuse**: if same reference + measured file but different power level, recalculates psSAR without re-running backend (instant)
- **Stall watchdog**: if no log activity for N seconds (env: SAR_PATTERN_VALIDATION_RUN_STALL_TIMEOUT_S, default 60s), shows timeout error
- **Server connection monitor**: JS ping loop in browser shows banner if Voila server becomes unreachable
- **Session restore**: on page reload, restores previous inputs and results from ui_state.json

## Run lifecycle (correct pattern)
```
handle_button_click (kernel thread)
  → if exact repeat: show warning, return
  → if same dataset + different power: recalculate instantly, return
  → button.disabled = True, _prepare_for_new_run()
  → _start_progress_updater() [dedicated ctx-copied thread]
  → _start_stall_watchdog(button, run_id)
  → io_loop.call_later(0.2, _start_workflow_run, button, ...)

_start_workflow_run (kernel thread, via call_later)
  → ctx = contextvars.copy_context()
  → Thread(target=ctx.run, args=(_run_workflow_task,), kwargs={button, run_id, ...}).start()

_run_workflow_task (ctx-copied worker thread)
  → runner.run_workflow(..., on_log_activity=lambda: _mark_run_activity(run_id))
  → on success: _dispatch_ui_update(_handle_workflow_success, results, run_id)
  → on failure: _dispatch_ui_update(_handle_workflow_failure, message, button, run_id)
  → finally:    _dispatch_ui_update(_finish_workflow_run, button, run_id)
```

## Known failure seam history
- Timer-thread polling (`_schedule_ui_callback` + `_poll_run_completion`) was tried and failed: Timer threads lack kernel context, widget updates dropped silently. Removed. Never reintroduce.

## Recovery principles
1. Worker thread produces result/failure only — never mutates widgets directly
2. All widget mutation through `_dispatch_ui_update` on kernel thread
3. Progress updater = dedicated thread with `contextvars.copy_context()`
4. Tests that only verify mocked callback paths do NOT prove real Voila callback delivery — always run e2e

## Test infrastructure
- `completed_workspace` fixture: pre-baked demo run in `tests/fixtures/demo_run/` — <50ms, no subprocess
- Run unit: `uv run python -m pytest -v -o "addopts=" tests/test_voila_frontend.py`
- Run e2e: `uv run python -m pytest -v -o "addopts=" --run-e2e -p no:xdist tests/test_voila_e2e.py`
