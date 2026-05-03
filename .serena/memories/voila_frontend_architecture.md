# Voila Frontend Architecture

## Files
- `runtime.py`: WorkspacePaths + workspace/runtime directory discovery
- `runner.py`: SarPatternValidationRunner — subprocess backend execution, log streaming, JSON result parsing
- `models.py`: UiState (pydantic), WorkflowResultPayload (pydantic)
- `state.py`: load_or_migrate_ui_state / save_ui_state — persists to notebooks/system_state/ui_state.json
- `ui.py`: SarGammaComparisonUI — main ipywidgets UI class (~900+ lines)
- `notebooks/voila.ipynb`: thin bootstrap, imports voila_frontend and calls bootstrap_voila_ui()

## Backend contract
- CLI entrypoint: `sar-pattern-validation` (workflow_cli.py)
- Outputs: JSON result payload + error field + log lines to stderr/file
- WorkflowResultPayload parsed from subprocess stdout

## Known failure seam (as of 2026-05-03)
- Backend can finish and logs complete, but frontend never transitions to "completed" state
- Root cause: widget mutation from worker thread (or fallback path in _dispatch_ui_update) bypasses UI loop ownership
- Fix direction: worker thread produces result/failure only → single UI-loop-owned completion path for all widget mutation

## Anti-patterns (do not reintroduce)
- Widget mutation from worker thread as fallback when _dispatch_ui_update fails
- Progress bar state owned by a thread independent of the completion callback
- Tests that only verify mocked callback paths without real Voila callback delivery

## Recovery plan (Milestone 6)
1. Freeze staged baseline
2. Refactor run lifecycle: worker result-only, UI loop owns widgets
3. Move progress-bar onto same path
4. Simplify run-IDs/watchdog/reuse until proven
5. Re-add higher-level behaviors incrementally
6. Tighten tests for "backend done → UI rendered" seam
7. Add voila_frontend/ARCHITECTURE.md
