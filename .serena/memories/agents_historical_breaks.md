# Historical Breaks & Root Causes (from AGENTS.md)

All breaks below were **silent** — no exception visible in notebook. Always diagnose via the E2E path (run voila, watch browser), not just unit tests.

## 1. LFS download failure → exit 128
- assets/no-data-transparent.png was in Git LFS → migrated out (git rm --cached, remove .gitattributes rule)
- data/database CSVs still in LFS → GIT_LFS_SKIP_SMUDGE=1 injected in runner.py subprocess env
- Root: uvx runs `git reset --hard <commit>` which triggers LFS smudge filters; no LFS creds in install env

## 2. Logs empty after HTML widget swap
- widgets.Output overflow_y='auto' silently ignored by traitlets deprecation → visual overflow bug
- Fix attempt: replaced Output with widgets.HTML + inline CSS → logs went empty
- Root: HTML.value= sends comm message requiring ipykernel's shell_parent context var; logging thread doesn't have it
- Fix: back to widgets.Output; emit display_data HTML entries (Output has thread-safe comm path)

## 3. CLI dead — ipywidgets import in __init__.py
- __init__.py re-exported from .voila_frontend which imports ipywidgets at module level
- ipywidgets is only a dev dep → import chain failed in uvx/production → empty stdout → JSONDecodeError
- Fix: removed voila_frontend re-exports from src/__init__.py

## 4. Merge dropped WorkflowResult + pydantic breakage
- Merge conflict resolution accidentally: removed WorkflowResult, used field (dataclass) instead of Field (pydantic), dropped measurement area fields
- Voila ran but workflows silently broke at completion/state stages
- Fix: restored WorkflowResult, field→Field, re-added measurement area fields (workflow_config.py, workflow_schema.py)

## 5. ValidationError crash on notebook restore
- ui_state.json had stale image paths → unhandled ValidationError on startup
- Fix: state.py catches ValidationError + JSONDecodeError, returns None (fresh start)

## 6. OutputWidgetHandler attached to wrong logger
- Handler only attached to ui module logger; runner.py used a different logger → runner logs never reached widget
- Fix: moved handler to sar_pattern_validation.voila_frontend package logger (parent); propagate=False

## 7. Backend stderr silently dropped on success
- result.stderr from subprocess captured but never logged on successful runs
- Fix: runner.py forwards every non-empty line of stderr through LOGGER after subprocess completes

## 8. Progress bar hung + exception (contextvars)
```
Exception in thread Thread-4 (update_progress):
    self._target(*self._args, **self._kwargs)
```
- Progress thread updated widget.value without copied context → ipykernel exception → thread dies → bar freezes
- Fix: ctx = contextvars.copy_context(); thread = Thread(target=ctx.run, args=(update_progress,))

## 9. Button never re-enables after workflow completes (2026-05-03)
- _schedule_ui_callback used threading.Timer for completion polling
- Timer threads lack kernel context → widget mutations from _poll_run_completion silently dropped
- Fix: restored _dispatch_ui_update pattern; worker dispatches directly with copied context (see agents_threading_and_dispatch)

## Symptom → root cause table
| Symptom | Widget/thing | Thread | Fix |
|---|---|---|---|
| Logs empty | widgets.HTML.value= | logging handler | Back to widgets.Output |
| Progress bar hangs | widgets.IntProgress.value= | Thread-4 (no ctx copy) | contextvars.copy_context() |
| Results never show | workflow never completes | subprocess | GIT_LFS_SKIP_SMUDGE=1 |
| Button never re-enables | all widget updates | Timer threads (no ctx) | _dispatch_ui_update from worker |
