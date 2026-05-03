# Threading Rules & Completion Dispatch (from AGENTS.md)

## The law: all widget mutation on the kernel thread

- Backend execution: always in a background thread
- Widget mutation and final state transitions: always on the kernel/UI-loop thread
- One deterministic completion path from backend outcome to rendered UI state

## Correct pattern: _dispatch_ui_update from contextvars-copied worker thread

```python
# 1. Copy context at thread creation (captures shell_parent and all kernel context vars)
ctx = contextvars.copy_context()
self._workflow_thread = threading.Thread(
    target=ctx.run,
    args=(self._run_workflow_task,),
    kwargs={"button": button, "run_id": run_id, ...},
)
self._workflow_thread.start()

# 2. Worker calls _dispatch_ui_update on completion — no direct widget mutation
def _run_workflow_task(self, *, button, run_id, ...):
    try:
        results = self.runner.run_workflow(...)
        self._dispatch_ui_update(self._handle_workflow_success, results=results, run_id=run_id)
    except Exception as error:
        self._dispatch_ui_update(self._handle_workflow_failure, message=str(error), button=button, run_id=run_id)
    finally:
        self._dispatch_ui_update(self._finish_workflow_run, button=button, run_id=run_id)
```

## _dispatch_ui_update behaviour
- On main/kernel thread: runs callback directly
- In background thread with io_loop available: `io_loop.add_callback(callback)` + blocks until done
- In background thread without io_loop: runs directly in worker (context copy makes this safe)
- Timeout fallback (10s): logs warning, runs callback directly in worker

## Progress bar: dedicated thread with copied context
```python
ctx = contextvars.copy_context()
self._progress_thread = threading.Thread(target=ctx.run, args=(update_progress,), daemon=True)
```
Join the thread in `_stop_progress_updater` before updating bar_style or clearing output.

## Stall watchdog: dispatch via _dispatch_ui_update
Watchdog thread (plain thread, no context copy) calls `_dispatch_ui_update` for failure/finish.
Works because `io_loop.add_callback` is thread-safe from any thread.

## Anti-patterns — never reintroduce
1. `threading.Timer` for completion dispatch — Timer threads lack kernel context; widget mutations silently dropped
2. `_pending_run_outcome` queue + `_poll_run_completion` polling via timers — same root cause, failed silently in production
3. Any widget `.value =` assignment from a thread not started with `contextvars.copy_context()`
4. `widgets.HTML.value =` from a logging handler thread — comm message requires shell_parent; use `widgets.Output` instead
5. `_schedule_ui_callback` pattern (removed) — Timer-based scheduling bypasses context; never bring back

## Why these fail silently
ipykernel requires the `shell_parent` context variable to route comm messages back to the browser.
Background threads created without `contextvars.copy_context()` don't have it.
Widget assignments from such threads reach `logging.Handler.handleError` with no visible traceback —
the only symptom is a frozen UI.
