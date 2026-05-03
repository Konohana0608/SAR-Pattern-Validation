# Task Completion Checklist

When a coding task is done, run in this order:

1. **Lint**: `make lint` (ruff check — fix any errors before continuing)
2. **Format**: `make format` (ruff format)
3. **Type check**: `make typecheck` (ty)
4. **Tests**: `make tests` or `make tests-fast` for quick iteration
5. **Voila smoke** (if voila_frontend changed): `make voila-smoke`
6. **Pre-commit**: `make run-pre-commit` (runs on staged files before commit)

## For voila_frontend changes specifically
- Run `tests/test_voila_frontend.py` (unit/mock layer)
- Run `tests/test_voila_e2e.py` (browser layer — needs voila server)
- Verify in real browser: backend completion → tables/images visible → progress stopped → button re-enabled
- Do NOT mark complete if only mocked tests pass — real callback delivery must be verified

## Known failing tests (2026-05-03, branch jgo/feedback-changes-clean)
- `test_stall_watchdog_surfaces_error_and_reenables_button`
- `test_handle_button_click_defers_start_until_kernel_io_loop_turn`
- `test_voila_e2e.py::test_same_session_rerun_updates_results_after_power_change`
- `test_voila_e2e.py::test_exact_repeat_shows_warning_without_rerunning`
- `test_voila_e2e.py::test_uploading_new_data_clears_prior_results`
