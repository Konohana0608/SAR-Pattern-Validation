# SAR Pattern Validation — Specification

## §I Interface

I1: `complete_workflow(measured_file_path, reference_file_path, ...)` — runs full registration + gamma pipeline; returns `WorkflowResult` on success, raises `WorkflowExecutionError` with `.issue: ValidationIssue | None` on structured failures.

I2: `ValidationIssue.code` is a machine-readable string (e.g. `MASK_TOO_SMALL`, `EMPTY_MEASURED_MASK`, `CSV_FORMAT_ERROR`). The Voila UI surfaces `issue.message` in the error/warning banner.

I3: `noise_floor` (W/kg) is the SAR floor below which pixels are excluded from the metric mask. Range [0, 0.1]. Must satisfy `noise_floor < measured_peak` for registration to proceed.

## §V Invariants

V1: ∀ registration call → `fixed_mask` active pixel count ≥ 1, else raise `ValidationIssue(code="EMPTY_MEASURED_MASK")` before `Execute()`. Applies at `workflows.py` after `make_metric_masks()`.

V2: ∀ `WorkflowExecutionError` raised inside `_complete_workflow` → `.issue` is preserved through exception handlers (no re-wrapping by generic `except Exception` clause). Applied via `except WorkflowExecutionError: raise` as first handler.

V3: `_apply_roi_policy` in `workflows.py` must receive `measured_mask_u8` (SAR ≥ noise cutoff, built by `loader.make_metric_masks()`) as its `measured_mask_u8` arg — never `measured_support_u8` (boundary-only). Gamma eval mask must exclude sub-cutoff (noise-filtered) pixels. Fix: `workflows.py:311`.

## §M Merge Log

Records every branch merged into `main-melanie`. Critical for squash-merge workflows: a squash-merge rewrites the tip hash, so once a PR is squash-merged the original branch tip listed here is the only reliable way to know what content was included.

| Date | Branch | Tip at merge | What it brought |
|------|--------|-------------|-----------------|
| 2026-05-15 | `jgo/6.6-validation-issue-channel` | `0f40141` | Task 6.6: `ValidationIssue` dataclass + `MASK_TOO_SMALL` / `CSV_FORMAT_ERROR` emit sites; notebook issues-aware banner; banner stdout fix + `status:error` guard; `MASK_TOO_SMALL` E2E test; backprop `EMPTY_MEASURED_MASK` guard + `except WorkflowExecutionError: raise` fix (§B1, §B2) |
| 2026-05-15 | `jgo/m6-results-table` | `b84e9a8` | M6 Task 5: two-table results layout in Voila notebook; widget notation fix; CI Voila E2E timeout + dependency updates |

Branches already incorporated before this log began (via GitHub PRs, squash-merged onto `main` / `main-melanie`):

| PR | Commit on main-melanie | What it brought |
|----|----------------------|-----------------|
| #18 | `4514399` | Bump actions/checkout 4→6 |
| #17 | `28ede53` | Bump actions/upload-artifact 4→7 |
| #16 | `b44255c` | Update measurement-validation test artifacts after registration direction change |
| #15 | `7b3c702` | User-configurable noise floor input (0 ≤ noise_floor ≤ 0.1 W/kg) |
| #13 | `61c3454` | Task 6.5: inscribed 22×22 mm square mask validity check |
| #8  | `fd5e4c3` | Vectorise gamma; `--output-dir`; lock deps; lint+type CI job; GitLFS for E2E |
| #7  | `ad1595d` | Task 6.4: feedback banners |
| #6  | `ea30322` | Task 6.1: reverse registration direction (gamma in measured frame) |
| #5  | `cf15668` | Scan-for-buttons grid fix; parallel CI stages |
| #4  | `7e33b2a` | Voila E2E Playwright test suite |
| #3  | `69d2bb7` | Run on oSPARC compatibility |
| #1  | `cc77226` | Fix numerical errors in CI |

## §B Bug Log

| ID | Date | Root cause | Invariant |
|----|------|-----------|-----------|
| B1 | 2026-05-15 | `noise_floor ≥ measured peak` → empty fixed mask → `VirtualSampledPointSet must have 1 or more points` crash in SimpleITK, surfaced as raw ITK traceback in Voila banner | V1 |
| B2 | 2026-05-15 | `_complete_workflow` generic `except Exception` handler re-wrapped `WorkflowExecutionError` raised from inside the `try` block, discarding `.issue` | V2 |
| B3 | 2026-05-15 | `workflows.py:311` passes `measured_support_u8` (boundary-only) to `_apply_roi_policy` instead of `measured_mask_u8`; noise-filtered (SAR < cutoff) pixels included in gamma eval mask → inflated pass rate (Task 6.4) | V3 |
