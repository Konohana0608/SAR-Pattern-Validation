# SAR Pattern Validation — Specification

## §G Goal

Recover UI polish + full measurement-validation toolbox from legacy dev branches onto `main-melanie`; regenerate all validation artifacts under current codebase; keep strict py3.9 Voila-frontend / uvx-py3.10+ backend separation throughout.

## §C Constraints

C1: Frontend (`notebooks/voila.ipynb`, `voila_frontend/ui.py`) runs in jupyter-math container → **python 3.9**. No walrus, no match-statement, no 3.10+ stdlib. All UI cherry-picks must pass `ty` type-check with py3.9 target.

C2: Backend (`workflows.py`, `plotting.py`, `report.py`, CLI) runs via **uvx python 3.10+**. Typing and language features at 3.10+ are fine there.

C3: Artifact regeneration runs against current `main-melanie` HEAD (not `develop` / legacy branches). Artifacts must be committed to LFS after regen.

C4: HTML report is pure Python (Jinja2 or stdlib string templates). No `pdflatex` runtime dependency for HTML path; LaTeX/PDF path stays as an optional CLI flag.

C5: LFS scope: measurement CSVs under `data/measurements/` and `data/database/`, artifact `.npz` under `tests/artifacts/`, plot PNGs under `tests/artifacts/measurement_validation/plots/`. All must remain LFS-tracked.

## §I Interface

I1: `complete_workflow(measured_file_path, reference_file_path, ...)` — runs full registration + gamma pipeline; returns `WorkflowResult` on success, raises `WorkflowExecutionError` with `.issue: ValidationIssue | None` on structured failures.

I2: `ValidationIssue.code` is a machine-readable string (e.g. `MASK_TOO_SMALL`, `EMPTY_MEASURED_MASK`, `CSV_FORMAT_ERROR`). The Voila UI surfaces `issue.message` in the error/warning banner.

I3: `noise_floor` (W/kg) is the SAR floor below which pixels are excluded from the metric mask. Range [0, 0.1]. Must satisfy `noise_floor < measured_peak` for registration to proceed.

I4 (planned): `generate_html_report(results: list[CaseResult], output_path: Path)` — renders filterable HTML table of all measurement-validation cases with pass/fail, gamma pass-rate, and inline thumbnail links.

## §V Invariants

V1: ∀ registration call → `fixed_mask` active pixel count ≥ 1, else raise `ValidationIssue(code="EMPTY_MEASURED_MASK")` before `Execute()`. Applies at `workflows.py` after `make_metric_masks()`.

V2: ∀ `WorkflowExecutionError` raised inside `_complete_workflow` → `.issue` is preserved through exception handlers (no re-wrapping by generic `except Exception` clause). Applied via `except WorkflowExecutionError: raise` as first handler.

V3: `_apply_roi_policy` in `workflows.py` must receive `measured_mask_u8` (SAR ≥ noise cutoff, built by `loader.make_metric_masks()`) as its `measured_mask_u8` arg — never `measured_support_u8` (boundary-only). Gamma eval mask must exclude sub-cutoff (noise-filtered) pixels. Fix: `workflows.py:311`.

V4: ∀ MASK_TOO_SMALL condition (pre-registration on `measured_mask_u8` or post-registration on `evaluator.evaluation_mask`) → raises `WorkflowExecutionError` with `severity="error"` and `code="MASK_TOO_SMALL"`; workflow stops at the first failing check. Pre-registration check fires before `Rigid2DRegistration.run()`.

V5: ∀ cherry-picked frontend commit → must not introduce any python ≥ 3.10 syntax or imports. CI `ty` check with `--python-version 3.9` is the enforcement gate.

V6: ∀ artifact regeneration run → artifacts are committed to LFS and the commit message references the `main-melanie` HEAD hash used. Regen must not silently overwrite passing cases with failures without a §B backprop entry.

## §T Tasks

Stream A — UI adjustments branch (`jgo/ui-adjustments` from `main-melanie`):

| ID | Status | Task | Cites |
|----|--------|------|-------|
| T1 | x | Create `jgo/ui-adjustments` from `main-melanie` HEAD | |
| T2 | x | Cherry-pick plotting renames + overlays from `develop`: `12cdd09` (Simulated→Reference title), `9746b05` (cropped-area dark-gray overlay + legend), `d24c9d8` (noise-floor medium-gray overlay all 6 panels) | C1,C2,V5 |
| T3 | . | Cherry-pick notebook layout from `develop`: `d774c11` (table below, center plots), `aed839e` (inline banner, swap tables, drop pass/fail button), `264c2d6` (stretch antenna grid to right-column height) | C1,V5 |
| T4 | . | Port boxed log widget + radio-button height limit from `86d7889` (`6.3-noise-floor`); verify py3.9 compat; test scrollable output widget in voila | C1,V5 |
| T5 | . | Run full test suite on `jgo/ui-adjustments`; fix any failures; open PR → `main-melanie` | V5 |

Stream B — Measurement validation toolbox (`main-melanie` direct or sub-branch):

| ID | Status | Task | Cites |
|----|--------|------|-------|
| T7 | . | Recover additional measurement CSVs (1950 / 5800 / 900 MHz bands) + `data/database/` reference CSVs from `develop` or "main"; verify LFS tracking | C5 |
| T8 | . | Extend `test_measurement_validation.py` with recovered bands; add `MeasurementValidationCase` entries for each new dataset | C2,C5 |
| T9 | . | Recover scripts to generate measurement validation HTML report by frequency band and various filtering | C2,C4,I4 |
| T10 | . | Regenerate all `tests/artifacts/measurement_validation/` (`.npz` + `_metrics.json` + plot PNGs) under `main-melanie` HEAD with `REGENERATE_MEASUREMENT_VALIDATION_ARTIFACTS=1 SAVE_MEASUREMENT_VALIDATION_PLOTS=1` | C3,C5,V6 |
| T11 | . | Run HTML report over regenerated artifacts; document which cases pass / fail / regress vs `develop` baseline; backprop any new failures via §B | V6,I4 |

## §M Merge Log

Records every branch merged into `main-melanie`. Critical for squash-merge workflows: a squash-merge rewrites the tip hash, so once a PR is squash-merged the original branch tip listed here is the only reliable way to know what content was included.

| Date | Branch | Tip at merge | What it brought |
|------|--------|-------------|-----------------|
| 2026-05-15 | `jgo/6.6-validation-issue-channel` | `497c17c` | Task 6.6: `ValidationIssue` dataclass + `MASK_TOO_SMALL` / `CSV_FORMAT_ERROR` emit sites; notebook issues-aware banner; banner stdout fix + `status:error` guard; `MASK_TOO_SMALL` E2E test; backprop §B1–§B4, §V1–§V4 |
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
| B4 | 2026-05-15 | MASK_TOO_SMALL checked only post-registration; pre-registration noise-filtered `measured_mask_u8` never verified against `min_inscribed_square_mm` | V4 |
| B5 | 2026-05-15 | MASK_TOO_SMALL emitted as `severity="warning"` appended to `issues`, allowing workflow to complete; physically it is a hard validity gate — comparison on a sub-22mm mask is invalid | V4 |
