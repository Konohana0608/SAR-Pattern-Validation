from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

REPORT_PROPERTY_NAME = "measurement_validation_case"


class MeasurementValidationReportCollector:
    def __init__(self) -> None:
        self.case_results: list[dict[str, Any]] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return

        payload: dict[str, Any] | None = None
        for name, value in report.user_properties:
            if name != REPORT_PROPERTY_NAME:
                continue
            payload = json.loads(value)  # type: ignore
            break

        if payload is None:
            return

        failure_message = None
        if report.failed:
            failure_message = str(report.longrepr)

        payload["status"] = report.outcome
        payload["pytest"] = {
            "nodeid": report.nodeid,
            "outcome": report.outcome,
            "duration_seconds": report.duration,
            "failure_message": failure_message,
        }
        self.case_results.append(payload)


def _build_frequency_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for case in cases if case["status"] == "passed")
    failed = sum(1 for case in cases if case["status"] == "failed")
    errors = sum(1 for case in cases if case["status"] == "error")
    total_failed_pixels = sum(int(case["failed_pixel_count"]) for case in cases)
    total_evaluated_pixels = sum(int(case["evaluated_pixel_count"]) for case in cases)

    return {
        "case_count": len(cases),
        "passed_case_count": passed,
        "failed_case_count": failed,
        "error_case_count": errors,
        "total_failed_pixels": total_failed_pixels,
        "total_evaluated_pixels": total_evaluated_pixels,
        "aggregate_pass_rate_percent": (
            100.0
            * (total_evaluated_pixels - total_failed_pixels)
            / total_evaluated_pixels
            if total_evaluated_pixels
            else None
        ),
    }


def _build_power_group_summary(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case["power_level_key"]), []).append(case)

    summaries = []
    for power_level_key, grouped_cases in sorted(
        grouped.items(), key=lambda entry: float(entry[1][0]["power_level_dbm"])
    ):
        exemplar = grouped_cases[0]
        summaries.append(
            {
                "group_key": exemplar["group_key"],
                "power_level_key": power_level_key,
                "power_level_dbm": exemplar["power_level_dbm"],
                "summary": _build_frequency_summary(grouped_cases),
                "case_ids": [case["case_id"] for case in grouped_cases],
            }
        )
    return summaries


def _build_report(
    *,
    cases: list[dict[str, Any]],
    output_path: Path,
    pytest_args: list[str],
    exit_code: int,
    duration_seconds: float,
    regenerate_artifacts: bool,
    save_plots: bool,
) -> dict[str, Any]:
    ordered_cases = sorted(
        cases,
        key=lambda case: (
            int(case["frequency_mhz"]),
            float(case["power_level_dbm"]),
            int(case["distance_mm"]),
            str(case["averaging_mass"]),
            str(case["case_id"]),
        ),
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in ordered_cases:
        grouped.setdefault(str(case["frequency_key"]), []).append(case)

    frequencies = []
    for frequency_key, grouped_cases in grouped.items():
        exemplar = grouped_cases[0]
        frequencies.append(
            {
                "frequency_key": frequency_key,
                "frequency_label": exemplar["frequency_label"],
                "frequency_mhz": exemplar["frequency_mhz"],
                "summary": _build_frequency_summary(grouped_cases),
                "power_groups": _build_power_group_summary(grouped_cases),
                "case_ids": [case["case_id"] for case in grouped_cases],
            }
        )

    passed_cases = sum(1 for case in ordered_cases if case["status"] == "passed")
    failed_cases = sum(1 for case in ordered_cases if case["status"] == "failed")
    error_cases = sum(1 for case in ordered_cases if case["status"] == "error")
    total_failed_pixels = sum(int(case["failed_pixel_count"]) for case in ordered_cases)
    total_evaluated_pixels = sum(
        int(case["evaluated_pixel_count"]) for case in ordered_cases
    )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "run": {
            "pytest_args": pytest_args,
            "exit_code": exit_code,
            "duration_seconds": duration_seconds,
            "regenerate_artifacts": regenerate_artifacts,
            "save_plots": save_plots,
            "python_version": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "tmpdir": os.environ.get("TMPDIR"),
        },
        "summary": {
            "case_count": len(ordered_cases),
            "passed_case_count": passed_cases,
            "failed_case_count": failed_cases,
            "error_case_count": error_cases,
            "frequency_count": len(frequencies),
            "total_failed_pixels": total_failed_pixels,
            "total_evaluated_pixels": total_evaluated_pixels,
            "aggregate_pass_rate_percent": (
                100.0
                * (total_evaluated_pixels - total_failed_pixels)
                / total_evaluated_pixels
                if total_evaluated_pixels
                else None
            ),
        },
        "frequencies": frequencies,
        "cases": ordered_cases,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run measurement validation pytest suite and write a JSON report."
    )
    parser.add_argument(
        "--output",
        default="tests/artifacts/measurement_validation/measurement_validation_report.json",
        help="Path to the JSON report file.",
    )
    parser.add_argument(
        "--regenerate-artifacts",
        action="store_true",
        help="Regenerate measurement validation artifacts during the pytest run.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save measurement validation plots during the pytest run.",
    )
    parser.add_argument(
        "--rerun",
        action="append",
        dest="rerun_frequencies",
        metavar="FREQ",
        help="Force rerun of a specific frequency (e.g., '900mhz', '2450mhz'). Can be specified multiple times.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to pytest after the default test target.",
    )
    return parser.parse_args(argv)


def _build_smart_test_filter(
    repo_root: Path, rerun_frequencies: list[str] | None
) -> str | None:
    """
    Determine which frequencies need to run based on existing artifacts.
    Returns a pytest -k filter expression, or None to run all tests.
    """
    artifact_dir = repo_root / "tests" / "artifacts" / "measurement_validation"
    if not artifact_dir.exists():
        # No artifacts yet, run all tests
        return None

    expected_group_counts = _discover_expected_group_counts(repo_root)
    if not expected_group_counts:
        return None

    rerun_set = _expand_rerun_targets(
        set(rerun_frequencies or []), set(expected_group_counts)
    )

    existing_group_counts = _discover_existing_group_counts(artifact_dir)

    if not existing_group_counts:
        # No complete artifacts found, run all tests
        return None

    missing_groups = {
        group_key
        for group_key, expected_count in expected_group_counts.items()
        if existing_group_counts.get(group_key, 0) < expected_count
    }

    if rerun_frequencies:
        groups_to_run = rerun_set | missing_groups
    else:
        groups_to_run = missing_groups

    if not groups_to_run and existing_group_counts:
        print(
            "All measurement groups have existing artifacts. Use --rerun FREQ or FREQ_DBM to force rerun."
        )
        return "no_test_id"  # Disable all tests

    if groups_to_run:
        # Build pytest -k filter
        filter_expr = " or ".join(
            f"test_measurement_workflow_cases_{group_key}"
            for group_key in sorted(groups_to_run)
        )
        return filter_expr

    return None


def _format_group_key(frequency_mhz: int, power_level_dbm: int) -> str:
    return f"{frequency_mhz}mhz_{power_level_dbm}dbm"


def _discover_expected_group_counts(repo_root: Path) -> dict[str, int]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    test_module = importlib.import_module("tests.test_measurement_validation")
    cases = test_module.CASES
    return dict(Counter(case.group_key for case in cases))


def _discover_existing_group_counts(artifact_dir: Path) -> dict[str, int]:
    existing: dict[str, int] = {}
    for frequency_dir in artifact_dir.iterdir():
        if not frequency_dir.is_dir() or frequency_dir.name in ("plots", "logs"):
            continue
        for power_dir in frequency_dir.iterdir():
            if not power_dir.is_dir() or not power_dir.name.endswith("dbm"):
                continue
            existing[f"{frequency_dir.name}_{power_dir.name}"] = len(
                list(power_dir.glob("*_metrics.json"))
            )
    return existing


def _expand_rerun_targets(targets: set[str], expected_groups: set[str]) -> set[str]:
    expanded = set()
    for target in targets:
        if target in expected_groups:
            expanded.add(target)
            continue
        expanded.update(
            group for group in expected_groups if group.startswith(f"{target}_")
        )
    return expanded


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmpdir = repo_root / ".tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Build smart test filter if not regenerating all artifacts
    test_filter = None
    if not args.regenerate_artifacts:
        test_filter = _build_smart_test_filter(repo_root, args.rerun_frequencies)

    previous_env = {
        "TMPDIR": os.environ.get("TMPDIR"),
        "REGENERATE_MEASUREMENT_VALIDATION_ARTIFACTS": os.environ.get(
            "REGENERATE_MEASUREMENT_VALIDATION_ARTIFACTS"
        ),
        "SAVE_MEASUREMENT_VALIDATION_PLOTS": os.environ.get(
            "SAVE_MEASUREMENT_VALIDATION_PLOTS"
        ),
    }

    os.environ["TMPDIR"] = str(tmpdir)
    os.environ["REGENERATE_MEASUREMENT_VALIDATION_ARTIFACTS"] = (
        "1" if args.regenerate_artifacts else "0"
    )
    os.environ["SAVE_MEASUREMENT_VALIDATION_PLOTS"] = "1" if args.save_plots else "0"

    collector = MeasurementValidationReportCollector()
    forwarded_args = list(args.pytest_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    pytest_args = [str(repo_root / "tests" / "test_measurement_validation.py")]
    if test_filter:
        pytest_args.extend(["-k", test_filter])
    pytest_args.extend(forwarded_args)

    started_at = time.perf_counter()
    try:
        exit_code = int(pytest.main(pytest_args, plugins=[collector]))
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    duration_seconds = time.perf_counter() - started_at
    report = _build_report(
        cases=collector.case_results,
        output_path=output_path,
        pytest_args=pytest_args,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        regenerate_artifacts=args.regenerate_artifacts,
        save_plots=args.save_plots,
    )
    output_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(
        json.dumps({"report_path": str(output_path), "exit_code": exit_code}, indent=2)
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
