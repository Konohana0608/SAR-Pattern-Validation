"""Tests for generate_measurement_validation_report_html.py dashboard rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_measurement_validation_report_html import (
    _format_validation_issues_html,
    _generate_html,
)


def _make_case(
    *,
    case_id: str = "test_case_1",
    status: str = "error",
    validation_issues: list[dict] | None = None,
    frequency_mhz: int = 900,
    frequency_key: str = "900mhz",
    frequency_label: str = "900 MHz",
    power_level_dbm: float = 23.0,
    distance_mm: int = 15,
    averaging_mass: str = "1g",
    evaluated_pixel_count: int = 0,
    failed_pixel_count: int = 0,
    pass_rate_percent: float | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "status": status,
        "validation_issues": validation_issues or [],
        "frequency_mhz": frequency_mhz,
        "frequency_key": frequency_key,
        "frequency_label": frequency_label,
        "power_level_dbm": power_level_dbm,
        "distance_mm": distance_mm,
        "averaging_mass": averaging_mass,
        "evaluated_pixel_count": evaluated_pixel_count,
        "failed_pixel_count": failed_pixel_count,
        "pass_rate_percent": pass_rate_percent,
    }


class TestFormatValidationIssuesHtml:
    def test_empty_issues_returns_empty_string(self) -> None:
        assert _format_validation_issues_html({"validation_issues": []}) == ""

    def test_missing_key_returns_empty_string(self) -> None:
        assert _format_validation_issues_html({}) == ""

    def test_renders_error_issue_with_code_and_message(self) -> None:
        case = {
            "validation_issues": [
                {
                    "severity": "error",
                    "code": "CSV_FORMAT_INVALID",
                    "message": "CSV could not be parsed.",
                }
            ]
        }
        html = _format_validation_issues_html(case)
        assert "CSV_FORMAT_INVALID" in html
        assert "CSV could not be parsed." in html
        assert "class='issue error'" in html

    def test_renders_warning_severity_class(self) -> None:
        case = {
            "validation_issues": [
                {
                    "severity": "warning",
                    "code": "MASK_TOO_SMALL",
                    "message": "Mask is small.",
                }
            ]
        }
        html = _format_validation_issues_html(case)
        assert "class='issue warning'" in html


class TestDashboardRendersAllIssueCodes:
    """Verify each ValidationIssue code appears in the rendered HTML with correct styling."""

    ISSUE_CODES = [
        ("CSV_FORMAT_INVALID", "error", "CSV format invalid: bad header"),
        (
            "MEASUREMENT_AREA_OUT_OF_BOUNDS",
            "error",
            "Measurement area is out of bounds.",
        ),
        ("MASK_TOO_SMALL", "error", "Gamma evaluation mask is too small."),
        ("NOISE_FLOOR_OUT_OF_BOUNDS", "error", "Noise floor is out of bounds."),
    ]

    @pytest.mark.parametrize("code,severity,message", ISSUE_CODES)
    def test_issue_code_renders_in_dashboard(
        self, code: str, severity: str, message: str
    ) -> None:
        case = _make_case(
            case_id=f"case_{code}",
            status="error",
            validation_issues=[
                {"severity": severity, "code": code, "message": message}
            ],
        )
        reports = [
            {
                "path": Path("fake.json"),
                "name": "fake.json",
                "generated_at": "2026-05-04",
                "report": {"cases": [case]},
            }
        ]
        html = _generate_html(reports, [case])

        assert code in html
        assert message in html
        assert f"class='issue {severity}'" in html

    def test_multiple_issues_per_case_all_render(self) -> None:
        issues = [
            {"severity": "error", "code": code, "message": msg}
            for code, _, msg in self.ISSUE_CODES
        ]
        case = _make_case(
            case_id="multi_issue_case",
            status="error",
            validation_issues=issues,
        )
        reports = [
            {
                "path": Path("fake.json"),
                "name": "fake.json",
                "generated_at": "2026-05-04",
                "report": {"cases": [case]},
            }
        ]
        html = _generate_html(reports, [case])

        for code, _, msg in self.ISSUE_CODES:
            assert code in html
            assert msg in html
