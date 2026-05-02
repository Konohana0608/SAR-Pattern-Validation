import json
import logging
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from sar_pattern_validation.workflow_cli import main


@pytest.mark.slow
def test_cli_main_function_with_measured_data(tmp_path: Path) -> None:
    """
    Test the main() function directly with measured data files from Git LFS.
    Uses actual CSV files from data/database/ directory.
    Uses same distance but different averaging (1g vs 10g) for realistic testing.
    Note: Requires git lfs pull to fetch the actual files.
    """
    # Locate database data files (stored in Git LFS)
    # Use files from the same measurement distance but different averaging volumes
    # This ensures spatial overlap and similar distributions
    repo_root = Path(__file__).parent.parent
    measured_file = repo_root / "data" / "database" / "dipole_2450MHz_Flat_10mm_1g.csv"
    reference_file = (
        repo_root / "data" / "database" / "dipole_2450MHz_Flat_10mm_10g.csv"
    )

    # Check if files exist and are not LFS pointers
    if not measured_file.exists() or not reference_file.exists():
        pytest.skip("Database data files not found")

    # Check if LFS files have been pulled (not just pointers)
    for file in [measured_file, reference_file]:
        with open(file) as f:
            first_line = f.readline()
            if first_line.startswith("version https://git-lfs.github.com"):
                pytest.skip("Git LFS files not pulled. Run: git lfs pull")

    # Set up output paths in tmp directory
    measured_image_path = tmp_path / "measured_image.png"
    reference_image_path = tmp_path / "reference_image.png"
    aligned_image_path = tmp_path / "aligned_measured.png"
    registered_image_path = tmp_path / "registered_measured.png"
    gamma_comparison_image_path = tmp_path / "gamma_comparison.png"

    # Build CLI arguments
    argv = [
        "--measured_file_path",
        str(measured_file),
        "--reference_file_path",
        str(reference_file),
        "--reference_image_save_path",
        str(reference_image_path),
        "--measured_image_save_path",
        str(measured_image_path),
        "--aligned_meas_save_path",
        str(aligned_image_path),
        "--registered_image_save_path",
        str(registered_image_path),
        "--gamma_comparison_image_path",
        str(gamma_comparison_image_path),
        "--power_level_dbm",
        "23.0",
    ]

    # Capture stdout
    import io
    from contextlib import redirect_stdout

    captured_output = io.StringIO()

    with redirect_stdout(captured_output):
        exit_code = main(argv)

    # Verify exit code
    assert exit_code == 0, "CLI should exit with code 0 on success"

    # Parse JSON output
    output_text = captured_output.getvalue()
    result = json.loads(output_text)

    # Verify JSON structure
    assert result["status"] == "success"
    assert "result" in result

    # Verify result contains expected fields
    workflow_result = result["result"]
    assert "pass_rate_percent" in workflow_result
    assert "evaluated_pixel_count" in workflow_result
    assert "passed_pixel_count" in workflow_result
    assert "failed_pixel_count" in workflow_result
    assert "measured_pssar" in workflow_result
    assert "reference_pssar" in workflow_result
    assert "scaling_error" in workflow_result

    # Verify numeric fields are reasonable
    assert 0.0 <= workflow_result["pass_rate_percent"] <= 100.0
    assert workflow_result["evaluated_pixel_count"] > 0
    assert (
        workflow_result["passed_pixel_count"] + workflow_result["failed_pixel_count"]
        == workflow_result["evaluated_pixel_count"]
    )

    # Verify output files were created
    assert gamma_comparison_image_path.exists(), (
        "Gamma comparison image should be saved"
    )
    assert registered_image_path.exists(), "Registered overlay image should be saved"


@pytest.mark.slow
def test_cli_main_function_with_example_data(tmp_path: Path) -> None:
    """
    Test the main() function directly with example data files.
    Uses actual CSV files from data/example/ directory.
    """
    # Locate example data files
    repo_root = Path(__file__).parent.parent
    measured_file = repo_root / "data" / "example" / "measured_sSAR1g.csv"
    reference_file = repo_root / "data" / "example" / "reference_sSAR1g.csv"

    if not measured_file.exists() or not reference_file.exists():
        pytest.skip("Example data files not found")

    # Set up output paths in tmp directory
    gamma_output = tmp_path / "gamma_comparison.png"
    registered_output = tmp_path / "registered.png"

    # Build CLI arguments
    argv = [
        "--measured_file_path",
        str(measured_file),
        "--reference_file_path",
        str(reference_file),
        "--gamma_comparison_image_path",
        str(gamma_output),
        "--registered_image_save_path",
        str(registered_output),
        "--power_level_dbm",
        "30.0",
        "--dose_to_agreement",
        "5.0",
        "--distance_to_agreement",
        "2.0",
    ]

    # Capture stdout
    import io
    from contextlib import redirect_stdout

    captured_output = io.StringIO()

    with redirect_stdout(captured_output):
        exit_code = main(argv)

    # Verify exit code
    assert exit_code == 0, "CLI should exit with code 0 on success"

    # Parse JSON output
    output_text = captured_output.getvalue()
    result = json.loads(output_text)

    # Verify JSON structure
    assert result["status"] == "success"
    assert "result" in result

    # Verify result contains expected fields
    workflow_result = result["result"]
    assert "pass_rate_percent" in workflow_result
    assert "evaluated_pixel_count" in workflow_result
    assert "passed_pixel_count" in workflow_result
    assert "failed_pixel_count" in workflow_result
    assert "measured_pssar" in workflow_result
    assert "reference_pssar" in workflow_result
    assert "scaling_error" in workflow_result

    # Verify numeric fields are reasonable
    assert 0.0 <= workflow_result["pass_rate_percent"] <= 100.0
    assert workflow_result["evaluated_pixel_count"] > 0
    assert (
        workflow_result["passed_pixel_count"] + workflow_result["failed_pixel_count"]
        == workflow_result["evaluated_pixel_count"]
    )

    # Verify output files were created
    assert gamma_output.exists(), "Gamma comparison image should be saved"
    assert registered_output.exists(), "Registered overlay image should be saved"


@pytest.mark.slow
@pytest.mark.integration
def test_cli_via_subprocess(tmp_path: Path) -> None:
    """
    Test the CLI as it would be invoked from a subprocess (like in the frontend).
    This tests the actual entry point defined in pyproject.toml.
    """
    # Locate example data files
    repo_root = Path(__file__).parent.parent
    measured_file = repo_root / "data" / "example" / "measured_sSAR1g.csv"
    reference_file = repo_root / "data" / "example" / "reference_sSAR1g.csv"

    if not measured_file.exists() or not reference_file.exists():
        pytest.skip("Example data files not found")

    # Set up output paths
    gamma_output = tmp_path / "gamma_via_subprocess.png"

    # Run the CLI via subprocess
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sar_pattern_validation.workflow_cli",
            "--measured_file_path",
            str(measured_file),
            "--reference_file_path",
            str(reference_file),
            "--gamma_comparison_image_path",
            str(gamma_output),
            "--power_level_dbm",
            "30.0",
        ],
        capture_output=True,
        text=True,
    )

    # Verify subprocess succeeded
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Parse JSON from stdout
    output_data = json.loads(result.stdout)
    assert output_data["status"] == "success"
    assert "result" in output_data

    # Verify output file exists
    assert gamma_output.exists()


def test_cli_error_handling(tmp_path: Path) -> None:
    """
    Test that CLI returns proper error JSON when given invalid inputs.
    Error JSON is printed to stdout (logs stay on stderr).
    """
    # Use non-existent file paths
    argv = [
        "--measured_file_path",
        str(tmp_path / "nonexistent_measured.csv"),
        "--reference_file_path",
        str(tmp_path / "nonexistent_reference.csv"),
    ]

    import io
    from contextlib import redirect_stdout

    captured_stdout = io.StringIO()

    with redirect_stdout(captured_stdout):
        exit_code = main(argv)

    # Should return non-zero exit code
    assert exit_code != 0, "CLI should return non-zero exit code on error"

    # Parse error JSON from stdout (not stderr)
    stdout_text = captured_stdout.getvalue()
    error_output = json.loads(stdout_text)
    assert error_output["status"] == "error"
    assert "error" in error_output
    assert "type" in error_output["error"]
    assert "message" in error_output["error"]


def test_cli_logs_traceback_and_returns_sanitized_error_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured_stdout = __import__("io").StringIO()

    with (
        patch(
            "sar_pattern_validation.workflow_cli.complete_workflow",
            side_effect=RuntimeError("Bad measurement input"),
        ),
        caplog.at_level(logging.ERROR),
        redirect_stdout(captured_stdout),
    ):
        exit_code = main([])

    payload = json.loads(captured_stdout.getvalue())

    assert exit_code == 1
    assert payload == {
        "status": "error",
        "error": {
            "type": "RuntimeError",
            "message": "Bad measurement input",
        },
    }
    assert "Workflow execution failed" in caplog.text
    assert "Traceback" in caplog.text


def test_cli_writes_backend_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "backend-test.log"
    captured_stdout = __import__("io").StringIO()

    with (
        patch(
            "sar_pattern_validation.workflow_cli.complete_workflow",
            side_effect=RuntimeError("Bad measurement input"),
        ),
        patch.dict(
            os.environ,
            {"SAR_PATTERN_VALIDATION_BACKEND_LOG_FILE": str(log_path)},
            clear=False,
        ),
        redirect_stdout(captured_stdout),
    ):
        exit_code = main([])

    assert exit_code == 1
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "Workflow execution failed" in log_text
    assert "Traceback" in log_text


@pytest.mark.slow
@pytest.mark.integration
def test_cli_via_uvx_like_frontend(tmp_path: Path) -> None:
    """
    Test the CLI as it would be invoked from the frontend using uvx.
    This mimics the exact subprocess.run pattern used in the frontend.
    Requires uvx to be installed: pip install uv
    """
    # Check if uvx is available
    import shutil

    if not shutil.which("uvx"):
        pytest.skip("uvx not found. Install with: pip install uv")

    # Use the same setup as test_cli_main_function_with_measured_data
    repo_root = Path(__file__).parent.parent
    measured_file = repo_root / "data" / "database" / "dipole_900MHz_Flat_15mm_1g.csv"
    reference_file = repo_root / "data" / "database" / "dipole_900MHz_Flat_15mm_10g.csv"

    # Check if files exist and are not LFS pointers
    if not measured_file.exists() or not reference_file.exists():
        pytest.skip("Database data files not found")

    # Check if LFS files have been pulled
    for file in [measured_file, reference_file]:
        with open(file) as f:
            first_line = f.readline()
            if first_line.startswith("version https://git-lfs.github.com"):
                pytest.skip("Git LFS files not pulled. Run: git lfs pull")

    # Set up output paths in tmp directory (same as measured_data test)
    measured_image_path = tmp_path / "measured_image.png"
    reference_image_path = tmp_path / "reference_image.png"
    aligned_image_path = tmp_path / "aligned_measured.png"
    registered_image_path = tmp_path / "registered_measured.png"
    gamma_comparison_image_path = tmp_path / "gamma_comparison.png"

    # Build command exactly like frontend does
    pkg_path = repo_root  # Use repo root as package path

    cmd = [
        "uvx",
        "--no-cache",
        "--from",
        str(pkg_path),
        "sar-pattern-validation",
        "--measured_file_path",
        str(measured_file),
        "--reference_file_path",
        str(reference_file),
        "--reference_image_save_path",
        str(reference_image_path),
        "--measured_image_save_path",
        str(measured_image_path),
        "--aligned_meas_save_path",
        str(aligned_image_path),
        "--registered_image_save_path",
        str(registered_image_path),
        "--gamma_comparison_image_path",
        str(gamma_comparison_image_path),
        "--power_level_dbm",
        "23.0",
    ]

    # Run the command exactly like frontend
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse result like frontend does - always read JSON from stdout
    # Return code distinguishes success vs error, but JSON is always on stdout
    output_data = json.loads(result.stdout)

    if result.returncode != 0:
        pytest.fail(f"CLI failed with return code {result.returncode}: {output_data}")

    # Verify the response
    assert output_data["status"] == "success"
    assert "result" in output_data

    # Verify result contains expected fields
    workflow_result = output_data["result"]
    assert "pass_rate_percent" in workflow_result
    assert "evaluated_pixel_count" in workflow_result
    assert "passed_pixel_count" in workflow_result
    assert "failed_pixel_count" in workflow_result
    assert "measured_pssar" in workflow_result
    assert "reference_pssar" in workflow_result
    assert "scaling_error" in workflow_result

    # Verify numeric fields are reasonable
    assert 0.0 <= workflow_result["pass_rate_percent"] <= 100.0
    assert workflow_result["evaluated_pixel_count"] > 0

    # Verify output files were created
    assert gamma_comparison_image_path.exists(), (
        "Gamma comparison image should be saved"
    )
    assert registered_image_path.exists(), "Registered overlay image should be saved"
    assert measured_image_path.exists(), "Measured image should be saved"
    assert reference_image_path.exists(), "Reference image should be saved"
