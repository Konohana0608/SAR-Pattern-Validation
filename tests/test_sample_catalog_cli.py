from __future__ import annotations

import json
from pathlib import Path

from sar_pattern_validation.sample_catalog_cli import main


def test_sample_catalog_cli_returns_json_payload(tmp_path: Path, capsys) -> None:
    database_path = tmp_path / "database"
    database_path.mkdir()
    (database_path / "dipole_900MHz_Flat_15mm_1g.csv").write_text(
        "x,y,sar\n0,0,1\n", encoding="utf-8"
    )
    (database_path / "patch_2450MHz_Flat_5mm_10g.csv").write_text(
        "x,y,sar\n0,0,1\n", encoding="utf-8"
    )

    exit_code = main(["--database-path", str(database_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "success"
    assert len(payload["catalog"]["samples"]) == 2
    assert payload["catalog"]["unique_entries"]["Antenna Type"] == ["dipole", "patch"]
    assert payload["catalog"]["unique_entries"]["Frequency [MHz]"] == [900.0, 2450.0]


def test_sample_catalog_cli_returns_structured_error(tmp_path: Path, capsys) -> None:
    database_path = tmp_path / "database"
    database_path.mkdir()
    (database_path / "unexpected.csv").write_text("x,y,sar\n0,0,1\n", encoding="utf-8")

    exit_code = main(["--database-path", str(database_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "ValueError"
