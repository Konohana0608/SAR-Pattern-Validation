from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sar_pattern_validation.sample_catalog import DatabaseSampleCatalog


def _serialize_catalog(catalog: DatabaseSampleCatalog) -> dict[str, Any]:
    return {
        "samples": catalog.to_dataframe().to_dict(orient="records"),
        "unique_entries": catalog.unique_entries_in_columns(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return sample catalog metadata as JSON for the Voila kernel shim."
    )
    parser.add_argument(
        "--database-path",
        required=True,
        help="Path to the reference database directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        catalog = DatabaseSampleCatalog.scan(Path(args.database_path))
        payload = {
            "status": "success",
            "catalog": _serialize_catalog(catalog),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc).strip() or "Catalog query failed.",
            },
        }
        print(json.dumps(payload, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
