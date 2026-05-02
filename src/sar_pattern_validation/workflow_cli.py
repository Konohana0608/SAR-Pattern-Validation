from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sar_pattern_validation.workflows import complete_workflow

_BACKEND_LOG_FILE_ENV = "SAR_PATTERN_VALIDATION_BACKEND_LOG_FILE"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _serialize(obj: Any) -> Any:
    """Recursively convert objects into JSON-serializable structures."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _configure_logging() -> None:
    """
    Ensure logs go to stderr (so stdout stays clean JSON).
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format=_LOG_FORMAT,
    )

    log_path_value = os.getenv(_BACKEND_LOG_FILE_ENV, "").strip()
    if not log_path_value:
        return

    log_path = Path(log_path_value)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
        ):
            return

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(file_handler)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint for sar-pattern-validation.

    - Parses CLI args (delegated to complete_workflow)
    - Runs workflow
    - Emits JSON result to stdout
    - Returns proper exit code
    """
    _configure_logging()

    try:
        # Pass argv through to your existing argparse logic
        args = argv if argv is not None else sys.argv[1:]

        result = complete_workflow(*args)

        # Convert result into JSON-safe structure
        payload = {
            "status": "success",
            "result": _serialize(result),
        }

        print(json.dumps(payload, indent=2))
        return 0

    except Exception as exc:
        logging.getLogger(__name__).exception("Workflow execution failed")

        error_payload = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc).strip() or "Workflow execution failed.",
            },
        }

        # Print error JSON to stdout (not stderr) so frontend can always parse stdout
        # Logs remain on stderr for debugging
        print(json.dumps(error_payload, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
