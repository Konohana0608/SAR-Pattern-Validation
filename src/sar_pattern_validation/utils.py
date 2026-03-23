from __future__ import annotations

import logging
from pathlib import Path

import SimpleITK as sitk

NOISY_LOGGER_LEVELS = {
    "matplotlib": logging.WARNING,
}


def ensure_output_path(path_to_convert: str | Path | None) -> Path | None:
    if path_to_convert is None:
        return None
    path = Path(path_to_convert)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def image_extent_mm(img: sitk.Image) -> tuple[float, float, float, float]:
    sx, sy = img.GetSpacing()
    ox, oy = img.GetOrigin()
    w, h = img.GetSize()
    xmin, xmax = sorted((ox - 0.5 * sx, ox + sx * (w - 0.5)))
    ymin, ymax = sorted((oy - 0.5 * sy, oy + sy * (h - 0.5)))
    return (1000 * xmin, 1000 * xmax, 1000 * ymin, 1000 * ymax)


def configure_root_logging(level: str) -> None:
    resolved = getattr(logging, level, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=resolved,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(resolved)
    for logger_name, logger_level in NOISY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(logger_level)
