from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SARImageLoader",
    "Rigid2DRegistration",
    "Transform2D",
    "show_registration_overlay",
    "GammaMapEvaluator",
    "PlottingConfig",
    "WorkflowConfig",
    "WorkflowResult",
    "DatabaseSampleCatalog",
    "DatabaseSampleFilters",
]

_EXPORTS = {
    "SARImageLoader": ("sar_pattern_validation.image_loader", "SARImageLoader"),
    "Rigid2DRegistration": (
        "sar_pattern_validation.registration2d",
        "Rigid2DRegistration",
    ),
    "Transform2D": ("sar_pattern_validation.registration2d", "Transform2D"),
    "show_registration_overlay": (
        "sar_pattern_validation.plotting",
        "show_registration_overlay",
    ),
    "GammaMapEvaluator": ("sar_pattern_validation.gamma_eval", "GammaMapEvaluator"),
    "PlottingConfig": ("sar_pattern_validation.workflow_config", "PlottingConfig"),
    "WorkflowConfig": ("sar_pattern_validation.workflow_config", "WorkflowConfig"),
    "WorkflowResult": ("sar_pattern_validation.workflow_config", "WorkflowResult"),
    "DatabaseSampleCatalog": (
        "sar_pattern_validation.sample_catalog",
        "DatabaseSampleCatalog",
    ),
    "DatabaseSampleFilters": (
        "sar_pattern_validation.sample_catalog",
        "DatabaseSampleFilters",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)
