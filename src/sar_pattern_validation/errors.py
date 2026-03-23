class SarPatternValidationError(Exception):
    """Base exception for the SAR Pattern Validation package."""


class CsvFormatError(SarPatternValidationError):
    """Raised when a SAR CSV cannot be parsed or lacks required fields."""


class ConfigValidationError(SarPatternValidationError):
    """Raised when workflow configuration is invalid."""


class WorkflowExecutionError(SarPatternValidationError):
    """Raised when workflow execution fails."""
