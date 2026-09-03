"""Compatibility imports for shared manual analysis-limit helpers."""

from base.analysis_limit_evaluation import (
    ManualLimitValidationError,
    limits_from_constant_values,
    limits_from_manual_config,
    limits_from_manual_segments,
    normalize_segments,
    validate_constant_limit_config,
    validate_manual_limit_config,
    validate_manual_segments,
)


__all__ = [
    "ManualLimitValidationError",
    "limits_from_constant_values",
    "limits_from_manual_config",
    "limits_from_manual_segments",
    "normalize_segments",
    "validate_constant_limit_config",
    "validate_manual_limit_config",
    "validate_manual_segments",
]
