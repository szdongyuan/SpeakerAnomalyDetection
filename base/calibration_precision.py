"""Shared display and submission precision for input calibration factors."""
import math


def format_calibration_factor(value):
    """Format a loaded or confirmed factor using the popup's six decimals."""
    return format(float(value), '.6f')


def normalize_calibration_factor(value):
    """Validate and normalize a newly submitted calibration factor."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('校准系数必须是大于零的有限数值')
    normalized = float(format_calibration_factor(value))
    if normalized <= 0:
        raise ValueError('六位小数精度下校准系数必须大于零')
    return normalized
