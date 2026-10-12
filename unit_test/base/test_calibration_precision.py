"""Six-decimal display and newly submitted factor boundaries."""
import math

import pytest

from base.calibration_precision import format_calibration_factor, normalize_calibration_factor


@pytest.mark.parametrize("value,expected,text", [
    (38.645595158142605, 38.645595, "38.645595"),
    (2.5, 2.5, "2.500000"),
    (0.000001, 0.000001, "0.000001"),
    (9.9999996, 10.0, "10.000000"),
    (1.2345674, 1.234567, "1.234567"),
    (1.2345676, 1.234568, "1.234568"),
    (math.nextafter(1.2345675, -math.inf), 1.234567, "1.234567"),
    (math.nextafter(1.2345675, math.inf), 1.234568, "1.234568"),
])
def test_six_decimal_format_and_normalization(value, expected, text):
    assert format_calibration_factor(value) == text
    assert normalize_calibration_factor(value) == expected
    assert normalize_calibration_factor(expected) == expected
    assert format_calibration_factor(expected) == text


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, "1e9999"])
def test_invalid_factors_are_rejected(value):
    with pytest.raises(ValueError, match="大于零的有限数值"):
        normalize_calibration_factor(value)


@pytest.mark.parametrize("value", [0.0000001, 2.5e-23, 5e-324])
def test_new_factor_rounding_to_zero_is_rejected(value):
    with pytest.raises(ValueError, match="六位小数精度下校准系数必须大于零"):
        normalize_calibration_factor(value)


def test_historical_tiny_factor_can_be_displayed():
    assert format_calibration_factor(5e-324) == "0.000000"
