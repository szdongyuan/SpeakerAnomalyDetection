"""Format finite configuration numbers without rounding their stored values."""

from decimal import Decimal


def format_config_number(value):
    """Keep the round-trip representation, omitting only an integral .0 suffix."""
    return str(value).removesuffix(".0")


def config_number_decimals(value, minimum):
    """Return enough decimal places to restore a saved value in a spin box."""
    return max(minimum, -Decimal(str(value)).as_tuple().exponent)
