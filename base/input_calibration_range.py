"""Read-only voltage range for the input calibration result display."""
from collections.abc import Mapping

from base.ve3668n_input import voltage_limits_for_range
from consts.ve3668n_consts import VE_RANGE_INDEX_CONFIG_KEY


def resolve_test_full_scale_peak_volts(sequence_config):
    if not isinstance(sequence_config, list) or not sequence_config:
        return None
    detail = sequence_config[0]
    for key in ("seq1", "acq", "detail"):
        if not isinstance(detail, Mapping):
            return None
        detail = detail.get(key)
    if not isinstance(detail, Mapping):
        return None
    try:
        _, maximum = voltage_limits_for_range(detail.get(VE_RANGE_INDEX_CONFIG_KEY, 0))
    except ValueError:
        return None
    return maximum
