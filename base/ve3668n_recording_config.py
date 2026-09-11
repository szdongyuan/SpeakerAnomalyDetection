"""Resolve queue-owned VK acquisition parameters without SDK or storage I/O."""
from collections.abc import Mapping

from base.ve3668n_input import (
    create_input_config, validate_sample_rate, validate_range_index, validate_device_snapshot,
)
from consts.ve3668n_consts import VE_DEFAULT_SAMPLE_RATE, VE_RANGE_INDEX_CONFIG_KEY


def resolve_ve_recording_config(detail, *, fallback_profile=None):
    """Return an owned config; explicit invalid values never become defaults."""
    if not isinstance(detail, Mapping):
        raise ValueError("recording detail must be a mapping")
    if "sample_rate" in detail:
        rate = validate_sample_rate(detail["sample_rate"])
    elif fallback_profile is None:
        rate = VE_DEFAULT_SAMPLE_RATE
    else:
        if not isinstance(fallback_profile, Mapping) or "sample_rate" not in fallback_profile:
            raise ValueError("fallback profile requires sample_rate")
        rate = validate_sample_rate(fallback_profile["sample_rate"])
    index = validate_range_index(detail.get(VE_RANGE_INDEX_CONFIG_KEY, 0))
    return create_input_config(rate, range_index=index)


def resolve_ve_recording_device(device, detail, *, fallback_profile=None):
    """Copy device identity and validate it with the queue's acquisition config."""
    if not isinstance(device, Mapping):
        raise ValueError("device must be a mapping")
    config = resolve_ve_recording_config(detail, fallback_profile=fallback_profile)
    return validate_device_snapshot({**device, "input_config": config})
