"""Queue-owned VK parameters resolve without device or storage I/O."""
from collections import UserDict
from types import MappingProxyType

import pytest

from base.ve3668n_recording_config import resolve_ve_recording_config
from unit_test.base.ve3668n_fakes import input_config


def test_queue_values_override_profile_without_mutation():
    detail = {"sample_rate": 96000, "ve_range_index": 4}
    profile = input_config(51200)
    result = resolve_ve_recording_config(detail, fallback_profile=profile)
    assert result == input_config(96000, range_min=-0.5, range_max=0.5)
    result["sample_rate"] = 8000
    assert detail == {"sample_rate": 96000, "ve_range_index": 4}
    assert profile == input_config(51200)


@pytest.mark.parametrize("bad", [-1, 7, True, False, 1.0, 1.5, "1", None])
def test_explicit_bad_range_is_not_defaulted(bad):
    with pytest.raises(ValueError, match="ve_range_index"):
        resolve_ve_recording_config({"sample_rate": 48000, "ve_range_index": bad})


@pytest.mark.parametrize("bad", [7999, 102401, 48000.5, 48000.0, True, "48000", None])
def test_explicit_bad_rate_is_not_replaced_by_profile_or_default(bad):
    for profile in (None, input_config(51200)):
        with pytest.raises(ValueError, match="sample_rate"):
            resolve_ve_recording_config({"sample_rate": bad}, fallback_profile=profile)


def test_missing_parameters_use_profile_then_historical_defaults():
    assert resolve_ve_recording_config({}) == input_config(51200)
    assert resolve_ve_recording_config({}, fallback_profile=input_config(32000)) == input_config(32000)
    assert resolve_ve_recording_config({"ve_range_index": 6}) == input_config(
        51200, range_min=-0.02, range_max=0.02)
    assert resolve_ve_recording_config({}, fallback_profile=input_config(
        8000, range_min=-5, range_max=5)) == input_config(8000)


@pytest.mark.parametrize("profile", [None, {}, [], "bad", {"sample_rate": None}])
def test_explicit_queue_rate_does_not_read_unrelated_profile(profile):
    assert resolve_ve_recording_config({"sample_rate": 96000}, fallback_profile=profile) == input_config(96000)


@pytest.mark.parametrize("profile", [{}, [], "bad", {"sample_rate": None},
                                     {"sample_rate": 102401}, {"sample_rate": True}])
def test_needed_malformed_profile_is_not_defaulted(profile):
    with pytest.raises(ValueError, match="sample_rate"):
        resolve_ve_recording_config({}, fallback_profile=profile)


@pytest.mark.parametrize("detail", [None, [], "", 48000])
def test_detail_requires_a_mapping(detail):
    with pytest.raises(ValueError, match="detail.*mapping"):
        resolve_ve_recording_config(detail)


@pytest.mark.parametrize("wrap", [dict, UserDict, MappingProxyType])
def test_mapping_snapshots_and_unrelated_queue_fields_are_supported(wrap):
    detail = wrap({"ve_range_index": 1, "duration": 10, "monitor_playback": True})
    profile = wrap(input_config(102400))
    assert resolve_ve_recording_config(detail, fallback_profile=profile) == input_config(
        102400, range_min=-5, range_max=5)
