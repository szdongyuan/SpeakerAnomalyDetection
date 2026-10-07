"""Exact input selection policy, independent of native library ownership."""
from types import SimpleNamespace

import pytest

from base.recording_input_device import resolve_input_device, validate_input_device


def backend_for(devices, api_names=("Selected API", "Other API")):
    apis = [{"name": name} for name in api_names]
    return SimpleNamespace(
        query_devices=lambda index=None: devices if index is None else next(
            device for device in devices if device["index"] == index),
        query_hostapis=lambda index=None: apis if index is None else apis[index])


@pytest.fixture
def selected():
    return dict(index=7, name="Full microphone name", hostapi=1,
                hostapi_name="Selected API", max_input_channels=3)


def test_resolve_changes_only_local_numbers_and_ignores_output_only(selected):
    original = selected.copy()
    backend = backend_for([{**selected, "index": 2, "hostapi": 0},
                           {**selected, "index": 3, "hostapi": 0, "max_input_channels": 0}])
    resolved = resolve_input_device(backend, selected, (2, 0))
    assert resolved == {**selected, "index": 2, "hostapi": 0}
    assert selected == original


def test_duplicate_identity_cannot_be_disambiguated_by_channel_capacity(selected):
    backend = backend_for([{**selected, "hostapi": 0},
                           {**selected, "index": 2, "hostapi": 0, "max_input_channels": 1}])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_input_device(backend, selected, (2,))


def test_matching_explicit_index_does_not_enumerate_duplicates(selected):
    backend = backend_for([{**selected, "hostapi": 0}, {**selected, "index": 2, "hostapi": 0}])
    assert validate_input_device(backend, selected, (2,))["index"] == 7


def test_legacy_request_never_remaps_to_another_slot(selected):
    selected.pop("hostapi_name")
    backend = backend_for([{**selected, "name": "other microphone"}, {**selected, "index": 2}])
    with pytest.raises(ValueError, match="identity changed at index 7"):
        resolve_input_device(backend, selected, (2,))


def test_legacy_request_keeps_strict_numeric_api(selected):
    selected.pop("hostapi_name")
    with pytest.raises(ValueError, match="identity changed at index 7: hostapi"):
        resolve_input_device(backend_for([{**selected, "hostapi": 0}]), selected, (2,))
