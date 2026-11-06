from copy import deepcopy

import pytest

from base.hardware_analysis_channel_sync import (
    recording_input_selection_key, synchronize_product_analysis_channels,
    update_sequence_analysis_channels,
)
from base.product_test_config_refresh import ProductTestConfigurationError
from unit_test.base.test_product_test_config_refresh import make_refresh_manager


@pytest.mark.parametrize("basis", [None, "wav"])
def test_hardware_sync_preserves_wav_positions_and_only_updates_physical_queues(basis):
    sequence = {"acq": {"mode": "RECORD_ONLY"}, "analysis_list": {
        "SPL": {"type": "SPL", "analysis_channel": 0, "analysis_channels": [0], "limit": 72}}}
    if basis:
        sequence["analysis_channel_basis"] = basis
    data = [{"seq1": sequence}]
    expected = deepcopy(data)
    update_sequence_analysis_channels(data, [2, 7])
    if basis == "wav":
        assert data == expected
    else:
        assert sequence["analysis_list"]["SPL"] == {
            "type": "SPL", "analysis_channel": 2, "analysis_channels": [2, 7], "limit": 72}


def test_station_configuration_blocks_legacy_hardware_sync_without_writes(tmp_path):
    manager, project, path = make_refresh_manager(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ProductTestConfigurationError, match="尚未"):
        synchronize_product_analysis_channels(manager, project["project_name"] + ".json", [0, 7])
    assert path.read_bytes() == before


def test_no_active_product_does_not_modify_saved_queues(tmp_path):
    manager, _, path = make_refresh_manager(tmp_path)
    before = path.read_bytes()
    assert synchronize_product_analysis_channels(manager, None, [0, 1]).active_file == ""
    assert path.read_bytes() == before


def test_input_identity_ignores_profile_refresh_but_detects_device_switch():
    device = {"backend": "vkinging", "machine_id": "first", "available": False}
    original = recording_input_selection_key(device, [0, 2])
    assert recording_input_selection_key({**device, "available": True}, [2, 0]) == original
    assert recording_input_selection_key({**device, "machine_id": "second"}, [0, 2]) != original
    assert recording_input_selection_key(device, [0, 2, 4]) != original
    soundcard = {"name": "mic", "hostapi": 0, "index": 1}
    assert recording_input_selection_key(soundcard, [0]) == recording_input_selection_key(
        {**soundcard, "max_input_channels": 8}, [0]
    )
    assert recording_input_selection_key(soundcard, [0]) != recording_input_selection_key(
        {**soundcard, "index": 3}, [0]
    )
    assert recording_input_selection_key(soundcard, [0]) != recording_input_selection_key(
        {**soundcard, "hostapi": 1}, [0]
    )
