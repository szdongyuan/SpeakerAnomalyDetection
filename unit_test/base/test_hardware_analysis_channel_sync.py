import json
from copy import deepcopy

import pytest

from base.hardware_analysis_channel_sync import (
    recording_input_selection_key,
    synchronize_product_analysis_channels,
)
from base.load_config import LoadUiConfig
from unit_test.base.test_product_test_config_refresh import make_refresh_manager


def read_queue(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("channels", [[0, 1, 2, 3, 4], [1], [7, 2, 7]])
def test_hardware_selection_replaces_analysis_channels_and_preserves_parameters(tmp_path, channels):
    manager, _, path = make_refresh_manager(tmp_path, condition_count=1, port_count=1)
    data = read_queue(path)
    analysis = data[0]["seq1"]["analysis_list"]
    for kind in ("SPL", "Spec", "FBA", "FFT", "LP", "LOUD"):
        analysis[kind] = {
            "type": kind, "analysis_channel": 0, "analysis_channels": [0, 1],
            "limit_checked": True, "custom_parameter": {"limit": 42},
        }
    analysis["RSC"] = {"type": "RSC", "custom_parameter": 17}
    LoadUiConfig.save_sequence_config_to_json(data, str(path))

    snapshot = synchronize_product_analysis_channels(manager, "A.json", channels)

    saved = read_queue(path)
    expected = deepcopy(data)
    for kind in ("SPL", "Spec", "FBA", "FFT", "LP", "LOUD"):
        item = expected[0]["seq1"]["analysis_list"][kind]
        item["analysis_channels"] = sorted(set(channels))
        item["analysis_channel"] = min(channels)
    assert saved == expected
    assert snapshot.queue_catalog["test"]["data"] == saved


def test_all_active_product_queues_are_updated_once_and_unreferenced_queue_is_untouched(tmp_path, monkeypatch):
    manager, project, path = make_refresh_manager(tmp_path, condition_count=1, port_count=1)
    second = path.with_name("second.json")
    unused = path.with_name("unused.json")
    for target in (second, unused):
        LoadUiConfig.save_sequence_config_to_json(read_queue(path), str(target))
    LoadUiConfig.save_data_to_json(
        {"test": str(path), "alias": str(path), "second": str(second), "unused": str(unused)},
        manager.queue_registry_path,
    )
    conditions = project["test_groups"][0]["test_conditions"]
    for name in ("alias", "second"):
        conditions.append({"condition_name": name, "test_queue": name, "trigger_state": ""})
    assert manager.save_project("A.json", project)[0]
    unused_before = unused.read_bytes()
    original_save = LoadUiConfig.save_sequence_config_to_json
    writes = []

    def save(data, target):
        writes.append(target)
        return original_save(data, target)

    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", save)
    synchronize_product_analysis_channels(manager, "A.json", [0, 2, 4])

    assert len(writes) == 2
    for target in (path, second):
        assert read_queue(target)[0]["seq1"]["analysis_list"]["SPL"]["analysis_channels"] == [0, 2, 4]
    assert unused.read_bytes() == unused_before
    synchronize_product_analysis_channels(manager, "A.json", [0, 2, 4])
    assert len(writes) == 2


def test_no_active_product_does_not_modify_saved_queues(tmp_path):
    manager, _, path = make_refresh_manager(tmp_path)
    before = path.read_bytes()
    assert synchronize_product_analysis_channels(manager, None, [0, 1]).active_file == ""
    assert path.read_bytes() == before


def test_legacy_queue_without_acquisition_mode_uses_recording_default(tmp_path):
    manager, _, path = make_refresh_manager(tmp_path)
    data = read_queue(path)
    del data[0]["seq1"]["acq"]["mode"]
    LoadUiConfig.save_sequence_config_to_json(data, str(path))

    synchronize_product_analysis_channels(manager, "A.json", [1, 3])

    assert read_queue(path)[0]["seq1"]["analysis_list"]["SPL"]["analysis_channels"] == [1, 3]


def test_retry_finishes_all_queues_after_one_atomic_write_fails(tmp_path, monkeypatch):
    manager, project, path = make_refresh_manager(tmp_path, condition_count=1, port_count=1)
    second = path.with_name("second.json")
    LoadUiConfig.save_sequence_config_to_json(read_queue(path), str(second))
    LoadUiConfig.save_data_to_json(
        {"test": str(path), "second": str(second)}, manager.queue_registry_path
    )
    project["test_groups"][0]["test_conditions"].append(
        {"condition_name": "second", "test_queue": "second", "trigger_state": ""}
    )
    assert manager.save_project("A.json", project)[0]
    save = LoadUiConfig.save_sequence_config_to_json
    attempts = []

    def fail_second(data, target):
        attempts.append(target)
        return False if len(attempts) == 2 else save(data, target)

    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", fail_second)
    with pytest.raises(OSError, match="分析通道同步失败"):
        synchronize_product_analysis_channels(manager, "A.json", [0, 2, 4])

    snapshot = synchronize_product_analysis_channels(manager, "A.json", [0, 2, 4])
    assert len(attempts) == 3
    for queue in snapshot.queue_catalog.values():
        assert queue["analysis_list"]["SPL"]["analysis_channels"] == [0, 2, 4]


def test_invalid_channels_leave_queue_unchanged(tmp_path):
    manager, _, path = make_refresh_manager(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        synchronize_product_analysis_channels(manager, "A.json", [])
    assert path.read_bytes() == before


def test_write_failure_is_reported_without_claiming_fresh_snapshot(tmp_path, monkeypatch):
    manager, _, path = make_refresh_manager(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *_: False)
    with pytest.raises(OSError, match="分析通道同步失败"):
        synchronize_product_analysis_channels(manager, "A.json", [0, 1, 2])
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
