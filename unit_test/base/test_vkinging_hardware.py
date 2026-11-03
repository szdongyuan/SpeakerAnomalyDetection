"""vkinging persistence uses the original hardware schema, with no native I/O."""
import json
import sqlite3
from types import MappingProxyType
from contextlib import contextmanager

import pytest

from base.db_manager import DataSave
from base.hardware_management import (
    HardwareManagementRepository, HardwareValidationError, HardwareRuntimeMatchError, match_runtime_device,
)
from consts import error_code
from base.soundcard_calibration_manager import (
    save_mic_channel_v2pa_factor, save_manual_mic_channel_v2pa_factor,
    load_mic_channel_v2pa_factors, clear_mic_channel_v2pa_factors,
)
import base.sound_device_manager as device_module
from base.sound_device_manager import SoundDeviceManager
from base.hardware_management import augment_runtime_device


@pytest.fixture
def repository(tmp_path):
    path = tmp_path / "system.db"
    data_save = DataSave(str(path))
    try:
        code, message = data_save.create_system_tables()
        assert code == error_code.OK, message
    finally:
        data_save.close()
    return HardwareManagementRepository(str(path))


def device(machine_id="machine-A", name="DAQ", channels=(0, 8, 15, 31)):
    return {
        "hostapi_name": "vkinging", "hardware_id": machine_id, "name": name,
        "address": "192.0.2.1", "model": "DAQ", "physical_channels": channels,
        "max_input_channels": max(channels) + 1,
    }


def register(repository, runtime=None, rate=48000):
    return repository.register_asset(runtime or device(), "vkinging", "Lab DAQ", rate)


def schema(path):
    with sqlite3.connect(path) as connection:
        master = connection.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()
        columns = {
            table: connection.execute(f"PRAGMA table_info({table})").fetchall()
            for table in ("hardware_assets", "hardware_channel_calibrations")
        }
    return master, columns


def conditions(limit=2.5):
    return {"input_mode": "IEPE", "unit": "V", "range_min": -limit, "range_max": limit}


@pytest.mark.parametrize("channels", [tuple(range(16)), tuple(range(32)), (0, 8, 15, 31)])
def test_registration_uses_machine_id_and_real_physical_rows_without_schema_changes(repository, channels):
    before = schema(repository.db_path)
    asset = register(repository, device(channels=channels))
    assert asset["hardware_id"] == "machine-A"
    assert asset["hostapi_name"] == "vkinging"
    assert asset["max_input_channels"] == max(channels) + 1
    assert asset["max_output_channels"] == 0
    assert [row["channel_index"] for row in repository.list_channels("machine-A")] == list(channels)
    assert all(row["factor_value"] is None for row in repository.list_channels("machine-A"))
    assert schema(repository.db_path) == before


def test_reconnect_and_duplicate_registration_keep_identity_and_calibration(repository):
    asset = register(repository)
    repository.update_asset_fields(asset["hardware_id"], {"samplerate": 51200, "bit_depth": 64})
    repository.update_mic_channel_calibrations(asset["hardware_id"], {15: 2.123456789}, {15: 94},
                                              channel_conditions={15: conditions()})
    before = repository.list_channels(asset["hardware_id"])
    renamed = device(name="DAQ-renamed")
    again = register(repository, renamed)
    assert again["hardware_id"] == asset["hardware_id"]
    assert again["samplerate"] == 51200
    assert again["bit_depth"] == 64
    assert repository.get_asset(asset["hardware_id"])["device_name"] == "DAQ-renamed"
    assert repository.list_channels(asset["hardware_id"]) == before
    assert len(repository.list_assets()) == 1
    other = register(repository, device("machine-B", name="DAQ-renamed"))
    assert other["hardware_id"] == "machine-B"
    current = [device("machine-B", name="DAQ"), renamed]
    assert match_runtime_device(asset, current, lambda item: item["hostapi_name"]) is renamed
    with pytest.raises(HardwareRuntimeMatchError):
        match_runtime_device(asset, current[:1], lambda item: item["hostapi_name"])


def test_machine_id_collision_with_another_hostapi_rejects_without_overwrite(repository):
    soundcard = repository.register_asset({"name": "DAQ", "max_input_channels": 1}, "ASIO", "Card", 48000)
    before_assets = repository.list_assets()
    before_channels = repository.list_channels(soundcard["hardware_id"])
    with pytest.raises(HardwareValidationError, match="hostapi"):
        register(repository, device(machine_id=soundcard["hardware_id"]))
    assert repository.list_assets() == before_assets
    assert repository.list_channels(soundcard["hardware_id"]) == before_channels


def test_samplerate_rules_dispatch_by_registered_hostapi(repository):
    asset = register(repository, rate=51200)
    assert asset["samplerate"] == 51200
    for rate in (4000, 102400, 51200):
        assert repository.update_asset_fields(asset["hardware_id"], {"samplerate": rate})
    for rate in (True, 51200.0, 3999, 102401):
        with pytest.raises(HardwareValidationError):
            repository.update_asset_fields(asset["hardware_id"], {"samplerate": rate})
        with pytest.raises(HardwareValidationError):
            register(repository, device("invalid-rate"), rate=rate)
    with pytest.raises(HardwareValidationError):
        repository.register_asset({"name": "DAQ", "max_input_channels": 1}, "ASIO", "Card", 51200)
    card = repository.register_asset({"name": "DAQ", "max_input_channels": 1}, "ASIO", "Card", 48000)
    with pytest.raises(HardwareValidationError):
        repository.update_asset_fields(card["hardware_id"], {"samplerate": 51200})
    assert card["hardware_id"] != "machine-A"


@pytest.mark.parametrize("standard_spl", [94, 114, None])
def test_input_calibration_saves_factor_and_only_actual_conditions(repository, standard_spl):
    structure_before = schema(repository.db_path)
    register(repository)
    kwargs = dict(hardware_id="machine-A", db_path=repository.db_path, input_conditions=conditions())
    if standard_spl is None:
        save_manual_mic_channel_v2pa_factor(15, 2.123456789012345, **kwargs)
    else:
        save_mic_channel_v2pa_factor(15, 2.123456789012345, standard_spl=standard_spl, **kwargs)
    row = repository.get_channel_calibration("machine-A", "input", 15)
    assert row["calibration_type"] == "mic_v2pa"
    assert row["factor_value"] == 2.123456789012345
    assert row["standard_spl"] == standard_spl
    assert row["max_voltage"] is None
    assert json.loads(row["coefficients_json"]) == conditions()
    repository.update_asset_fields("machine-A", {"samplerate": 44100})
    assert load_mic_channel_v2pa_factors("machine-A", repository.db_path) == {15: 2.123456789012345}
    register(repository, device("machine-B"))
    assert load_mic_channel_v2pa_factors("machine-B", repository.db_path) == {}
    assert repository.get_channel_calibration("machine-A", "input", 8)["factor_value"] is None
    assert schema(repository.db_path) == structure_before


def test_calibration_transaction_failure_preserves_factors_and_conditions(repository):
    register(repository)
    repository.update_mic_channel_calibrations("machine-A", {0: 1.25}, {0: 94},
                                              channel_conditions={0: conditions()})
    before = repository.list_channels("machine-A")
    # The first row would succeed; the second missing placeholder must roll it back.
    with pytest.raises(HardwareValidationError):
        repository.update_mic_channel_calibrations("machine-A", {0: 9, 2: 8}, {0: 114, 2: None},
                                                  channel_conditions={0: conditions(5), 2: conditions(5)})
    assert repository.list_channels("machine-A") == before


def test_reset_only_selected_device_channels_and_soundcard_payloads_unchanged(repository):
    register(repository)
    register(repository, device("machine-B"))
    for machine in ("machine-A", "machine-B"):
        repository.update_mic_channel_calibrations(machine, {0: 1, 8: 2}, {0: 94, 8: None},
                                                  channel_conditions={0: conditions(), 8: conditions()})
    clear_mic_channel_v2pa_factors("machine-A", [8], repository.db_path)
    assert load_mic_channel_v2pa_factors("machine-A", repository.db_path) == {0: 1}
    assert load_mic_channel_v2pa_factors("machine-B", repository.db_path) == {0: 1, 8: 2}
    card = repository.register_asset({"name": "Card", "max_input_channels": 1, "max_output_channels": 1},
                                     "ASIO", "Card", 48000)
    repository.update_mic_channel_calibrations(card["hardware_id"], {0: 3}, {0: 94})
    assert repository.get_channel_calibration(card["hardware_id"], "input", 0)["coefficients_json"] is None
    repository.update_output_amplitude_calibration(card["hardware_id"], [1, 2], 5)
    output = repository.get_output_amplitude_calibration(card["hardware_id"])
    assert json.loads(output["coefficients_json"]) == {"calibration_coefficients": [1, 2]}


@pytest.mark.parametrize("invalid", [
    None, {}, {**conditions(), "source": "manual"}, {**conditions(), "unit": "Pa"},
    {**conditions(), "input_mode": "voltage"}, {**conditions(), "range_max": float("nan")},
    {**conditions(), "range_min": -3, "range_max": 3},
])
def test_invalid_or_missing_conditions_do_not_modify_vkinging_calibration(repository, invalid):
    register(repository)
    before = repository.list_channels("machine-A")
    with pytest.raises(HardwareValidationError, match="conditions"):
        save_manual_mic_channel_v2pa_factor(0, 2, hardware_id="machine-A", db_path=repository.db_path,
                                          input_conditions=invalid)
    assert repository.list_channels("machine-A") == before


def test_vkinging_measured_standard_spl_is_94_or_114(repository):
    register(repository)
    with pytest.raises(HardwareValidationError, match="94 or 114"):
        save_mic_channel_v2pa_factor(0, 2, 100, "machine-A", repository.db_path, input_conditions=conditions())


@pytest.mark.parametrize("payload", [
    "BROKEN", None, "[]", json.dumps({**conditions(), "source": "manual"}),
    json.dumps({**conditions(), "unit": "Pa"}), json.dumps(conditions(3)),
])
@pytest.mark.parametrize("read", ["get_channel", "list_calibrations", "list_channels", "load_factors"])
def test_corrupt_stored_vkinging_conditions_are_rejected_without_rewriting(repository, payload, read):
    register(repository)
    repository.update_mic_channel_calibrations("machine-A", {8: 2}, {8: 94},
                                              channel_conditions={8: conditions()})
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute("UPDATE hardware_channel_calibrations SET coefficients_json=? "
                           "WHERE hardware_id='machine-A' AND direction='input' AND channel_index=8", (payload,))
        before = connection.execute("SELECT * FROM hardware_channel_calibrations ORDER BY rowid").fetchall()
    readers = {
        "get_channel": lambda: repository.get_channel_calibration("machine-A", "input", 8),
        "list_calibrations": lambda: repository.list_channel_calibrations("machine-A", "input", "mic_v2pa"),
        "list_channels": lambda: repository.list_channels("machine-A", "input"),
        "load_factors": lambda: load_mic_channel_v2pa_factors("machine-A", repository.db_path),
    }
    with pytest.raises(HardwareValidationError, match="conditions"):
        readers[read]()
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("SELECT * FROM hardware_channel_calibrations ORDER BY rowid").fetchall() == before


def forbid_portaudio(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("vkinging must not query or apply a PortAudio device")
    for name in ("query_hostapis", "query_devices", "_terminate", "_initialize"):
        monkeypatch.setattr(device_module.sd, name, forbidden)
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", forbidden)
    monkeypatch.setattr(SoundDeviceManager, "change_default_input_device", forbidden)


def test_vkinging_helpers_save_sparse_channels_without_portaudio(repository, tmp_path, monkeypatch):
    asset = register(repository, rate=51200)
    mic = augment_runtime_device(device(), asset)
    config_path = tmp_path / "config" / "selected.json"
    monkeypatch.setattr(device_module, "AUDIO_DEVICE_CONFIG_PATH", str(config_path))
    forbid_portaudio(monkeypatch)
    assert SoundDeviceManager.get_device_hostapi_name(mic) == "vkinging"
    assert SoundDeviceManager.serialize_device(mic)["hardware_id"] == "machine-A"
    assert SoundDeviceManager.channels_for_device(mic) == [0, 8, 15, 31]
    assert SoundDeviceManager.channels_for_device(mic, "speaker") == []
    assert SoundDeviceManager.restore_saved_mic_channels_strict(mic, [1]) == []
    assert SoundDeviceManager.restore_mic_channels(mic, [1]) == []
    assert SoundDeviceManager.restore_mic_channels(mic, None) == [0, 8, 15, 31]
    SoundDeviceManager.save_selected_devices(MappingProxyType(mic), None, [8, 15, 31])
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["version"] == 3
    assert saved["mic"]["hardware_id"] == "machine-A"
    assert saved["mic"]["hostapi_name"] == "vkinging"
    assert saved["speaker"] is None
    assert saved["mic_channels"] == [8, 15, 31]


def test_registered_vkinging_startup_restores_pending_database_snapshot_without_native_calls(
        repository, tmp_path, monkeypatch):
    asset = register(repository, rate=51200)
    mic = augment_runtime_device(device(), asset)
    config_path = tmp_path / "selected.json"
    monkeypatch.setattr(device_module, "AUDIO_DEVICE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(device_module, "HardwareManagementRepository", lambda: repository)
    SoundDeviceManager.save_selected_devices(mic, None, [8, 31])
    forbid_portaudio(monkeypatch)
    result = SoundDeviceManager().get_startup_devices()
    assert result["mic"]["hardware_id"] == "machine-A"
    assert result["mic"]["samplerate"] == 51200
    assert result["mic"]["physical_channels"] == (0, 8, 15, 31)
    assert result["mic_channels"] == [8, 31]
    assert result["speaker"] is None
    assert result["device_available"] is False
    assert result["validation_pending"] is True
    assert result["can_retry_saved_devices"] is True
    assert "index" not in result["mic"]


def test_startup_reports_corrupt_stored_conditions_without_rewriting(repository, tmp_path, monkeypatch):
    asset = register(repository)
    mic = augment_runtime_device(device(), asset)
    repository.update_mic_channel_calibrations("machine-A", {8: 2}, {8: 94},
                                              channel_conditions={8: conditions()})
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute("UPDATE hardware_channel_calibrations SET coefficients_json='BROKEN' "
                           "WHERE hardware_id='machine-A' AND channel_index=8")
    monkeypatch.setattr(device_module, "AUDIO_DEVICE_CONFIG_PATH", str(tmp_path / "selected.json"))
    monkeypatch.setattr(device_module, "HardwareManagementRepository", lambda: repository)
    SoundDeviceManager.save_selected_devices(mic, None, [8])
    forbid_portaudio(monkeypatch)
    result = SoundDeviceManager().get_startup_devices()
    assert result["device_available"] is False
    assert "conditions" in result["startup_device_error_reason"]
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("SELECT coefficients_json FROM hardware_channel_calibrations "
                                  "WHERE hardware_id='machine-A' AND channel_index=8").fetchone()[0] == "BROKEN"


def test_runtime_restore_uses_machine_id_and_registered_physical_intersection(repository, monkeypatch):
    asset = register(repository)
    mic = augment_runtime_device(device(), asset)
    saved = device_module.build_selected_device_payload(mic, None, [8, 31])
    monkeypatch.setattr(device_module, "HardwareManagementRepository", lambda: repository)
    forbid_portaudio(monkeypatch)
    runtime = device(name="renamed", channels=(8, 31, 63))
    result = SoundDeviceManager()._try_resolve_registered_startup_devices(saved, [device("machine-B"), runtime])
    assert result["device_available"] is True
    assert result["mic"]["name"] == "renamed"
    assert result["mic"]["physical_channels"] == (8, 31)
    assert result["mic_channels"] == [8, 31]
    unavailable = SoundDeviceManager()._try_resolve_registered_startup_devices(saved, [device(channels=(0, 8))])
    assert unavailable["device_available"] is False
    assert unavailable["mic"] is None


@pytest.mark.parametrize("runtime_validated", [False, True])
@pytest.mark.parametrize("output_case", ["absent", "matching", "wrong_api", "ambiguous"])
def test_vkinging_restore_preserves_independent_registered_output_without_portaudio(
        repository, tmp_path, monkeypatch, runtime_validated, output_case):
    mic_asset = register(repository, rate=51200)
    mic = augment_runtime_device(device(), mic_asset)
    output_runtime = {"name": "Speaker", "index": 9, "hostapi": 0, "hostapi_name": "ASIO", "max_output_channels": 2,
                      "max_input_channels": 0, "default_samplerate": 48000.0}
    output_asset = repository.register_asset(output_runtime, "ASIO", "Desk output", 48000)
    speaker = augment_runtime_device(output_runtime, output_asset)
    config_path = tmp_path / "selected.json"
    monkeypatch.setattr(device_module, "AUDIO_DEVICE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(device_module, "HardwareManagementRepository", lambda: repository)
    forbid_portaudio(monkeypatch)
    SoundDeviceManager.save_selected_devices(mic, speaker, [8, 31])
    saved_bytes = config_path.read_bytes()
    saved = json.loads(saved_bytes)
    if runtime_validated:
        snapshots = [device(name="renamed")]
        if output_case != "absent":
            snapshots.append(dict(output_runtime, hostapi_name="other" if output_case == "wrong_api" else "ASIO"))
        if output_case == "ambiguous":
            snapshots.append(dict(output_runtime, index=10))
        result = SoundDeviceManager()._try_resolve_registered_startup_devices(saved, snapshots)
    else:
        result = SoundDeviceManager().get_startup_devices()
    assert result["mic"]["hardware_id"] == "machine-A"
    assert result["mic_channels"] == [8, 31]
    assert result["validation_pending"] is not runtime_validated
    assert result["device_available"] is runtime_validated
    assert result["speaker"]["hardware_id"] == output_asset["hardware_id"]
    assert result["speaker"]["hostapi_name"] == "ASIO"
    assert result["speaker"]["display_name"] == "Desk output"
    assert result["speaker"]["samplerate"] == 48000
    if runtime_validated and output_case == "matching":
        assert result["speaker"]["index"] == 9
        assert result["speaker"]["hostapi"] == 0
    else:
        assert "index" not in result["speaker"]
    assert config_path.read_bytes() == saved_bytes


def test_read_only_database_rejects_writes_without_changing_existing_data(repository, monkeypatch):
    register(repository)
    repository.update_mic_channel_calibrations("machine-A", {0: 1}, {0: 94},
                                              channel_conditions={0: conditions()})
    assets_before = repository.list_assets()
    channels_before = repository.list_channels("machine-A")
    structure_before = schema(repository.db_path)

    @contextmanager
    def readonly_connection():
        connection = sqlite3.connect("file:" + repository.db_path.replace("\\", "/") + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    monkeypatch.setattr(repository, "_connection", readonly_connection)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        register(repository, device(name="new-route"))
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        repository.update_mic_channel_calibrations("machine-A", {0: 9}, {0: None},
                                                  channel_conditions={0: conditions(5)})
    assert repository.list_assets() == assets_before
    assert repository.list_channels("machine-A") == channels_before
    assert schema(repository.db_path) == structure_before


def test_registration_transaction_rolls_back_route_and_new_placeholders_on_failure(repository):
    register(repository, device(channels=(0,)))
    before_assets = repository.list_assets()
    before_channels = repository.list_channels("machine-A")
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute("CREATE TRIGGER reject_test_channel BEFORE INSERT ON hardware_channel_calibrations "
                           "WHEN NEW.channel_index = 8 BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="test failure"):
        register(repository, device(name="renamed", channels=(0, 7, 8)))
    assert repository.list_assets() == before_assets
    assert repository.list_channels("machine-A") == before_channels
    structure_before = schema(repository.db_path)
    assert repository.delete_asset("machine-A")
    assert repository.list_channels("machine-A") == []
    assert schema(repository.db_path) == structure_before
