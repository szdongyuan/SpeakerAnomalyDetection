import builtins
import json
import os
import sqlite3

import base.sound_device_manager as module
from base.hardware_management import MissingHardwareTablesError
from base.sound_device_manager import SoundDeviceManager


def test_sound_device_manager_no_longer_exports_registered_repository_read_exceptions():
    assert not hasattr(module, "REGISTERED_REPOSITORY_READ_EXCEPTIONS")


def _device(index, name, hostapi=0, samplerate=48000.0, inputs=0, outputs=0):
    return {
        "index": index,
        "name": name,
        "hostapi": hostapi,
        "default_samplerate": samplerate,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


def _registered_asset(
    hardware_id,
    device_name,
    *,
    hostapi_name="API",
    display_name=None,
    hardware_type="audio_interface",
    samplerate=44100,
    bit_depth=24,
    latency_ms=55,
    inputs=0,
    outputs=0,
):
    return {
        "hardware_id": hardware_id,
        "hardware_type": hardware_type,
        "display_name": display_name or device_name,
        "device_name": device_name,
        "hostapi_name": hostapi_name,
        "samplerate": samplerate,
        "bit_depth": bit_depth,
        "latency_ms": latency_ms,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "updated_at": "2026-06-23 12:00:00",
    }


def _augmented_device(
    index,
    name,
    hardware_id,
    *,
    hostapi=0,
    hostapi_name="API",
    samplerate=48000.0,
    registered_samplerate=44100,
    bit_depth=24,
    latency_ms=55,
    inputs=0,
    outputs=0,
):
    device = _device(index, name, hostapi=hostapi, samplerate=samplerate, inputs=inputs, outputs=outputs)
    device.update(
        {
            "hardware_id": hardware_id,
            "display_name": f"{name} Display",
            "device_name": name,
            "hostapi_name": hostapi_name,
            "hardware_type": "audio_interface",
            "samplerate": registered_samplerate,
            "bit_depth": bit_depth,
            "latency_ms": latency_ms,
        }
    )
    return device


def _input_channel(hardware_id, channel_index):
    return {
        "channel_id": f"{hardware_id}-in-{channel_index}",
        "hardware_id": hardware_id,
        "direction": "input",
        "channel_index": channel_index,
        "channel_label": f"In{channel_index + 1}",
    }


class _FakeHardwareRepository:
    def __init__(self, *, tables_exist=True, assets=None, channels=None):
        self._tables_exist = tables_exist
        self.assets = {asset["hardware_id"]: dict(asset) for asset in assets or []}
        self.channels = {}
        for hardware_id, rows in (channels or {}).items():
            self.channels[hardware_id] = [dict(row) for row in rows]
        self.tables_exist_calls = 0
        self.get_asset_calls = []
        self.list_assets_calls = 0
        self.list_assets_for_selection_calls = 0
        self.list_channels_calls = []

    def tables_exist(self):
        self.tables_exist_calls += 1
        return self._tables_exist

    def get_asset(self, hardware_id):
        self.get_asset_calls.append(hardware_id)
        asset = self.assets.get(hardware_id)
        return dict(asset) if asset is not None else None

    def list_assets(self):
        self.list_assets_calls += 1
        return [dict(asset) for asset in self.assets.values()]

    def list_assets_for_selection(self):
        self.list_assets_for_selection_calls += 1
        grouped = {}
        for asset in self.list_assets():
            api_group = grouped.setdefault(asset["hostapi_name"], {"input": [], "output": []})
            if asset["max_input_channels"] > 0:
                api_group["input"].append(dict(asset))
            if asset["max_output_channels"] > 0:
                api_group["output"].append(dict(asset))
        return grouped

    def list_channels(self, hardware_id, direction=None):
        self.list_channels_calls.append((hardware_id, direction))
        rows = [dict(row) for row in self.channels.get(hardware_id, [])]
        if direction is not None:
            rows = [row for row in rows if row.get("direction") == direction]
        return rows


def _install_audio(monkeypatch, tmp_path, *, devices_by_api=None, default_mic=None, default_speaker=None):
    config_path = tmp_path / "audio_device_config.json"
    monkeypatch.setattr(module, "AUDIO_DEVICE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(SoundDeviceManager, "refresh_available_device", staticmethod(lambda: None))
    monkeypatch.setattr(SoundDeviceManager, "get_api_info", staticmethod(lambda index=None: {"name": "API"}))
    monkeypatch.setattr(
        SoundDeviceManager,
        "get_device_info",
        staticmethod(lambda: devices_by_api or {"API": {"input": [], "output": []}}),
    )
    monkeypatch.setattr(
        SoundDeviceManager,
        "get_default_device",
        lambda self, device, refresh=True: ("OK", default_mic if device == "mic" else default_speaker),
    )
    return config_path


def test_v2_save_shape_with_hardware_id_omits_runtime_indexes(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    mic = _augmented_device(3, "Registered Mic", "mic-1", inputs=4, samplerate=96000.0)
    speaker = _augmented_device(8, "Registered Speaker", "speaker-1", outputs=2, samplerate=44100.0)

    SoundDeviceManager.save_selected_devices(mic, speaker, [1, "0", 99])

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert set(payload) == {"version", "mic", "speaker", "mic_channels"}
    assert payload["version"] == 2
    assert payload["mic_channels"] == [0, 1]
    assert "index" not in payload["mic"]
    assert "index" not in payload["speaker"]
    assert payload["mic"]["hardware_id"] == "mic-1"
    assert payload["speaker"]["hardware_id"] == "speaker-1"
    assert payload["mic"]["name"] == payload["mic"]["device_name"] == "Registered Mic"
    assert payload["speaker"]["name"] == payload["speaker"]["device_name"] == "Registered Speaker"
    assert payload["mic"]["default_samplerate"] == 96000.0
    assert payload["speaker"]["default_samplerate"] == 44100.0
    assert payload["mic"]["samplerate"] == 44100
    assert payload["mic"]["bit_depth"] == 24
    assert payload["mic"]["latency_ms"] == 55


def test_legacy_save_behavior_is_preserved_without_hardware_id(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    mic = _device(3, "Legacy Mic", inputs=2)
    speaker = _device(8, "Legacy Speaker", outputs=2)

    SoundDeviceManager.save_selected_devices(mic, speaker, [1, 99])

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert set(payload) == {"mic", "speaker", "mic_channels"}
    assert payload["mic"] == {"name": "Legacy Mic", "hostapi_name": "API", "default_samplerate": 48000.0}
    assert payload["speaker"] == {"name": "Legacy Speaker", "hostapi_name": "API", "default_samplerate": 48000.0}
    assert payload["mic_channels"] == [1]


def test_missing_config_does_not_apply_or_save_default_devices(monkeypatch, tmp_path):
    mic = _device(1, "Default Mic", inputs=2)
    speaker = _device(2, "Default Speaker", outputs=2)
    config_path = _install_audio(monkeypatch, tmp_path, default_mic=mic, default_speaker=speaker)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []
    assert not config_path.exists()


def test_invalid_json_is_unavailable_and_not_retryable(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text("{bad json", encoding="utf-8")

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False


def test_saved_devices_missing_are_unavailable_and_retryable(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "mic": {"name": "Saved Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {"name": "Saved Speaker", "hostapi_name": "API", "default_samplerate": 48000.0},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    applied = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert "不存在" in result["startup_device_error_reason"] or "无效" in result["startup_device_error_reason"]
    assert applied == []


def test_refresh_exception_is_unavailable_without_fallback_or_save(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "mic": {"name": "Saved Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {"name": "Saved Speaker", "hostapi_name": "API", "default_samplerate": 48000.0},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    applied = []
    saved = []
    monkeypatch.setattr(
        SoundDeviceManager,
        "refresh_available_device",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("device scan failed"))),
    )
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "device scan failed" in result["startup_device_error_reason"]
    assert applied == []
    assert saved == []


def test_get_device_info_exception_is_unavailable_without_fallback_or_save(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "mic": {"name": "Saved Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {"name": "Saved Speaker", "hostapi_name": "API", "default_samplerate": 48000.0},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    applied = []
    saved = []
    monkeypatch.setattr(
        SoundDeviceManager,
        "get_device_info",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("device list failed"))),
    )
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "device list failed" in result["startup_device_error_reason"]
    assert applied == []
    assert saved == []


def test_incomplete_saved_device_descriptor_is_unavailable_and_not_retryable(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "mic": {"name": "Saved Mic", "hostapi_name": "API"},
                "speaker": {"name": "Saved Speaker", "hostapi_name": "API", "default_samplerate": 48000.0},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []


def test_saved_devices_restore_when_present(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "mic": {
                    "name": "Saved Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "name": "Saved Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is True
    assert result["mic"] == {**mic, "samplerate": 44100}
    assert result["speaker"] == {**speaker, "samplerate": 44100}
    assert result["mic_channels"] == [1]
    assert applied == [(5, 6)]
    assert saved == [({**mic, "samplerate": 44100}, {**speaker, "samplerate": 44100}, [1])]


def test_legacy_startup_restore_persists_saved_samplerate_across_restart(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "mic": {
                    "name": "Saved Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "name": "Saved Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )
    applied = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))

    first_result = SoundDeviceManager().get_startup_devices()

    assert first_result["device_available"] is True
    assert first_result["mic"] == {**mic, "samplerate": 44100}
    assert first_result["speaker"] == {**speaker, "samplerate": 44100}
    rewritten_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert rewritten_payload["mic"]["samplerate"] == 44100
    assert rewritten_payload["speaker"]["samplerate"] == 44100

    second_result = SoundDeviceManager().get_startup_devices()

    assert second_result["device_available"] is True
    assert second_result["mic"] == {**mic, "samplerate": 44100}
    assert second_result["speaker"] == {**speaker, "samplerate": 44100}
    assert applied == [(5, 6), (5, 6)]


def test_legacy_startup_restore_without_saved_samplerate_is_unavailable(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "mic": {
                    "name": "Saved Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "speaker": {
                    "name": "Saved Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "采样率" in result["startup_device_error_reason"]
    assert applied == []
    assert saved == []


def test_saved_devices_restore_change_default_failure_returns_retryable_unavailable(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "mic": {
                    "name": "Saved Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "name": "Saved Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )
    saved = []
    monkeypatch.setattr(
        SoundDeviceManager,
        "change_default_device",
        staticmethod(lambda *args: (_ for _ in ()).throw(RuntimeError("apply failed"))),
    )
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "apply failed" in result["startup_device_error_reason"]
    assert saved == []


def test_saved_devices_restore_save_failure_returns_retryable_unavailable(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    original_payload = {
        "mic": {
            "name": "Saved Mic",
            "hostapi_name": "API",
            "default_samplerate": 48000.0,
            "samplerate": 44100,
        },
        "speaker": {
            "name": "Saved Speaker",
            "hostapi_name": "API",
            "default_samplerate": 48000.0,
            "samplerate": 44100,
        },
        "mic_channels": [1],
    }
    original_config = json.dumps(original_payload)
    config_path.write_text(original_config, encoding="utf-8")
    applied = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))

    def partial_save(*args):
        config_path.write_text('{"mic":', encoding="utf-8")
        raise RuntimeError("save failed")

    monkeypatch.setattr(
        SoundDeviceManager,
        "save_selected_devices",
        staticmethod(partial_save),
    )

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "save failed" in result["startup_device_error_reason"]
    assert applied == [(5, 6)]
    assert config_path.read_text(encoding="utf-8") == original_config


def test_save_selected_devices_failure_removes_new_partial_config(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)

    def partial_save(*args):
        config_path.write_text('{"mic":', encoding="utf-8")
        raise RuntimeError("save failed")

    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(partial_save))

    try:
        SoundDeviceManager()._save_selected_devices_with_config_rollback(mic, speaker, [1])
    except RuntimeError as exc:
        assert str(exc) == "save failed"
    else:
        raise AssertionError("expected save failure")

    assert not config_path.exists()


def test_saved_devices_restore_save_failure_reports_rollback_write_failure(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    original_config = json.dumps(
        {
            "mic": {
                "name": "Saved Mic",
                "hostapi_name": "API",
                "default_samplerate": 48000.0,
                "samplerate": 44100,
            },
            "speaker": {
                "name": "Saved Speaker",
                "hostapi_name": "API",
                "default_samplerate": 48000.0,
                "samplerate": 44100,
            },
            "mic_channels": [1],
        }
    )
    partial_config = '{"mic":'
    config_path.write_text(original_config, encoding="utf-8")
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: None))

    def partial_save(*args):
        config_path.write_text(partial_config, encoding="utf-8")
        raise RuntimeError("save failed")

    real_open = builtins.open

    def fail_restore_write(path, mode="r", *args, **kwargs):
        if str(path) == str(config_path) and mode == "wb":
            raise OSError("restore failed")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(partial_save))
    monkeypatch.setattr(builtins, "open", fail_restore_write)

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert "save failed" in result["startup_device_error_reason"]
    assert "rollback" in result["startup_device_error_reason"] or "restor" in result["startup_device_error_reason"]
    assert "restore failed" in result["startup_device_error_reason"]
    assert config_path.read_text(encoding="utf-8") == partial_config


def test_save_selected_devices_failure_reports_new_partial_config_delete_failure(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    partial_config = '{"mic":'

    def partial_save(*args):
        config_path.write_text(partial_config, encoding="utf-8")
        raise RuntimeError("save failed")

    def fail_remove(path):
        if str(path) == str(config_path):
            raise OSError("delete failed")
        raise AssertionError(f"unexpected remove path: {path}")

    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(partial_save))
    monkeypatch.setattr(module.os, "remove", fail_remove)

    try:
        SoundDeviceManager()._save_selected_devices_with_config_rollback(mic, speaker, [1])
    except RuntimeError as exc:
        assert "save failed" in str(exc)
        assert "rollback" in str(exc) or "restor" in str(exc) or "cleanup" in str(exc)
        assert "delete failed" in str(exc)
    else:
        raise AssertionError("expected rollback cleanup failure")

    assert config_path.read_text(encoding="utf-8") == partial_config


def test_v2_startup_restore_by_hardware_id_succeeds_with_exact_runtime_match(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=4)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "mic-1",
                    "name": "Stale Mic",
                    "device_name": "Stale Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 12345.0,
                    "samplerate": 12345,
                    "bit_depth": 8,
                    "latency_ms": 999,
                },
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Stale Speaker",
                    "device_name": "Stale Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 12345.0,
                    "samplerate": 12345,
                    "bit_depth": 8,
                    "latency_ms": 999,
                },
                "mic_channels": [2, "0"],
            }
        ),
        encoding="utf-8",
    )
    mic_asset = _registered_asset("mic-1", "Runtime Mic", display_name="DB Mic", inputs=4)
    speaker_asset = _registered_asset("speaker-1", "Runtime Speaker", display_name="DB Speaker", outputs=2)
    repository = _FakeHardwareRepository(
        assets=[mic_asset, speaker_asset],
        channels={"mic-1": [_input_channel("mic-1", 0), _input_channel("mic-1", 2)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is True
    assert result["mic_channels"] == [0, 2]
    assert applied == [(5, 6)]
    assert saved == [
        (
            {
                **runtime_mic,
                "hardware_id": "mic-1",
                "display_name": "DB Mic",
                "device_name": "Runtime Mic",
                "hardware_type": "audio_interface",
                "hostapi_name": "API",
                "samplerate": 44100,
                "bit_depth": 24,
                "latency_ms": 55,
            },
            {
                **runtime_speaker,
                "hardware_id": "speaker-1",
                "display_name": "DB Speaker",
                "device_name": "Runtime Speaker",
                "hardware_type": "audio_interface",
                "hostapi_name": "API",
                "samplerate": 44100,
                "bit_depth": 24,
                "latency_ms": 55,
            },
            [0, 2],
        )
    ]
    assert repository.get_asset_calls == ["mic-1", "speaker-1"]
    assert repository.list_channels_calls == [("mic-1", "input")]


def test_v2_startup_restore_change_default_failure_returns_retryable_unavailable(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=4)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1", "name": "Stale Mic", "device_name": "Stale Mic"},
                "speaker": {"hardware_id": "speaker-1", "name": "Stale Speaker", "device_name": "Stale Speaker"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", display_name="DB Mic", inputs=4),
            _registered_asset("speaker-1", "Runtime Speaker", display_name="DB Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    saved = []
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    monkeypatch.setattr(
        SoundDeviceManager,
        "change_default_device",
        staticmethod(lambda *args: (_ for _ in ()).throw(RuntimeError("apply failed"))),
    )
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "apply failed" in result["startup_device_error_reason"]
    assert saved == []


def test_v2_startup_restore_save_failure_returns_retryable_unavailable(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=4)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    original_payload = {
        "version": 2,
        "mic": {"hardware_id": "mic-1", "name": "Stale Mic", "device_name": "Stale Mic"},
        "speaker": {"hardware_id": "speaker-1", "name": "Stale Speaker", "device_name": "Stale Speaker"},
        "mic_channels": [0],
    }
    original_config = json.dumps(original_payload)
    config_path.write_text(original_config, encoding="utf-8")
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", display_name="DB Mic", inputs=4),
            _registered_asset("speaker-1", "Runtime Speaker", display_name="DB Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    applied = []
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))

    def partial_save(*args):
        config_path.write_text('{"version": 2, "mic":', encoding="utf-8")
        raise RuntimeError("save failed")

    monkeypatch.setattr(
        SoundDeviceManager,
        "save_selected_devices",
        staticmethod(partial_save),
    )

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "save failed" in result["startup_device_error_reason"]
    assert applied == [(5, 6)]
    assert config_path.read_text(encoding="utf-8") == original_config


def test_registered_db_row_wins_over_saved_mutable_metadata(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=2)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "mic-1",
                    "name": "Runtime Mic",
                    "device_name": "Runtime Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 48000,
                    "bit_depth": 32,
                    "latency_ms": 100,
                },
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Runtime Speaker",
                    "device_name": "Runtime Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 48000,
                    "bit_depth": 32,
                    "latency_ms": 100,
                },
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", samplerate=44100, bit_depth=16, latency_ms=25, inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", samplerate=44100, bit_depth=16, latency_ms=25, outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 1)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: None))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is True
    restored_mic = saved[0][0]
    assert restored_mic["samplerate"] == 44100
    assert restored_mic["bit_depth"] == 16
    assert restored_mic["latency_ms"] == 25


def test_v2_registered_restore_does_not_require_valid_legacy_descriptors(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=2)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1"},
                "speaker": {"hardware_id": "speaker-1"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: None))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is True
    assert result["mic"]["hardware_id"] == "mic-1"
    assert result["speaker"]["hardware_id"] == "speaker-1"
    assert applied == [(5, 6)]


def test_registered_startup_restore_matches_all_runtime_devices_not_channel_groups(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=1, outputs=2)
    runtime_speaker = _device(6, "Runtime Speaker", inputs=2, outputs=1)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_speaker], "output": [runtime_mic]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1"},
                "speaker": {"hardware_id": "speaker-1"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is True
    assert result["mic"]["hardware_id"] == "mic-1"
    assert result["speaker"]["hardware_id"] == "speaker-1"
    assert applied == [(5, 6)]
    assert saved[0][0]["name"] == "Runtime Mic"
    assert saved[0][1]["name"] == "Runtime Speaker"


def test_registered_startup_restore_rejects_runtime_mic_without_input_capacity(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=0, outputs=2)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [], "output": [runtime_mic, runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1"},
                "speaker": {"hardware_id": "speaker-1"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []


def test_registered_startup_restore_rejects_runtime_mic_channel_beyond_capacity(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=1)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1"},
                "speaker": {"hardware_id": "speaker-1"},
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 1)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []


def test_registered_startup_restore_rejects_runtime_speaker_without_output_capacity(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=2)
    runtime_speaker = _device(6, "Runtime Speaker", inputs=2, outputs=0)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic, runtime_speaker], "output": []}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1"},
                "speaker": {"hardware_id": "speaker-1"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []


def test_v2_startup_restore_invalid_registered_channels_is_unavailable(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=4)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1", "name": "Runtime Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Runtime Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [0, 3],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=4),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []


def test_v2_startup_restore_rejects_saved_channel_beyond_registered_asset_capacity(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=4)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1", "name": "Runtime Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Runtime Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [0, 3],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0), _input_channel("mic-1", 3)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []


def test_v2_startup_restore_empty_registered_channels_is_unavailable(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1", "name": "Runtime Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Runtime Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True


def test_v2_startup_restore_runtime_no_match_is_unavailable_and_retryable(monkeypatch, tmp_path):
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [_device(5, "Other Mic", inputs=2)], "output": [_device(6, "Runtime Speaker", outputs=2)]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1", "name": "Runtime Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Runtime Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True


def test_v2_startup_restore_runtime_ambiguous_match_is_unavailable_and_retryable(monkeypatch, tmp_path):
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={
            "API": {
                "input": [_device(5, "Runtime Mic", inputs=2), _device(9, "Runtime Mic", inputs=2)],
                "output": [_device(6, "Runtime Speaker", outputs=2)],
            }
        },
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1", "name": "Runtime Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Runtime Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [_input_channel("mic-1", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True


def test_v2_hardware_tables_missing_is_unavailable_without_legacy_fallback(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    original_payload = {
        "version": 2,
        "mic": {
            "hardware_id": "mic-1",
            "name": "Legacy Mic",
            "device_name": "Missing Registered Mic",
            "hostapi_name": "API",
            "default_samplerate": 48000.0,
            "samplerate": 44100,
        },
        "speaker": {
            "hardware_id": "speaker-1",
            "name": "Legacy Speaker",
            "device_name": "Missing Registered Speaker",
            "hostapi_name": "API",
            "default_samplerate": 48000.0,
            "samplerate": 44100,
        },
        "mic_channels": [1],
    }
    config_path.write_text(json.dumps(original_payload), encoding="utf-8")
    repository = _FakeHardwareRepository(tables_exist=False)
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "已注册硬件当前不可用" in result["startup_notice_message"]
    assert applied == []
    assert saved == []
    assert repository.tables_exist_calls == 1
    assert repository.get_asset_calls == []
    assert json.loads(config_path.read_text(encoding="utf-8")) == original_payload


def test_v2_no_hardware_tables_and_invalid_legacy_fields_is_registered_unavailable(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1", "name": "", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {"hardware_id": "speaker-1", "name": "Speaker", "hostapi_name": "API"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(tables_exist=False)
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert "已注册硬件当前不可用" in result["startup_notice_message"]
    assert repository.get_asset_calls == []


def test_v2_missing_hardware_tables_error_from_get_asset_is_unavailable_without_legacy_fallback(
    monkeypatch, tmp_path
):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    original_payload = {
        "version": 2,
        "mic": {
            "hardware_id": "mic-1",
            "name": "Legacy Mic",
            "hostapi_name": "API",
            "default_samplerate": 48000.0,
            "samplerate": 44100,
        },
        "speaker": {
            "hardware_id": "speaker-1",
            "name": "Legacy Speaker",
            "hostapi_name": "API",
            "default_samplerate": 48000.0,
            "samplerate": 44100,
        },
        "mic_channels": [1],
    }
    config_path.write_text(json.dumps(original_payload), encoding="utf-8")

    class MissingTableRepository(_FakeHardwareRepository):
        def get_asset(self, hardware_id):
            self.get_asset_calls.append(hardware_id)
            raise MissingHardwareTablesError("missing hardware tables")

    repository = MissingTableRepository()
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "已注册硬件当前不可用" in result["startup_notice_message"]
    assert repository.get_asset_calls == ["mic-1"]
    assert applied == []
    assert saved == []
    assert json.loads(config_path.read_text(encoding="utf-8")) == original_payload


def test_v2_missing_hardware_tables_error_from_list_channels_is_unavailable_without_legacy_fallback(
    monkeypatch, tmp_path
):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    original_payload = {
        "version": 2,
        "mic": {
            "hardware_id": "mic-1",
            "name": "Legacy Mic",
            "hostapi_name": "API",
            "default_samplerate": 48000.0,
            "samplerate": 44100,
        },
        "speaker": {
            "hardware_id": "speaker-1",
            "name": "Legacy Speaker",
            "hostapi_name": "API",
            "default_samplerate": 48000.0,
            "samplerate": 44100,
        },
        "mic_channels": [1],
    }
    config_path.write_text(json.dumps(original_payload), encoding="utf-8")

    class MissingTableRepository(_FakeHardwareRepository):
        def list_channels(self, hardware_id, direction=None):
            self.list_channels_calls.append((hardware_id, direction))
            raise MissingHardwareTablesError("missing hardware tables")

    repository = MissingTableRepository(
        assets=[
            _registered_asset("mic-1", "Legacy Mic", inputs=2),
            _registered_asset("speaker-1", "Legacy Speaker", outputs=2),
        ],
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "已注册硬件当前不可用" in result["startup_notice_message"]
    assert repository.get_asset_calls == ["mic-1", "speaker-1"]
    assert repository.list_channels_calls == [("mic-1", "input")]
    assert applied == []
    assert saved == []
    assert json.loads(config_path.read_text(encoding="utf-8")) == original_payload


def test_v2_hardware_id_not_found_without_saved_samplerate_is_unavailable(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "missing-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "speaker": {
                    "hardware_id": "missing-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository()
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "已删除" in result["startup_device_error_reason"] or "不存在" in result["startup_device_error_reason"]
    assert repository.get_asset_calls == ["missing-mic", "missing-speaker"]
    assert applied == []


def test_v2_hardware_id_not_found_with_saved_samplerate_does_not_fallback_to_legacy_matching(
    monkeypatch, tmp_path
):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository()
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "已删除" in result["startup_device_error_reason"] or "不存在" in result["startup_device_error_reason"]
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_with_other_registered_assets_prompts_reselect(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("other-mic", "Other Mic", inputs=2),
            _registered_asset("other-speaker", "Other Speaker", outputs=2),
        ],
        channels={"other-mic": [_input_channel("other-mic", 0)]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert result["startup_device_error_reason"] == "已保存的注册硬件已删除或不存在，请在硬件管理中重新选择设备。"
    assert "请先" not in result["startup_device_error_reason"]
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert repository.list_channels_calls == [("other-mic", "input")]
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_with_no_alternate_mic_channels_prompts_register_first(
    monkeypatch, tmp_path
):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("other-mic", "Other Mic", inputs=2),
            _registered_asset("other-speaker", "Other Speaker", outputs=2),
        ]
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert result["startup_device_error_reason"] == "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    assert result["startup_recovery_action"] == "register_hardware"
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert repository.list_channels_calls == [("other-mic", "input")]
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_with_invalid_alternate_mic_channel_prompts_register_first(
    monkeypatch, tmp_path
):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    invalid_channel = _input_channel("other-mic", 0)
    invalid_channel["channel_index"] = True
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("other-mic", "Other Mic", inputs=2),
            _registered_asset("other-speaker", "Other Speaker", outputs=2),
        ],
        channels={"other-mic": [invalid_channel]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert result["startup_device_error_reason"] == "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    assert result["startup_recovery_action"] == "register_hardware"
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert repository.list_channels_calls == [("other-mic", "input")]
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_channel_probe_read_failure_is_retryable_unavailable(
    monkeypatch, tmp_path
):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )

    class FailingChannelRepository(_FakeHardwareRepository):
        def list_channels(self, hardware_id, direction=None):
            self.list_channels_calls.append((hardware_id, direction))
            raise MissingHardwareTablesError("missing hardware tables")

    repository = FailingChannelRepository(
        assets=[
            _registered_asset("other-mic", "Other Mic", inputs=2),
            _registered_asset("other-speaker", "Other Speaker", outputs=2),
        ]
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "已注册硬件当前不可用" in result["startup_device_error_reason"]
    assert result["startup_recovery_action"] is None
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert repository.list_channels_calls == [("other-mic", "input")]
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_with_assets_split_across_selection_groups_prompts_register_first(
    monkeypatch, tmp_path
):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("other-mic", "Other Mic", hostapi_name="API A", inputs=2),
            _registered_asset("other-speaker", "Other Speaker", hostapi_name="API B", outputs=2),
        ]
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert result["startup_device_error_reason"] == "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    assert result["startup_recovery_action"] == "register_hardware"
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_with_only_registered_mic_prompts_register_first(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("other-mic", "Other Mic", inputs=2),
        ]
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert result["startup_device_error_reason"] == "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    assert result["startup_recovery_action"] == "register_hardware"
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_with_only_registered_speaker_prompts_register_first(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("other-speaker", "Other Speaker", outputs=2),
        ]
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert result["startup_device_error_reason"] == "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    assert result["startup_recovery_action"] == "register_hardware"
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_with_empty_registry_prompts_register_first(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    repository = _FakeHardwareRepository()
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert result["startup_device_error_reason"] == "已保存的注册硬件已删除或不存在，请先在硬件管理中注册硬件。"
    assert result["startup_recovery_action"] == "register_hardware"
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert applied == []
    assert saved == []


def test_v2_deleted_saved_hardware_list_assets_read_failure_is_retryable_unavailable(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "deleted-mic",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "speaker": {
                    "hardware_id": "deleted-speaker",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                    "samplerate": 44100,
                },
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )

    class FailingListRepository(_FakeHardwareRepository):
        def list_assets(self):
            self.list_assets_calls += 1
            raise sqlite3.OperationalError("database read failed")

    repository = FailingListRepository()
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []
    assert "已注册硬件当前不可用" in result["startup_device_error_reason"]
    assert "请先" not in result["startup_device_error_reason"]
    assert repository.get_asset_calls == ["deleted-mic", "deleted-speaker"]
    assert repository.list_assets_for_selection_calls == 1
    assert repository.list_assets_calls == 1
    assert applied == []
    assert saved == []


def test_v2_repository_read_failure_is_unavailable_even_with_valid_legacy_matching(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "mic-1",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )

    class FailingRepository(_FakeHardwareRepository):
        def get_asset(self, hardware_id):
            raise sqlite3.OperationalError("database read failed")

    repository = FailingRepository()
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []


def test_v2_repository_table_check_failure_is_unavailable_even_with_valid_legacy_matching(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "mic-1",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )

    class FailingRepository(_FakeHardwareRepository):
        def tables_exist(self):
            self.tables_exist_calls += 1
            raise sqlite3.OperationalError("database open failed")

    repository = FailingRepository()
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert repository.get_asset_calls == []
    assert applied == []
    assert saved == []


def test_v2_hardware_id_only_repository_read_failure_is_unavailable_and_retryable(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1"},
                "speaker": {"hardware_id": "speaker-1"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )

    class FailingRepository(_FakeHardwareRepository):
        def get_asset(self, hardware_id):
            raise sqlite3.OperationalError("database read failed")

    repository = FailingRepository()
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []


def test_v2_list_channels_read_failure_is_registered_hardware_unavailable(monkeypatch, tmp_path):
    runtime_mic = _device(5, "Runtime Mic", inputs=2)
    runtime_speaker = _device(6, "Runtime Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [runtime_mic], "output": [runtime_speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1"},
                "speaker": {"hardware_id": "speaker-1"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )

    class FailingChannelRepository(_FakeHardwareRepository):
        def list_channels(self, hardware_id, direction=None):
            self.list_channels_calls.append((hardware_id, direction))
            raise sqlite3.OperationalError("database channel read failed")

    repository = FailingChannelRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert "已注册硬件当前不可用" in result["startup_device_error_reason"]
    assert "麦克风通道" not in result["startup_device_error_reason"]
    assert repository.list_channels_calls == [("mic-1", "input")]
    assert applied == []
    assert saved == []


def test_v2_non_integer_registered_channel_index_is_unavailable_without_legacy_fallback(monkeypatch, tmp_path):
    mic = _device(5, "Legacy Mic", inputs=2)
    speaker = _device(6, "Legacy Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {
                    "hardware_id": "mic-1",
                    "name": "Legacy Mic",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "speaker": {
                    "hardware_id": "speaker-1",
                    "name": "Legacy Speaker",
                    "hostapi_name": "API",
                    "default_samplerate": 48000.0,
                },
                "mic_channels": [1],
            }
        ),
        encoding="utf-8",
    )
    bad_channel = _input_channel("mic-1", 1)
    bad_channel["channel_index"] = "not-an-integer"
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Legacy Mic", inputs=2),
            _registered_asset("speaker-1", "Legacy Speaker", outputs=2),
        ],
        channels={"mic-1": [bad_channel]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []


def test_v2_hardware_id_only_channel_row_parse_failure_is_unavailable_and_retryable(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "mic-1"},
                "speaker": {"hardware_id": "speaker-1"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )
    bad_channel = _input_channel("mic-1", 0)
    bad_channel["channel_index"] = "not-an-integer"
    repository = _FakeHardwareRepository(
        assets=[
            _registered_asset("mic-1", "Runtime Mic", inputs=2),
            _registered_asset("speaker-1", "Runtime Speaker", outputs=2),
        ],
        channels={"mic-1": [bad_channel]},
    )
    monkeypatch.setattr(module, "HardwareManagementRepository", lambda: repository, raising=False)
    applied = []
    saved = []
    monkeypatch.setattr(SoundDeviceManager, "change_default_device", staticmethod(lambda *args: applied.append(args)))
    monkeypatch.setattr(SoundDeviceManager, "save_selected_devices", staticmethod(lambda *args: saved.append(args)))

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []
    assert applied == []
    assert saved == []


def test_invalid_saved_channels_make_startup_unavailable(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "mic": {"name": "Saved Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {"name": "Saved Speaker", "hostapi_name": "API", "default_samplerate": 48000.0},
                "mic_channels": [99],
            }
        ),
        encoding="utf-8",
    )

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["can_retry_saved_devices"] is True
    assert result["mic_channels"] == []


def test_partially_invalid_saved_channels_make_startup_unavailable(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "mic": {"name": "Saved Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {"name": "Saved Speaker", "hostapi_name": "API", "default_samplerate": 48000.0},
                "mic_channels": [0, 99],
            }
        ),
        encoding="utf-8",
    )

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []


def test_unparseable_saved_channels_make_startup_unavailable(monkeypatch, tmp_path):
    mic = _device(5, "Saved Mic", inputs=2)
    speaker = _device(6, "Saved Speaker", outputs=2)
    config_path = _install_audio(
        monkeypatch,
        tmp_path,
        devices_by_api={"API": {"input": [mic], "output": [speaker]}},
    )
    config_path.write_text(
        json.dumps(
            {
                "mic": {"name": "Saved Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {"name": "Saved Speaker", "hostapi_name": "API", "default_samplerate": 48000.0},
                "mic_channels": [0, "bad"],
            }
        ),
        encoding="utf-8",
    )

    result = SoundDeviceManager().get_startup_devices()

    assert result["device_available"] is False
    assert result["mic"] is None
    assert result["speaker"] is None
    assert result["mic_channels"] == []


def test_clear_selected_devices_for_deleted_hardware_removes_matching_config(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "deleted-mic", "name": "Saved Mic"},
                "speaker": {"hardware_id": "speaker-1", "name": "Saved Speaker"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )

    result = SoundDeviceManager.clear_selected_devices_for_deleted_hardware("deleted-mic")

    assert result.status == "cleared"
    assert result.matched is True
    assert result.cleared is True
    assert result.clear_failed is False
    assert not config_path.exists()


def test_clear_selected_devices_for_deleted_hardware_preserves_non_matching_config(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    payload = {
        "version": 2,
        "mic": {"hardware_id": "mic-1", "name": "Saved Mic"},
        "speaker": {"hardware_id": "speaker-1", "name": "Saved Speaker"},
        "mic_channels": [0],
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = SoundDeviceManager.clear_selected_devices_for_deleted_hardware("other-hardware")

    assert result.status == "no_match"
    assert result.matched is False
    assert result.cleared is False
    assert result.clear_failed is False
    assert json.loads(config_path.read_text(encoding="utf-8")) == payload


def test_clear_selected_devices_for_deleted_hardware_ignores_invalid_config(monkeypatch, tmp_path):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text("{bad json", encoding="utf-8")

    result = SoundDeviceManager.clear_selected_devices_for_deleted_hardware("mic-1")

    assert result.status == "no_match"
    assert result.matched is False
    assert result.cleared is False
    assert result.clear_failed is False
    assert config_path.read_text(encoding="utf-8") == "{bad json"


def test_clear_selected_devices_for_deleted_hardware_reports_matching_config_clear_failure(
    monkeypatch, tmp_path
):
    config_path = _install_audio(monkeypatch, tmp_path)
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "mic": {"hardware_id": "deleted-mic", "name": "Saved Mic"},
                "speaker": {"hardware_id": "speaker-1", "name": "Saved Speaker"},
                "mic_channels": [0],
            }
        ),
        encoding="utf-8",
    )

    def fail_remove(path):
        assert path == str(config_path)
        raise OSError("permission denied")

    monkeypatch.setattr(os, "remove", fail_remove)

    result = SoundDeviceManager.clear_selected_devices_for_deleted_hardware("deleted-mic")

    assert result.status == "clear_failed"
    assert result.matched is True
    assert result.cleared is False
    assert result.clear_failed is True
    assert "permission denied" in result.error
    assert config_path.exists()
