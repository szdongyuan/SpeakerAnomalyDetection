import json

import base.sound_device_manager as module
from base.sound_device_manager import SoundDeviceManager


def _device(index, name, hostapi=0, samplerate=48000.0, inputs=0, outputs=0):
    return {
        "index": index,
        "name": name,
        "hostapi": hostapi,
        "default_samplerate": samplerate,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


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
                "mic": {"name": "Saved Mic", "hostapi_name": "API", "default_samplerate": 48000.0},
                "speaker": {"name": "Saved Speaker", "hostapi_name": "API", "default_samplerate": 48000.0},
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
    assert result["mic"] == mic
    assert result["speaker"] == speaker
    assert result["mic_channels"] == [1]
    assert applied == [(5, 6)]
    assert saved == [(mic, speaker, [1])]


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
