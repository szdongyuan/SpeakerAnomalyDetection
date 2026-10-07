"""Real ordinary input producers with in-memory audio inventory only."""
import json
import pickle
from types import SimpleNamespace

import pytest

from base import hardware_selection, sound_device_manager
from consts import error_code
from unit_test.base.test_recording_capture import request


@pytest.fixture
def inventory(monkeypatch):
    devices = [dict(index=0, name="Full microphone name", hostapi=0,
                    max_input_channels=3, max_output_channels=2)]
    apis = [dict(name="Selected API", devices=[0], default_output_device=0)]
    backend = SimpleNamespace(
        default=SimpleNamespace(_default_device=[0, 0], hostapi=0, device=[0, 0]),
        query_devices=lambda index=None: devices if index is None else devices[index],
        query_hostapis=lambda index=None: apis if index is None else apis[index],
        _terminate=lambda: None, _initialize=lambda: None)
    monkeypatch.setattr(sound_device_manager, "sd", backend)
    return devices


def test_enumerated_input_has_api_identity_without_changing_output(inventory):
    group = sound_device_manager.SoundDeviceManager.get_device_info()["Selected API"]
    assert group["input"][0]["hostapi_name"] == "Selected API"
    assert "hostapi_name" not in group["output"][0]
    assert "hostapi_name" not in inventory[0]


def test_default_input_carries_identity(inventory):
    code, device = sound_device_manager.SoundDeviceManager().get_default_device("mic", refresh=False)
    assert code == error_code.OK
    assert device == {**inventory[0], "hostapi_name": "Selected API"}


def test_restored_input_keeps_identity_without_persisting_new_field(tmp_path, inventory):
    path = tmp_path / "hardware.json"
    saved = {"api_name": "Selected API", "mic_name": inventory[0]["name"], "mic_channels": [2]}
    path.write_text(json.dumps(saved), encoding="utf-8")
    mic, _, channels, _ = hardware_selection.restore_or_default(path=str(path), apply_defaults=False)
    assert mic["hostapi_name"] == "Selected API"
    assert channels == [2]
    hardware_selection.save_if_changed(mic, None, channels, [], path=str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == saved


@pytest.mark.parametrize("value", [None, "", "  ", 1, False, []])
def test_invalid_explicit_hostapi_name_is_rejected(tmp_path, value):
    device = dict(request(tmp_path).device, hostapi_name=value)
    with pytest.raises(ValueError, match="hostapi_name"):
        request(tmp_path, device=device)


def test_calibration_identity_survives_pickle_without_changing_parameters(tmp_path, inventory):
    _, device = sound_device_manager.SoundDeviceManager().get_default_device("mic", refresh=False)
    metadata = {"factor": 2.5, "source": "selected input"}
    req = request(tmp_path, device=device, purpose="calibration", channels=(2,),
                  sample_rate=48000, trim_samples=0, calibration_metadata=metadata)
    restored = pickle.loads(pickle.dumps(req))
    assert restored == req
    assert restored.device["hostapi_name"] == "Selected API"
    assert restored.sample_rate == 48000 and restored.channels == (2,)
    assert dict(restored.calibration_metadata) == metadata


def test_legacy_snapshot_remains_valid(tmp_path):
    assert "hostapi_name" not in request(tmp_path).device
