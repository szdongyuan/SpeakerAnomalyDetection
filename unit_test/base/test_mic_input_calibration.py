import json
from unittest import mock

import pytest

from base.soundcard_calibration_manager import (
    get_mic_v2pa_factor,
    load_mic_input_calibration,
    save_mic_input_calibration,
)


DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 2,
}


@pytest.fixture(autouse=True)
def host_api_name():
    with mock.patch(
        "base.soundcard_calibration_manager.SoundDeviceManager.get_api_info",
        return_value={"name": "Test API"},
    ):
        yield


def _save(path, **overrides):
    kwargs = {
        "v2pa_factor": 2.5,
        "input_device": DEVICE,
        "input_channel": 1,
        "standard_spl_db": 94.0,
        "sample_rate_hz": 44100,
        "duration_seconds": 10.0,
        "calibration_path": str(path),
        "calibrated_at": "2026-08-12T10:00:00+08:00",
    }
    kwargs.update(overrides)
    return save_mic_input_calibration(**kwargs)


def test_single_channel_calibration_round_trip(tmp_path):
    path = tmp_path / "mic_input_calibration.json"

    saved, _ = _save(path)

    assert saved is True
    assert get_mic_v2pa_factor(DEVICE, [1], str(path)) == 2.5
    payload = load_mic_input_calibration(str(path))
    assert payload["input"] == {
        "api_name": "Test API",
        "device_name": "Test Microphone",
        "channel_index": 1,
    }
    assert payload["calibration"]["standard_spl_db"] == 94.0


def test_device_index_change_does_not_invalidate_stable_identity(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    assert _save(path)[0] is True

    reindexed_device = {**DEVICE, "index": 99}

    assert get_mic_v2pa_factor(reindexed_device, [1], str(path)) == 2.5


def test_missing_json_never_falls_back_to_legacy_text(tmp_path):
    legacy_path = tmp_path / "mic_calibration.txt"
    legacy_path.write_text(
        "v2pa_factor: \n36.115\nDatetime: \n2025-03-10\n",
        encoding="utf-8",
    )

    assert get_mic_v2pa_factor(
        DEVICE,
        [1],
        str(tmp_path / "missing.json"),
    ) == 0.0


@pytest.mark.parametrize(
    "input_device,input_channels",
    [
        (DEVICE, []),
        (DEVICE, [0, 1]),
        (DEVICE, [0]),
        ({**DEVICE, "name": "Other Microphone"}, [1]),
        ({**DEVICE, "hostapi": 4}, [1]),
    ],
)
def test_factor_requires_exactly_matching_device_and_channel(
    tmp_path,
    input_device,
    input_channels,
):
    path = tmp_path / "mic_input_calibration.json"
    assert _save(path)[0] is True

    if input_device.get("hostapi") == 4:
        api_result = {"name": "Other API"}
    else:
        api_result = {"name": "Test API"}
    with mock.patch(
        "base.soundcard_calibration_manager.SoundDeviceManager.get_api_info",
        return_value=api_result,
    ):
        assert get_mic_v2pa_factor(input_device, input_channels, str(path)) == 0.0


@pytest.mark.parametrize(
    "factor",
    [0.0, -1.0, float("nan"), float("inf"), True, "2.5"],
)
def test_invalid_factor_is_not_saved(tmp_path, factor):
    path = tmp_path / "mic_input_calibration.json"

    saved, message = _save(path, v2pa_factor=factor)

    assert saved is False
    assert "无效" in message
    assert not path.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"version": 99},
        {"version": True},
        {
            "version": 1,
            "input": {
                "api_name": "Test API",
                "device_name": "Test Microphone",
                "channel_index": 1,
            },
            "calibration": {
                "v2pa_factor": "not-a-number",
                "standard_spl_db": 94.0,
                "sample_rate_hz": 44100,
                "duration_seconds": 10.0,
                "calibrated_at": "2026-08-12",
            },
        },
        {
            "version": 1,
            "input": {
                "api_name": "Test API",
                "device_name": "Test Microphone",
                "channel_index": 1,
            },
            "calibration": {
                "v2pa_factor": 2.5,
                "standard_spl_db": 94.0,
                "sample_rate_hz": "44100",
                "duration_seconds": 10.0,
                "calibrated_at": "2026-08-12",
            },
        },
    ],
)
def test_invalid_json_payload_is_rejected(tmp_path, payload):
    path = tmp_path / "mic_input_calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_mic_input_calibration(str(path)) is None
    assert get_mic_v2pa_factor(DEVICE, [1], str(path)) == 0.0


def test_atomic_save_failure_preserves_previous_calibration(tmp_path):
    path = tmp_path / "mic_input_calibration.json"
    assert _save(path, v2pa_factor=1.25)[0] is True
    original = path.read_bytes()

    with mock.patch(
        "base.soundcard_calibration_manager.os.replace",
        side_effect=PermissionError("read only"),
    ):
        saved, message = _save(path, v2pa_factor=3.0)

    assert saved is False
    assert "保存失败" in message
    assert path.read_bytes() == original
    assert get_mic_v2pa_factor(DEVICE, [1], str(path)) == 1.25
    assert list(tmp_path.glob("*.tmp")) == []
