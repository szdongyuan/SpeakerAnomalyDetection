from unittest import mock
from copy import deepcopy

import pytest

from base.soundcard_calibration_manager import (
    MicCalibrationFormatError,
    MicCalibrationIOError,
)
from base.recording_process_protocol import FrozenConfig
from base.ve3668n_stores import VECalibrationStore, VEStoreIOError
from unit_test.base.ve3668n_fakes import device_info, input_config, wav_metadata_v2


DEVICE = {
    "index": 7,
    "name": "Test Microphone",
    "hostapi": 3,
    "max_input_channels": 4,
}


def _record(factor, standard_spl):
    return {
        "v2pa_factor": factor,
        "standard_spl_db": standard_spl,
        "sample_rate_hz": 48000,
        "duration_seconds": 5.0,
        "calibrated_at": "2026-08-25T10:00:00+08:00",
    }


def test_snapshot_maps_physical_input_order_to_wav_local_channels():
    from base.recording_calibration_snapshot import (
        build_recording_wav_calibration_metadata,
    )

    with mock.patch(
        "base.recording_calibration_snapshot.load_mic_channel_calibrations",
        return_value={0: _record(1.25, 94), 2: _record(3.5, 114.0)},
    ) as loader:
        metadata = build_recording_wav_calibration_metadata([0, 2], DEVICE)

    loader.assert_called_once_with(DEVICE, None)
    assert metadata == {
        "recorded_channels": [
            {
                "wav_channel_index": 0,
                "physical_input_channel": 0,
                "v2pa_factor": 1.25,
                "standard_spl": 94,
                "calibrated": True,
            },
            {
                "wav_channel_index": 1,
                "physical_input_channel": 2,
                "v2pa_factor": 3.5,
                "standard_spl": 114.0,
                "calibrated": True,
            },
        ]
    }


def test_snapshot_keeps_missing_physical_channel_as_uncalibrated():
    from base.recording_calibration_snapshot import (
        build_recording_wav_calibration_metadata,
    )

    with mock.patch(
        "base.recording_calibration_snapshot.load_mic_channel_calibrations",
        return_value={2: _record(3.5, 94.0)},
    ):
        metadata = build_recording_wav_calibration_metadata([0, 2], DEVICE)

    assert metadata["recorded_channels"][0] == {
        "wav_channel_index": 0,
        "physical_input_channel": 0,
        "v2pa_factor": None,
        "standard_spl": None,
        "calibrated": False,
    }
    assert metadata["recorded_channels"][1]["physical_input_channel"] == 2
    assert metadata["recorded_channels"][1]["calibrated"] is True


@pytest.mark.parametrize("channels", [None, [], [True], [-1], [0, 0], [0.5]])
def test_snapshot_rejects_empty_or_invalid_physical_channels(channels):
    from base.recording_calibration_snapshot import (
        build_recording_wav_calibration_metadata,
    )

    with mock.patch(
        "base.recording_calibration_snapshot.load_mic_channel_calibrations"
    ) as loader:
        with pytest.raises(ValueError):
            build_recording_wav_calibration_metadata(channels, DEVICE)
    loader.assert_not_called()


def test_snapshot_invalid_device_identity_produces_complete_uncalibrated_map():
    from base.recording_calibration_snapshot import (
        build_recording_wav_calibration_metadata,
    )

    with mock.patch(
        "base.recording_calibration_snapshot.load_mic_channel_calibrations",
        return_value={},
    ) as loader:
        metadata = build_recording_wav_calibration_metadata([0, 2], None)

    loader.assert_called_once_with(None, None)
    assert [item["calibrated"] for item in metadata["recorded_channels"]] == [
        False,
        False,
    ]


@pytest.mark.parametrize(
    "error",
    [MicCalibrationFormatError("bad"), MicCalibrationIOError("denied")],
)
def test_snapshot_propagates_calibration_boundary_errors(error):
    from base.recording_calibration_snapshot import (
        build_recording_wav_calibration_metadata,
    )

    with mock.patch(
        "base.recording_calibration_snapshot.load_mic_channel_calibrations",
        side_effect=error,
    ) as loader:
        with pytest.raises(type(error), match=str(error)):
            build_recording_wav_calibration_metadata([0], DEVICE, "cal.json")
    loader.assert_called_once_with(DEVICE, "cal.json")


def _ve_store(tmp_path):
    device = device_info(machine_id="TEST-DEVICE")
    store = VECalibrationStore(tmp_path / "calibrations.json")
    store.save(device, 7, v2pa_factor=10.0, standard_spl=94.0,
               calibration_sample_rate=51200, calibration_duration_seconds=10.0,
               calibrated_at="2026-08-28T10:00:00+08:00")
    return device, store


@pytest.mark.parametrize("backend", ["soundcard", "ve"])
def test_direct_factor_snapshot_wav_and_resolver_remain_frozen_after_update(tmp_path, monkeypatch, backend):
    import numpy as np
    from scipy.io import wavfile
    from base import soundcard_calibration_manager as manager
    from base.recording_calibration_snapshot import build_recording_wav_calibration_metadata
    from base.wav_calibration_metadata import (
        append_wav_calibration_metadata, inspect_wav_calibration_metadata,
        resolve_wav_channel_v2pa_factor,
    )

    factor = 0.12345678901234567
    timestamp = "2026-10-07T15:00:00+08:00"
    if backend == "ve":
        device = device_info()
        store = VECalibrationStore(tmp_path / "ve.json")

        def save(value):
            store.save_factor(device, 7, v2pa_factor=value, calibrated_at=timestamp)

        def snapshot():
            return build_recording_wav_calibration_metadata(
                (7, 1), device, ve_calibration_store=store,
            )
    else:
        monkeypatch.setattr(manager.SoundDeviceManager, "get_api_info",
                            lambda index: {"name": "Test API"})
        path = tmp_path / "mic.json"

        def save(value):
            manager.save_mic_channel_factor(
                value, DEVICE, 2, calibration_path=str(path), calibrated_at=timestamp,
            )

        def snapshot():
            return build_recording_wav_calibration_metadata((2, 0), DEVICE, str(path))

    save(factor)
    frozen = FrozenConfig.snapshot(snapshot())
    expected = frozen.to_dict()
    channel = expected["recorded_channels"][0]
    assert channel["calibrated"] is True
    assert channel["v2pa_factor"] == factor
    if backend == "ve":
        assert expected["schema_version"] == 2
        assert channel["factor_source"] == "calibrated"
        assert channel["calibration"]["standard_spl"] is None
        sample_rate = expected["acquisition"]["sample_rate"]
    else:
        assert channel["standard_spl"] is None
        sample_rate = 48000
    wav_path = tmp_path / "recording.wav"
    audio = np.array([[0.25, -0.5], [-0.75, 1.0]], dtype=np.float32)
    wavfile.write(wav_path, sample_rate, audio)
    original_audio = wav_path.read_bytes()
    assert append_wav_calibration_metadata(wav_path, frozen.to_dict())
    written = wav_path.read_bytes()
    save(99.0)
    assert snapshot()["recorded_channels"][0]["v2pa_factor"] == 99.0
    assert frozen.to_dict() == expected
    result = inspect_wav_calibration_metadata(wav_path)
    assert result.metadata == expected
    resolution = resolve_wav_channel_v2pa_factor(result.metadata, 0)
    assert resolution.factor == factor
    assert resolution.has_valid_metadata and resolution.used_file_metadata
    assert wav_path.read_bytes() == written
    assert written[8:len(original_audio)] == original_audio[8:]
    np.testing.assert_array_equal(wavfile.read(wav_path)[1], audio)


@pytest.mark.parametrize("limit", [10.0, 5.0, 2.5, 1.0, 0.5, 0.1, 0.02])
def test_ve_route_uses_explicit_store_and_current_profile_without_legacy_lookup(tmp_path, limit):
    from base.recording_calibration_snapshot import build_recording_wav_calibration_metadata

    device, store = _ve_store(tmp_path)
    device['input_config'] = input_config(44100, range_min=-limit, range_max=limit)
    original = store.path.read_bytes()
    with mock.patch('base.recording_calibration_snapshot.load_mic_channel_calibrations') as legacy:
        actual = build_recording_wav_calibration_metadata(
            (7, 1), FrozenConfig.snapshot(device), ve_calibration_store=store,
        )
    legacy.assert_not_called()
    expected = wav_metadata_v2()
    expected['acquisition'].update(range_min=-limit, range_max=limit)
    assert actual == expected
    assert store.path.read_bytes() == original


def test_ve_route_requires_store_without_creating_or_loading_a_fallback():
    from base.recording_calibration_snapshot import build_recording_wav_calibration_metadata

    with mock.patch('base.recording_calibration_snapshot.load_mic_channel_calibrations') as legacy:
        with pytest.raises(ValueError, match='store'):
            build_recording_wav_calibration_metadata((7, 1), device_info())
    legacy.assert_not_called()


def test_ve_builder_uses_one_observation_and_freezes_original_provenance(tmp_path):
    from base import ve3668n_wav_metadata

    device, store = _ve_store(tmp_path)
    records = store.observe(device)

    class ObservedStore:
        def __init__(self):
            self.calls = []

        def observe(self, current):
            self.calls.append(deepcopy(current))
            return records  # Deliberately shared: the builder must own its result.

    observed = ObservedStore()
    profile = input_config(44100)
    actual = ve3668n_wav_metadata.build_ve_recording_metadata(
        FrozenConfig.snapshot(device), (7, 1), FrozenConfig.snapshot(profile), observed,
    )
    assert actual == wav_metadata_v2()
    assert len(observed.calls) == 1
    assert observed.calls[0]['input_config']['sample_rate'] == 44100
    frozen = FrozenConfig.snapshot(actual)
    records[7]['v2pa_factor'] = 200.0
    records[7]['calibration_sample_rate'] = 48000
    profile['sample_rate'] = 51200
    device['machine_id'] = 'other-device'
    store.reset(observed.calls[0], 7)
    assert actual == wav_metadata_v2()
    assert frozen.to_dict() == wav_metadata_v2()


def test_invalidated_record_becomes_none_snapshot_and_is_not_revived_by_rate(tmp_path):
    from base import ve3668n_wav_metadata

    device, store = _ve_store(tmp_path)
    changed = {**device, 'input_config': input_config(range_max=5.0)}
    with pytest.raises(ValueError):
        store.observe(changed)
    actual = ve3668n_wav_metadata.build_ve_recording_metadata(
        device, (7, 1), input_config(44100), store,
    )
    assert actual == wav_metadata_v2(('none', 'none'))
    assert store.get_record(device, 7)['status'] == 'invalidated'
    assert store.get_factor(device, 7) is None


def test_ve_builder_propagates_store_failure():
    from base import ve3668n_wav_metadata

    class BrokenStore:
        def observe(self, device):
            raise VEStoreIOError('cannot observe calibration')

    with pytest.raises(VEStoreIOError, match='cannot observe'):
        ve3668n_wav_metadata.build_ve_recording_metadata(
            device_info(), (7, 1), input_config(), BrokenStore(),
        )


@pytest.mark.parametrize('channels', [(7, 7), (8,), (True,), (0,), ()])
def test_ve_builder_rejects_invalid_or_unavailable_physical_channels(tmp_path, channels):
    from base import ve3668n_wav_metadata

    device, store = _ve_store(tmp_path)
    with pytest.raises(ValueError):
        ve3668n_wav_metadata.build_ve_recording_metadata(device, channels, input_config(), store)


@pytest.mark.parametrize('profile', [input_config(3999), input_config(True),
                                   input_config(sensitivity=1000), None])
def test_ve_builder_requires_a_valid_current_profile_even_though_file_rates_are_historical(
    tmp_path, profile,
):
    from base import ve3668n_wav_metadata

    device, store = _ve_store(tmp_path)
    original = store.path.read_bytes()
    with pytest.raises(ValueError):
        ve3668n_wav_metadata.build_ve_recording_metadata(device, (7, 1), profile, store)
    assert store.path.read_bytes() == original
