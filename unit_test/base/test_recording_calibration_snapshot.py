from unittest import mock

import pytest

from base.soundcard_calibration_manager import (
    MicCalibrationFormatError,
    MicCalibrationIOError,
)


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
