import numpy as np
import pytest

from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadResult,
    WavCalibrationMetadataReadStatus,
)
from base.wav_channel_mapping import (
    WavChannelMappingError,
    resolve_wav_plot_channels,
)


def _result(status, channels=None):
    metadata = (
        None
        if channels is None
        else {"recorded_channels": channels}
    )
    return WavCalibrationMetadataReadResult(status, metadata)


def _valid(channels):
    return _result(WavCalibrationMetadataReadStatus.VALID, channels)


def _channel(wav_index, physical=...):
    result = {
        "wav_channel_index": wav_index,
        "v2pa_factor": None,
        "standard_spl": None,
        "calibrated": False,
    }
    if physical is not ...:
        result["physical_input_channel"] = physical
    return result


def test_new_metadata_wins_over_recent_session_mapping_and_orders_by_wav_index():
    result = resolve_wav_plot_channels(
        _valid([_channel(1, 4), _channel(0, 2)]),
        column_count=2,
        saved_active_input_channels=[0, 1],
    )

    assert result == (2, 4)


def test_valid_old_metadata_uses_exact_canonical_recent_session_mapping():
    result = resolve_wav_plot_channels(
        _valid([_channel(0), _channel(1)]),
        column_count=2,
        saved_active_input_channels=np.asarray([0, 3]),
    )

    assert result == (0, 3)


@pytest.mark.parametrize(
    ("metadata_result", "saved_channels", "column_count", "expected"),
    [
        (
            _result(WavCalibrationMetadataReadStatus.ABSENT),
            [7, 8],
            2,
            (0, 1),
        ),
        (_valid([_channel(0), _channel(1)]), None, 2, (0, 1)),
        (_valid([_channel(0), _channel(1)]), [3], 2, (0, 1)),
        (_valid([_channel(0), _channel(1)]), [0, "3"], 2, (0, 1)),
        (_valid([_channel(0), _channel(1)]), [3, 0], 2, (0, 1)),
        (_valid([_channel(0), _channel(1)]), [0, 0], 2, (0, 1)),
    ],
)
def test_legacy_mapping_falls_back_sequentially_without_current_hardware(
    metadata_result,
    saved_channels,
    column_count,
    expected,
):
    assert resolve_wav_plot_channels(
        metadata_result,
        column_count=column_count,
        saved_active_input_channels=saved_channels,
    ) == expected


def test_invalid_authoritative_metadata_is_rejected():
    with pytest.raises(WavChannelMappingError, match="invalid authoritative"):
        resolve_wav_plot_channels(
            _result(WavCalibrationMetadataReadStatus.INVALID),
            column_count=2,
        )


@pytest.mark.parametrize(
    "channels",
    [
        [_channel(0), _channel(0)],
        [_channel(0), _channel(2)],
        [_channel(0)],
        [_channel(0), _channel(1), _channel(2)],
    ],
)
def test_authoritative_metadata_must_cover_every_wav_column_once(channels):
    with pytest.raises(WavChannelMappingError, match="audio columns"):
        resolve_wav_plot_channels(
            _valid(channels),
            column_count=2,
            saved_active_input_channels=[4, 5],
        )


@pytest.mark.parametrize(
    "channels",
    [
        [_channel(0, 2), _channel(1)],
        [_channel(0, 2), _channel(1, 2)],
        [_channel(0, -1), _channel(1, 2)],
        [_channel(0, True), _channel(1, 2)],
        [_channel(0, 1.5), _channel(1, 2)],
    ],
)
def test_complete_physical_metadata_must_be_strict_and_unique(channels):
    with pytest.raises(WavChannelMappingError, match="physical input channel"):
        resolve_wav_plot_channels(
            _valid(channels),
            column_count=2,
        )


@pytest.mark.parametrize("column_count", [True, -1, 1.5])
def test_audio_column_count_is_a_non_negative_integer(column_count):
    with pytest.raises(WavChannelMappingError, match="column count"):
        resolve_wav_plot_channels(
            _result(WavCalibrationMetadataReadStatus.ABSENT),
            column_count=column_count,
        )
