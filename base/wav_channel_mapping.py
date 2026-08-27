from numbers import Integral

import numpy as np

from base.recording_channel_selection import (
    RecordingChannelSelectionError,
    canonicalize_recording_input_channels,
)
from base.wav_calibration_metadata import (
    PHYSICAL_INPUT_CHANNEL_FIELD,
    WavCalibrationMetadataReadStatus,
)


class WavChannelMappingError(ValueError):
    """The authoritative WAV metadata cannot describe the audio columns."""


def resolve_wav_plot_channels(
    metadata_result,
    *,
    column_count,
    saved_active_input_channels=None,
):
    """Resolve ordered physical plot identities for WAV columns.

    Hardware state is intentionally absent from this API. A saved recent-session
    mapping is eligible only for valid legacy metadata without physical channel
    identities.
    """
    if (
        isinstance(column_count, bool)
        or not isinstance(column_count, Integral)
        or int(column_count) < 0
    ):
        raise WavChannelMappingError(
            "audio column count must be a non-negative integer"
        )
    expected_count = int(column_count)
    status = getattr(metadata_result, "status", None)

    if status is WavCalibrationMetadataReadStatus.INVALID:
        raise WavChannelMappingError(
            "invalid authoritative WAV calibration metadata"
        )
    if status is WavCalibrationMetadataReadStatus.ABSENT:
        return tuple(range(expected_count))
    if status is not WavCalibrationMetadataReadStatus.VALID:
        raise WavChannelMappingError(
            "invalid authoritative WAV metadata diagnostic result"
        )

    metadata = getattr(metadata_result, "metadata", None)
    recorded_channels = (
        metadata.get("recorded_channels")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(recorded_channels, list):
        raise WavChannelMappingError(
            "authoritative WAV metadata does not cover the audio columns"
        )

    by_wav_index = {}
    for channel in recorded_channels:
        wav_index = (
            channel.get("wav_channel_index")
            if isinstance(channel, dict)
            else None
        )
        if (
            isinstance(wav_index, bool)
            or not isinstance(wav_index, Integral)
            or int(wav_index) < 0
            or int(wav_index) in by_wav_index
        ):
            raise WavChannelMappingError(
                "authoritative WAV metadata does not cover the audio columns"
            )
        by_wav_index[int(wav_index)] = channel

    if tuple(sorted(by_wav_index)) != tuple(range(expected_count)):
        raise WavChannelMappingError(
            "authoritative WAV metadata does not cover the audio columns"
        )
    ordered_channels = [by_wav_index[index] for index in range(expected_count)]

    physical_presence = [
        PHYSICAL_INPUT_CHANNEL_FIELD in channel
        for channel in ordered_channels
    ]
    if any(physical_presence):
        if not all(physical_presence):
            raise WavChannelMappingError(
                "physical input channel metadata must be complete"
            )
        physical_channels = []
        for channel in ordered_channels:
            physical = channel[PHYSICAL_INPUT_CHANNEL_FIELD]
            if (
                isinstance(physical, (bool, np.bool_))
                or not isinstance(physical, Integral)
                or int(physical) < 0
                or int(physical) in physical_channels
            ):
                raise WavChannelMappingError(
                    "physical input channel metadata must contain unique non-negative integers"
                )
            physical_channels.append(int(physical))
        return tuple(physical_channels)

    recent_mapping = _strict_saved_channel_mapping(
        saved_active_input_channels,
        expected_count,
    )
    if recent_mapping is not None:
        return recent_mapping
    return tuple(range(expected_count))


def _strict_saved_channel_mapping(channels, expected_count):
    if channels is None:
        return None
    try:
        normalized = canonicalize_recording_input_channels(
            channels,
            max_input_channels=None,
        )
    except (RecordingChannelSelectionError, TypeError, ValueError):
        return None

    try:
        original = tuple(channels)
    except TypeError:
        return None
    if len(normalized) != expected_count or original != normalized:
        return None
    return normalized
