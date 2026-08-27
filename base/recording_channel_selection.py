from numbers import Integral

import numpy as np


class RecordingChannelSelectionError(ValueError):
    pass


def canonicalize_recording_input_channels(channels, *, max_input_channels=None):
    maximum = None
    if max_input_channels is not None:
        if isinstance(max_input_channels, bool) or not isinstance(
            max_input_channels, Integral
        ):
            raise RecordingChannelSelectionError(
                "max input channels must be a non-negative integer"
            )
        maximum = int(max_input_channels)
        if maximum < 0:
            raise RecordingChannelSelectionError(
                "max input channels must be a non-negative integer"
            )

    if not isinstance(channels, (list, tuple, np.ndarray)):
        raise RecordingChannelSelectionError(
            "input channel selection must be a sequence"
        )
    if isinstance(channels, np.ndarray) and channels.ndim != 1:
        raise RecordingChannelSelectionError(
            "input channel selection must be one-dimensional"
        )
    if len(channels) == 0:
        if maximum is None:
            raise RecordingChannelSelectionError(
                "cannot verify an input channel for empty selection"
            )
        if maximum == 0:
            raise RecordingChannelSelectionError(
                "selected device has no input channels"
            )
        return (0,)

    normalized = []
    for raw in channels:
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Integral):
            raise RecordingChannelSelectionError(
                "each input channel must be an integer"
            )
        channel = int(raw)
        if channel < 0:
            raise RecordingChannelSelectionError(
                "input channel must be non-negative"
            )
        if maximum is not None and channel >= maximum:
            raise RecordingChannelSelectionError(
                f"input channel {channel} is outside device range 0..{maximum - 1}"
            )
        normalized.append(channel)

    return tuple(sorted(set(normalized)))
