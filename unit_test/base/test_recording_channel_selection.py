from fractions import Fraction

import numpy as np
import pytest

from base.recording_channel_selection import (
    RecordingChannelSelectionError,
    canonicalize_recording_input_channels,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([0, 2], (0, 2)),
        ([2, 0], (0, 2)),
        ([2, 0, 2], (0, 2)),
        (np.array([3, 1]), (1, 3)),
    ],
)
def test_canonicalize_sorts_and_deduplicates(raw, expected):
    assert canonicalize_recording_input_channels(raw, max_input_channels=4) == expected


def test_empty_selection_uses_legacy_first_channel_when_device_has_input():
    assert canonicalize_recording_input_channels([], max_input_channels=2) == (0,)


def test_unavailable_device_capability_skips_only_upper_bound_check():
    assert canonicalize_recording_input_channels([3, 1], max_input_channels=None) == (1, 3)


def test_empty_selection_fails_when_device_input_presence_is_unknown():
    with pytest.raises(RecordingChannelSelectionError, match="cannot verify an input channel"):
        canonicalize_recording_input_channels([], max_input_channels=None)


@pytest.mark.parametrize("raw", [[True], ["1"], [1.0], [1.5], [-1], [4], object()])
def test_invalid_selection_has_semantic_diagnostic(raw):
    with pytest.raises(RecordingChannelSelectionError, match="input channel"):
        canonicalize_recording_input_channels(raw, max_input_channels=4)


def test_empty_selection_fails_for_device_without_inputs():
    with pytest.raises(RecordingChannelSelectionError, match="no input channels"):
        canonicalize_recording_input_channels([], max_input_channels=0)


@pytest.mark.parametrize(
    "raw",
    [
        np.array(1),
        np.empty((0, 2), dtype=int),
        np.array([[0, 1]]),
    ],
)
def test_numpy_selection_must_be_one_dimensional(raw):
    with pytest.raises(RecordingChannelSelectionError, match="one-dimensional"):
        canonicalize_recording_input_channels(raw, max_input_channels=4)


@pytest.mark.parametrize(
    "maximum",
    [
        True,
        np.bool_(False),
        "4",
        4.0,
        1.5,
        Fraction(4, 1),
        Fraction(3, 2),
        -1,
    ],
)
def test_invalid_max_input_channels_has_semantic_diagnostic(maximum):
    with pytest.raises(RecordingChannelSelectionError, match="max input channels"):
        canonicalize_recording_input_channels([0], max_input_channels=maximum)
