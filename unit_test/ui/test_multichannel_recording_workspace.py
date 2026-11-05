import logging

import numpy as np
import pytest

from base.recording_channel_selection import RecordingChannelSelectionError
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin


class _ChannelPanelSpy:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.channel_calls = []

    def set_channels(self, channels):
        snapshot = tuple(channels)
        self.channel_calls.append(snapshot)
        self.events.append(("set_channels", snapshot))


class _SelectionHost(SequenceWidgetUiOpsMixin):
    def __init__(self, channels=(0,), maximum=4):
        self.mic_channels = list(channels)
        self.mic = (
            {} if maximum is None else {"max_input_channels": maximum}
        )
        self.events = []
        self.left_panel = _ChannelPanelSpy(self.events)
        self.default_logger = logging.getLogger(__name__)
        self._configured_input_channels = None
        self._recording_input_channels = None
        self._pending_configured_input_channels = None
        self._channel_selection_error = ""
        self._active_input_channels = [0]


def test_valid_idle_refresh_syncs_left_channels_and_clears_selection_error():
    host = _SelectionHost(channels=(2, 0))
    host._channel_selection_error = "old error"

    host.refresh_channel_windows()

    assert host._configured_input_channels == (0, 2)
    assert host._pending_configured_input_channels is None
    assert host._active_input_channels == [0, 2]
    assert host._channel_selection_error == ""
    assert host.left_panel.channel_calls == [(0, 2)]


def test_invalid_refresh_keeps_last_valid_channels_and_blocks_next_recording():
    host = _SelectionHost(channels=(0, 2))
    host.refresh_channel_windows()
    host.left_panel.channel_calls.clear()
    host.mic_channels = [4]

    host.refresh_channel_windows()

    assert host._configured_input_channels == (0, 2)
    assert host._active_input_channels == [0, 2]
    assert host.left_panel.channel_calls == []
    assert "outside device range" in host._channel_selection_error
    with pytest.raises(RecordingChannelSelectionError, match="outside device range"):
        host._snapshot_recording_input_channels({})


def test_invalid_first_refresh_keeps_placeholder_but_not_valid_capture_state():
    host = _SelectionHost(channels=(), maximum=None)

    host.refresh_channel_windows()

    assert host._configured_input_channels is None
    assert host._active_input_channels == [0]
    assert host.left_panel.channel_calls == []
    assert "cannot verify" in host._channel_selection_error


def test_unknown_capability_accepts_nonempty_selection_and_skips_only_upper_bound():
    host = _SelectionHost(channels=(5, 2), maximum=None)

    host.refresh_channel_windows()

    assert host._configured_input_channels == (2, 5)
    assert host._channel_selection_error == ""


def test_refresh_during_recording_updates_pending_only():
    host = _SelectionHost(channels=(2, 0))
    host._configured_input_channels = (0,)
    host._recording_input_channels = (0,)

    host.refresh_channel_windows()

    assert host._configured_input_channels == (0, 2)
    assert host._pending_configured_input_channels == (0, 2)
    assert host._recording_input_channels == (0,)
    assert host._active_input_channels == [0]
    assert host.events == []


def test_terminal_cleanup_applies_different_pending_selection_once():
    host = _SelectionHost()
    host._recording_input_channels = (0,)
    host._configured_input_channels = (0, 2)
    host._pending_configured_input_channels = (0, 2)

    host._finalize_recording_channel_selection()
    host._finalize_recording_channel_selection()

    assert host._recording_input_channels is None
    assert host._pending_configured_input_channels is None
    assert host._active_input_channels == [0, 2]
    assert host.left_panel.channel_calls == [(0, 2)]


def test_terminal_cleanup_drops_identical_pending_selection():
    host = _SelectionHost()
    host._recording_input_channels = (0, 2)
    host._configured_input_channels = (0, 2)
    host._pending_configured_input_channels = (0, 2)
    host._active_input_channels = [0, 2]

    host._finalize_recording_channel_selection()

    assert host._recording_input_channels is None
    assert host._pending_configured_input_channels is None
    assert host.events == []


def test_failed_startup_clears_run_snapshot_and_applies_pending_selection():
    host = _SelectionHost()
    host._recording_input_channels = (0,)
    host._configured_input_channels = (1, 2)
    host._pending_configured_input_channels = (1, 2)

    host._abort_recording_channel_selection()

    assert host._recording_input_channels is None
    assert host._pending_configured_input_channels is None
    assert host._active_input_channels == [1, 2]
    assert host.left_panel.channel_calls == [(1, 2)]


def test_idle_mapping_change_synchronizes_left_channel_list():
    host = _SelectionHost(channels=(2, 0))
    host._configured_input_channels = (0,)

    host.refresh_channel_windows()

    assert host.events == [
        ("set_channels", (0, 2)),
    ]


def test_identical_idle_mapping_is_a_noop():
    host = _SelectionHost(channels=(2, 0))
    host._configured_input_channels = (0, 2)
    host._active_input_channels = [0, 2]

    host.refresh_channel_windows()

    assert host.events == []


def test_recording_start_copies_an_immutable_run_snapshot_to_api_boundaries():
    host = _SelectionHost(channels=(2, 0))
    host._configured_input_channels = (0, 2)
    recorded_dict = {}

    run_snapshot = host._snapshot_recording_input_channels(recorded_dict)
    host.mic_channels.append(3)

    assert run_snapshot == (0, 2)
    assert host._recording_input_channels == (0, 2)
    assert host._active_input_channels == [0, 2]
    assert recorded_dict["input_channels"] == [0, 2]
    assert recorded_dict["channels"] == 2


def test_final_array_normalizer_preserves_exact_multichannel_columns():
    recorded = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=np.float64,
    )

    normalized = SequenceWidgetStreamingOpsMixin._normalize_final_recording_array(
        recorded,
        (0, 2),
    )

    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(normalized, recorded.astype(np.float32))


def test_final_array_normalizer_accepts_one_dimensional_single_channel():
    recorded = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    normalized = SequenceWidgetStreamingOpsMixin._normalize_final_recording_array(
        recorded,
        (2,),
    )

    assert normalized.shape == (3, 1)
    np.testing.assert_array_equal(normalized[:, 0], recorded)


@pytest.mark.parametrize(
    "recorded",
    [
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.ones((3, 1), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32),
    ],
)
def test_final_array_normalizer_rejects_shape_mismatch_without_remapping(recorded):
    with pytest.raises(ValueError, match="expected 2 channels"):
        SequenceWidgetStreamingOpsMixin._normalize_final_recording_array(
            recorded,
            (0, 2),
        )
