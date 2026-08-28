import logging
from types import SimpleNamespace

import numpy as np
import pytest

from base.recording_channel_selection import RecordingChannelSelectionError
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin


class _WorkspaceSpy:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.channel_calls = []
        self.clear_count = 0

    def set_channels(self, channels):
        snapshot = tuple(channels)
        self.channel_calls.append(snapshot)
        self.events.append(("set_channels", snapshot))

    def clear_plots(self):
        self.clear_count += 1
        self.events.append(("clear_plots",))


class _SelectionHost(SequenceWidgetUiOpsMixin):
    def __init__(self, channels=(0,), maximum=4):
        self.mic_channels = list(channels)
        self.mic = (
            {} if maximum is None else {"max_input_channels": maximum}
        )
        self.events = []
        self.channel_workspace = _WorkspaceSpy(self.events)
        self.default_logger = logging.getLogger(__name__)
        self._configured_input_channels = None
        self._recording_input_channels = None
        self._pending_configured_input_channels = None
        self._channel_selection_error = ""
        self._active_input_channels = [0]
        self._waveform_presentation_owner = "hardware"

    def _end_streaming_waveform_session(self):
        self.events.append(("release_live_display",))


class _ProjectionWindow:
    def __init__(self, channel):
        self.channel_index = channel
        self.calls = []
        self.current_data = None
        self.fail_next_set = False

    def set_data(self, time_axis, amplitude):
        if self.fail_next_set:
            self.fail_next_set = False
            raise RuntimeError("second channel Qt failure")
        current_data = (
            np.asarray(time_axis).copy(),
            np.asarray(amplitude).copy(),
        )
        self.current_data = current_data
        self.calls.append(current_data)

    def snapshot_plot_state(self):
        if self.current_data is None:
            return None
        return tuple(value.copy() for value in self.current_data)

    def restore_plot_state(self, state):
        self.current_data = (
            None
            if state is None
            else tuple(value.copy() for value in state)
        )


class _ProjectionWorkspace:
    def __init__(self, channels):
        self.channel_calls = []
        self._windows = [_ProjectionWindow(channel) for channel in channels]

    def all_subwindows(self):
        return list(self._windows)

    def set_channels(self, channels):
        channels = tuple(channels)
        self.channel_calls.append(channels)
        self._windows = [_ProjectionWindow(channel) for channel in channels]

    def clear_plots(self):
        return None


class _ProjectionHost(SequenceWidgetStreamingOpsMixin):
    def __init__(self, channels=(0, 2)):
        self._active_input_channels = list(channels)
        self.channel_workspace = _ProjectionWorkspace(channels)


class _NewRecordingOwnershipHost(
    _SelectionHost,
    SequenceWidgetAnalysisOpsMixin,
):
    def __init__(self, *, fail_reset=False):
        super().__init__(channels=(0, 2))
        self._configured_input_channels = (0, 2)
        self._active_input_channels = [2, 4]
        self._waveform_presentation_owner = "direct_import"
        self._record_workflow_busy = False
        self.last_play_count = None
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.player_status_flag = False
        self.toolsbar = SimpleNamespace(
            replayer_btn=SimpleNamespace(
                setDisabled=lambda _value: None
            ),
            data_btn=SimpleNamespace(setDisabled=lambda _value: None),
        )
        self._fail_reset = fail_reset
        self.capture_count = 0
        self.process_count = 0

    def checked_work_status_message(self):
        return False

    def _close_analysis_windows(self):
        return None

    def _lock_sn_for_recording_if_needed(self):
        return None

    def _clear_plot_area(self):
        self.events.append(
            (
                "generic_clear",
                tuple(self._active_input_channels),
                self._waveform_presentation_owner,
            )
        )
        self.channel_workspace.clear_plots()

    def _cleanup_streaming_resources(self):
        return None

    def update_player_btn_is_playing(self):
        return None

    def reset_work_pram(self, _label, count=None):
        self.events.append(
            (
                "reset_enter",
                tuple(self._active_input_channels),
                self._waveform_presentation_owner,
            )
        )
        if self._fail_reset:
            raise RuntimeError("early reset failure")
        self._begin_new_recording_presentation()
        recorded_dict = {}
        self._snapshot_recording_input_channels(recorded_dict)
        return recorded_dict, 48000

    def _cleanup_failed_recording_initialization(self, _message):
        self._abort_recording_channel_selection()
        self._record_workflow_busy = False
        return True

    def _capture_recording_wav_calibration_metadata(self):
        self.capture_count += 1

    def _should_use_streaming_recording(self):
        return False

    def _begin_recent_session_for_current_run(self):
        return None

    def _start_process_recording(self, _recorded_dict, _sample_rate, *, tcp_completion_address=None):
        self.process_count += 1


def test_workspace_builds_ordered_physical_channel_windows(ui_qapp):
    workspace = ChannelPlotWorkspace()

    workspace.set_channels([0, 2, 3])
    ui_qapp.processEvents()

    windows = workspace.all_subwindows()
    assert [window.channel_index for window in windows] == [0, 2, 3]
    assert [window.title_bar.title_label.text() for window in windows] == ["In1", "In3", "In4"]
    assert workspace.all_subwindows() is not windows

    old_windows = windows
    workspace.set_channels([1])
    assert all(not window.isVisible() for window in old_windows)
    assert [window.channel_index for window in workspace.all_subwindows()] == [1]
    workspace.close()
    workspace.deleteLater()
    ui_qapp.processEvents()


def test_workspace_clear_plots_clears_every_ordered_window(ui_qapp, monkeypatch):
    workspace = ChannelPlotWorkspace()
    workspace.set_channels((0, 2, 3))
    ui_qapp.processEvents()
    cleared = []
    for window in workspace.all_subwindows():
        monkeypatch.setattr(window, "clear_plot", lambda ch=window.channel_index: cleared.append(ch))

    workspace.clear_plots()

    assert cleared == [0, 2, 3]
    workspace.close()
    workspace.deleteLater()
    ui_qapp.processEvents()


def test_valid_idle_refresh_rebuilds_workspace_and_clears_selection_error():
    host = _SelectionHost(channels=(2, 0))
    host._channel_selection_error = "old error"

    host.refresh_channel_windows()

    assert host._configured_input_channels == (0, 2)
    assert host._pending_configured_input_channels is None
    assert host._active_input_channels == [0, 2]
    assert host._channel_selection_error == ""
    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host.channel_workspace.clear_count == 1


def test_invalid_refresh_keeps_last_valid_workspace_and_blocks_next_recording():
    host = _SelectionHost(channels=(0, 2))
    host.refresh_channel_windows()
    host.channel_workspace.channel_calls.clear()
    host.channel_workspace.clear_count = 0
    host.mic_channels = [4]

    host.refresh_channel_windows()

    assert host._configured_input_channels == (0, 2)
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.channel_calls == []
    assert host.channel_workspace.clear_count == 0
    assert "outside device range" in host._channel_selection_error
    with pytest.raises(RecordingChannelSelectionError, match="outside device range"):
        host._snapshot_recording_input_channels({})


def test_invalid_first_refresh_keeps_placeholder_but_not_valid_capture_state():
    host = _SelectionHost(channels=(), maximum=None)

    host.refresh_channel_windows()

    assert host._configured_input_channels is None
    assert host._active_input_channels == [0]
    assert host.channel_workspace.channel_calls == []
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


def test_refresh_during_temporary_recent_view_updates_pending_only():
    host = _SelectionHost(channels=(2, 0))
    host._configured_input_channels = (0,)
    host._waveform_presentation_owner = "recent"
    host._active_input_channels = [3]

    host.refresh_channel_windows()

    assert host._configured_input_channels == (0, 2)
    assert host._pending_configured_input_channels == (0, 2)
    assert host._active_input_channels == [3]
    assert host.events == []


def test_terminal_cleanup_applies_different_pending_selection_once_and_clears_plots():
    host = _SelectionHost()
    host._recording_input_channels = (0,)
    host._configured_input_channels = (0, 2)
    host._pending_configured_input_channels = (0, 2)

    host._finalize_recording_channel_selection()
    host._finalize_recording_channel_selection()

    assert host._recording_input_channels is None
    assert host._pending_configured_input_channels is None
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host.channel_workspace.clear_count == 1


def test_terminal_cleanup_drops_identical_pending_selection_without_clearing_final_plot():
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
    assert host.channel_workspace.channel_calls == [(1, 2)]
    assert host.channel_workspace.clear_count == 1


def test_idle_mapping_change_invalidates_and_releases_stale_display_session():
    host = _SelectionHost(channels=(2, 0))
    host._configured_input_channels = (0,)

    host.refresh_channel_windows()

    assert host.events == [
        ("release_live_display",),
        ("set_channels", (0, 2)),
        ("clear_plots",),
    ]


def test_identical_idle_mapping_is_a_noop():
    host = _SelectionHost(channels=(2, 0))
    host._configured_input_channels = (0, 2)
    host._active_input_channels = [0, 2]

    host.refresh_channel_windows()

    assert host.events == []


def test_identical_hardware_refresh_still_ends_direct_import_and_clears_plot():
    host = _SelectionHost(channels=(2, 0))
    host._configured_input_channels = (0, 2)
    host._active_input_channels = [0, 2]
    host._waveform_presentation_owner = "direct_import"

    host.refresh_channel_windows()

    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.channel_calls == []
    assert host.channel_workspace.clear_count == 1


def test_new_recording_ends_direct_import_before_installing_run_snapshot():
    host = _SelectionHost(channels=(0, 2))
    host._configured_input_channels = (0, 2)
    host._active_input_channels = [2, 4]
    host._waveform_presentation_owner = "direct_import"

    run_channels = host._snapshot_recording_input_channels({})

    assert run_channels == (0, 2)
    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.clear_count == 1


def test_reset_ends_direct_import_and_restores_configured_hardware_workspace():
    host = _SelectionHost(channels=(0, 2))
    host._configured_input_channels = (0, 2)
    host._active_input_channels = [2, 4]
    host._waveform_presentation_owner = "direct_import"
    host.left_panel = None
    host._reset_product_condition_display_state = lambda: None

    SequenceWidgetAnalysisOpsMixin._reset_manual_product_condition_cycle(
        host,
        clear_waveforms=True,
    )

    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host.channel_workspace.clear_count == 1


def test_reset_without_valid_configured_mapping_still_clears_direct_import():
    host = _SelectionHost(channels=(), maximum=None)
    host._configured_input_channels = None
    host._active_input_channels = [2, 4]
    host._waveform_presentation_owner = "direct_import"
    host.left_panel = None
    host._reset_product_condition_display_state = lambda: None

    SequenceWidgetAnalysisOpsMixin._reset_manual_product_condition_cycle(
        host,
        clear_waveforms=True,
    )

    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == []
    assert host.channel_workspace.channel_calls == []
    assert host.channel_workspace.clear_count == 1


def test_new_recording_transitions_before_first_clear_and_early_reset_failure(
    ui_qapp,
):
    host = _NewRecordingOwnershipHost(fail_reset=True)

    host.judge_play_and_record()

    assert host.events[:3] == [
        ("release_live_display",),
        ("set_channels", (0, 2)),
        ("clear_plots",),
    ]
    assert ("reset_enter", (0, 2), "hardware") in host.events
    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.clear_count == 1


def test_successful_new_recording_transition_is_idempotent_through_snapshot(
    ui_qapp,
):
    host = _NewRecordingOwnershipHost()

    host.judge_play_and_record()

    assert host.events[:3] == [
        ("release_live_display",),
        ("set_channels", (0, 2)),
        ("clear_plots",),
    ]
    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host.channel_workspace.clear_count == 1
    assert host._recording_input_channels == (0, 2)
    assert host.capture_count == 1
    assert host.process_count == 1


def test_reset_work_pram_transitions_before_data_clear_can_fail():
    host = _SelectionHost(channels=(0, 2))
    host._configured_input_channels = (0, 2)
    host._active_input_channels = [2, 4]
    host._waveform_presentation_owner = "direct_import"
    host.data_struct = SimpleNamespace(
        clear_data=lambda: (_ for _ in ()).throw(
            RuntimeError("clear data failed")
        )
    )

    with pytest.raises(RuntimeError, match="clear data failed"):
        SequenceWidgetAnalysisOpsMixin.reset_work_pram(host, "label")

    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host.channel_workspace.clear_count == 1


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


def test_final_projection_draws_each_wav_column_to_its_physical_channel_window():
    host = _ProjectionHost((0, 2))
    recorded = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=np.float32,
    )

    host.plot_waveform_to_workspace(
        recorded,
        2.0,
        channel_mapping=(0, 2),
    )

    assert host.channel_workspace.channel_calls == []
    windows = host.channel_workspace.all_subwindows()
    np.testing.assert_array_equal(windows[0].calls[-1][1], recorded[:, 0])
    np.testing.assert_array_equal(windows[1].calls[-1][1], recorded[:, 1])
    np.testing.assert_array_equal(
        windows[0].calls[-1][0],
        np.asarray([0.0, 0.5, 1.0]),
    )


def test_final_projection_rebuilds_only_for_a_different_explicit_mapping():
    host = _ProjectionHost((0,))
    recorded = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)

    host.plot_waveform_to_workspace(
        recorded,
        2.0,
        channel_mapping=(0, 2),
    )

    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host._active_input_channels == [0, 2]


@pytest.mark.parametrize("has_live_projection", [False, True])
def test_final_projection_rolls_back_every_window_when_second_channel_fails(
    has_live_projection,
):
    host = _ProjectionHost((0, 2))
    windows = host.channel_workspace.all_subwindows()
    live_data = [
        (
            np.asarray([0.0, 0.5]),
            np.asarray([1.0, 2.0], dtype=np.float32),
        ),
        (
            np.asarray([0.0, 0.5]),
            np.asarray([10.0, 20.0], dtype=np.float32),
        ),
    ]
    if has_live_projection:
        for window, (time_axis, amplitude) in zip(windows, live_data):
            window.set_data(time_axis, amplitude)
    before = [window.snapshot_plot_state() for window in windows]
    windows[1].fail_next_set = True
    recorded = np.asarray(
        [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        dtype=np.float32,
    )

    with pytest.raises(RuntimeError, match="second channel Qt failure"):
        host.plot_waveform_to_workspace(
            recorded,
            2.0,
            channel_mapping=(0, 2),
        )

    after = [window.snapshot_plot_state() for window in windows]
    for before_state, after_state in zip(before, after):
        if before_state is None:
            assert after_state is None
        else:
            np.testing.assert_array_equal(after_state[0], before_state[0])
            np.testing.assert_array_equal(after_state[1], before_state[1])


def test_final_projection_restores_real_qt_windows_after_second_channel_failure(
    ui_qapp,
    monkeypatch,
):
    host = _ProjectionHost((0, 2))
    workspace = ChannelPlotWorkspace()
    workspace.set_channels((0, 2))
    host.channel_workspace = workspace
    windows = workspace.all_subwindows()
    live_time = np.asarray([0.0, 0.5])
    live_columns = (
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([10.0, 20.0], dtype=np.float32),
    )
    for window, live_column in zip(windows, live_columns):
        window.set_data(live_time, live_column)
    monkeypatch.setattr(
        windows[1],
        "set_data",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("Qt setData failed")),
    )

    with pytest.raises(RuntimeError, match="Qt setData failed"):
        host.plot_waveform_to_workspace(
            np.asarray([[3.0, 30.0], [4.0, 40.0]], dtype=np.float32),
            2.0,
            channel_mapping=(0, 2),
        )

    for window, live_column in zip(windows, live_columns):
        restored_time, restored_column = window.plot_item.getData()
        np.testing.assert_array_equal(restored_time, live_time)
        np.testing.assert_array_equal(restored_column, live_column)
    workspace.close()
    workspace.deleteLater()
    ui_qapp.processEvents()


def test_presentation_snapshot_restores_exact_channel_plot_states():
    host = _ProjectionHost((0, 2))
    host.default_logger = logging.getLogger(__name__)
    original_columns = (
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([10.0, 20.0], dtype=np.float32),
    )
    time_axis = np.asarray([0.0, 0.5])
    for window, column in zip(
        host.channel_workspace.all_subwindows(),
        original_columns,
    ):
        window.set_data(time_axis, column)
    states = SequenceWidgetAnalysisOpsMixin._snapshot_channel_workspace_plot_states(
        host
    )
    host.channel_workspace.set_channels((3, 4))

    restored = SequenceWidgetAnalysisOpsMixin._restore_channel_workspace_plot_states(
        host,
        (0, 2),
        states,
    )

    assert restored is True
    for window, column in zip(
        host.channel_workspace.all_subwindows(),
        original_columns,
    ):
        restored_time, restored_column = window.current_data
        np.testing.assert_array_equal(restored_time, time_axis)
        np.testing.assert_array_equal(restored_column, column)
