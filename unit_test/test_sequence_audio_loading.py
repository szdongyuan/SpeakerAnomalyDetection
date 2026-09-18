import copy
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from ui.sequence.channel_plot_workspace import ChannelPlotSubWindow, ChannelPlotWorkspace
from base.wav_calibration_metadata import (
    WavCalibrationMetadataReadResult,
    WavCalibrationMetadataReadStatus,
)


class _ButtonSpy:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setDisabled(self, disabled):
        self.enabled = not bool(disabled)


class _AnalysisInstance:
    _sequence_analysis_key = "spl"
    _channel_mismatch = False
    _channel_mismatch_info = None

    def __init__(self, host):
        self.host = host

    def calculate_spl(self):
        self.host.data_struct.analysis_result_dict["SPL"] = (True, 0.0)
        return True

    def hide(self):
        return None


class _AIJudgmentAnalysisInstance:
    _sequence_analysis_key = "ai"
    _channel_mismatch = False
    _channel_mismatch_info = None

    def __init__(self, host, label):
        self.host = host
        self.label = label
        self.result = None
        self.export_detail = {}

    def calculate_ai_scores(self, *_args):
        if self.label not in ("OK", "NG"):
            return
        is_ok = self.label == "OK"
        self.host.data_struct.analysis_result_dict["AI"] = (is_ok, 0.2)
        self.result = self.label
        self.export_detail = {"label": self.label}

    def hide(self):
        return None


class _RuleJudgmentAnalysisInstance(_AnalysisInstance):
    def __init__(self, host, is_ok):
        super().__init__(host)
        self.is_ok = is_ok

    def calculate_spl(self):
        self.host.data_struct.analysis_result_dict["SPL"] = (self.is_ok, 0.0)
        return True


class _RecordingAnalysisHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {
                        "mode": "RECORD_ONLY",
                        "detail": {"sample_rate": 44100},
                    }
                }
            }
        ]
        self.analysis_config = {
            "display_sequence": ["spl"],
            "spl": {"type": "SPL"},
        }
        self.analysis_window = []
        self._analysis_result_summary_window = None
        self.data_struct = SimpleNamespace(analysis_result_dict={})
        self._prepare_live_mic_calibration_batch = Mock()
        self.count_board = SimpleNamespace(
            mode="test",
            set_test_result_file=Mock(),
            set_test_text=lambda: None,
        )
        self._active_product_condition_key = "01"
        self._manual_product_condition_results = {}
        self._current_recent_session_id = ""
        self.recorded_signal_info = {
            "source_type": "recorded",
            "labels": "not_labeled",
        }
        self.data_btn = _ButtonSpy()
        self.replayer_btn = _ButtonSpy()
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self.product_results = []

    def screen(self):
        size = SimpleNamespace(width=lambda: 1600, height=lambda: 900)
        return SimpleNamespace(size=lambda: size)

    def instance_analysis_class(self, _key, _type, _params):
        self.analysis_window.append(_AnalysisInstance(self))

    def _can_output_ok_ng(self):
        return True, ""

    def _summarize_ok_ng(self):
        return True, "OK"

    def _sync_left_panel_analysis_details(self, _state):
        return None

    def _is_directional_cycle_active(self):
        return False

    def _update_manual_product_condition_result_after_analysis(self, label):
        self.product_results.append(label)
        self._manual_product_condition_results["01"] = label
        return "OK"

    def _persist_current_test_audio_label(self, *_args, **_kwargs):
        return True

    def _finalize_test_run(self, *_args, **_kwargs):
        return None

    def update_player_btn_is_paused(self):
        return None

    def _capture_analysis_report_failure(self, *_args):
        raise AssertionError("本测试不应产生分析异常")


class _CombinedJudgmentHost(_RecordingAnalysisHost):
    _summarize_ok_ng = SequenceWidgetStreamingOpsMixin._summarize_ok_ng
    _can_output_ok_ng = SequenceWidgetStreamingOpsMixin._can_output_ok_ng

    def __init__(self, ai_label, threshold_ok):
        super().__init__()
        self.ai_label = ai_label
        self.threshold_ok = threshold_ok
        self.analysis_config = {
            "display_sequence": ["ai", "spl"],
            "ai": {"type": "AI", "analyse_model_name": "demo"},
            "spl": {"type": "SPL", "limit_checked": True},
        }

    def instance_analysis_class(self, key, _type, _params):
        if key == "ai":
            self.analysis_window.append(
                _AIJudgmentAnalysisInstance(self, self.ai_label)
            )
        else:
            self.analysis_window.append(
                _RuleJudgmentAnalysisInstance(self, self.threshold_ok)
            )

    def _update_manual_product_condition_result_after_analysis(self, label):
        self.product_results.append(label)
        return label


def test_ai_and_rule_results_use_the_same_overall_judgment():
    scenarios = [
        ("OK", False, "NG"),
        ("NG", True, "NG"),
        ("OK", True, "OK"),
        (None, True, "OK"),
    ]

    for ai_label, threshold_ok, expected in scenarios:
        host = _CombinedJudgmentHost(ai_label, threshold_ok)

        with patch(
            "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
        ):
            host.run(show_windows=False)

        assert host.product_results == [expected]


class _RecordedAudioLoadHost(SequenceWidgetAnalysisOpsMixin):
    def __init__(self):
        self.sequence_config = [
            {
                "seq1": {
                    "acq": {
                        "mode": "RECORD_ONLY",
                        "detail": {"sample_rate": 48000},
                    }
                }
            }
        ]
        self.data_struct = SimpleNamespace(
            store_wave_data=None,
            store_wave_data_multi=None,
            sample_rate=0,
            audio_lenth=0,
            wav_calibration_metadata={"old": True},
            wav_calibration_metadata_authoritative=True,
            wav_calibration_warning_shown=True,
        )
        self._active_product_condition_key = ""
        self._active_input_channels = [0, 1]
        self.clear_calls = 0
        self.plotted = None
        self.plot_mapping = None

    def clear_all_direction_waveforms(self):
        self.clear_calls += 1

    def plot_waveform_to_workspace(
        self,
        wave,
        sample_rate,
        *,
        channel_mapping=None,
    ):
        self.plotted = (np.asarray(wave), sample_rate)
        self.plot_mapping = (
            None
            if channel_mapping is None
            else tuple(channel_mapping)
        )


def test_recorded_audio_loader_preserves_multichannel_shape():
    host = _RecordedAudioLoadHost()
    channels_first = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    with patch(
        "librosa.load",
        return_value=(channels_first, 48000),
    ) as load:
        host._load_audio_file_to_data_struct("D:/audio/recorded.wav")

    load.assert_called_once_with(
        "D:/audio/recorded.wav",
        sr=48000,
        mono=False,
    )
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi,
        channels_first.T,
    )
    np.testing.assert_allclose(
        host.data_struct.store_wave_data,
        channels_first.T.mean(axis=1),
    )
    assert host._active_input_channels == [0, 1]
    assert host.clear_calls == 1
    assert host.data_struct.wav_calibration_metadata is None
    assert host.data_struct.wav_calibration_metadata_authoritative is False
    assert host.data_struct.wav_calibration_warning_shown is False


@pytest.mark.parametrize("has_metadata", [True, False])
def test_recent_audio_loader_replaces_metadata_and_preserves_channels(has_metadata):
    host = _RecordedAudioLoadHost()
    host.default_logger = logging.getLogger(__name__)
    host.data_struct.wav_calibration_metadata = {"old": True}
    host.data_struct.wav_calibration_metadata_authoritative = True
    host.data_struct.wav_calibration_warning_shown = True
    channels_first = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    metadata = {
        "recorded_channels": [
            {
                "wav_channel_index": 0,
                "v2pa_factor": 2.5,
                "standard_spl": 94.0,
                "calibrated": True,
            },
            {
                "wav_channel_index": 1,
                "v2pa_factor": None,
                "standard_spl": None,
                "calibrated": False,
            },
        ]
    }

    def decode(*_args, **_kwargs):
        assert host.data_struct.wav_calibration_metadata == {"old": True}
        assert host.data_struct.wav_calibration_metadata_authoritative is True
        assert host.data_struct.wav_calibration_warning_shown is True
        return channels_first, 48000

    with (
        patch(
            "librosa.load",
            side_effect=decode,
        ) as load,
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=_metadata_result(
                (WavCalibrationMetadataReadStatus.VALID if has_metadata
                 else WavCalibrationMetadataReadStatus.ABSENT),
                metadata["recorded_channels"] if has_metadata else None,
            ),
        ) as inspect_metadata,
    ):
        host._load_audio_file_to_data_struct(
            "D:/audio/recent.wav", presentation_owner="recent_view"
        )

    load.assert_called_once_with(
        "D:/audio/recent.wav",
        sr=48000,
        mono=False,
    )
    inspect_metadata.assert_called_once_with(
        "D:/audio/recent.wav",
        logger=host.default_logger,
    )
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data_multi,
        channels_first.T,
    )
    assert host._active_input_channels == [0, 1]
    assert host.data_struct.wav_calibration_metadata == (metadata if has_metadata else None)
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    assert host.data_struct.wav_calibration_warning_shown is False


def test_recent_session_view_restores_wav_metadata_state():
    from unit_test.test_recent_session_view import _DummySequenceWidget

    host = _DummySequenceWidget()
    original_metadata = {"recorded_channels": [{"wav_channel_index": 0}]}
    host.data_struct.wav_calibration_metadata = original_metadata
    host.data_struct.wav_calibration_metadata_authoritative = True
    host.data_struct.wav_calibration_warning_shown = True
    original_skip = object()
    host._analysis_preflight_warning_shown = True
    host._analysis_preflight_skips = {"current": original_skip}
    host._analysis_channel_local_columns = {"current": 1}
    original_loader = host._load_audio_file_to_data_struct

    def replace_audio(file_path, sample_rate=None, **kwargs):
        original_loader(file_path, sample_rate)
        host.data_struct.wav_calibration_metadata = {"history": True}
        host.data_struct.wav_calibration_metadata_authoritative = False
        host.data_struct.wav_calibration_warning_shown = False
        host._analysis_preflight_warning_shown = False
        host._analysis_preflight_skips = {"history": object()}
        host._analysis_channel_local_columns = {"history": 0}

    host._load_audio_file_to_data_struct = replace_audio

    host._show_recent_session_analysis_by_id("recent_1")

    assert host.data_struct.wav_calibration_metadata == original_metadata
    assert host.data_struct.wav_calibration_metadata_authoritative is True
    assert host.data_struct.wav_calibration_warning_shown is True
    assert host._analysis_preflight_warning_shown is True
    assert host._analysis_preflight_skips == {"current": original_skip}
    assert host._analysis_channel_local_columns == {"current": 1}


def _metadata_result(status, channels=None):
    return WavCalibrationMetadataReadResult(
        status,
        None if channels is None else {"recorded_channels": channels},
    )


def _metadata_channel(wav_index, physical=...):
    channel = {
        "wav_channel_index": wav_index,
        "v2pa_factor": None,
        "standard_spl": None,
        "calibrated": False,
    }
    if physical is not ...:
        channel["physical_input_channel"] = physical
    return channel


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_recent_audio_loader_uses_exact_saved_mapping_for_valid_old_metadata():
    host = _RecordedAudioLoadHost()
    host.default_logger = logging.getLogger(__name__)
    channels_first = np.asarray(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=np.float32,
    )
    diagnostic = _metadata_result(
        WavCalibrationMetadataReadStatus.VALID,
        [_metadata_channel(0), _metadata_channel(1)],
    )

    with (
        patch(
            "librosa.load",
            return_value=(channels_first, 48000),
        ),
        patch(
            "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
            return_value=diagnostic,
        ),
    ):
        host._load_audio_file_to_data_struct(
            "D:/audio/recent.wav",
            saved_active_input_channels=[1, 3],
            presentation_owner="recent_view",
        )

    assert host._active_input_channels == [1, 3]
    assert host.plot_mapping == (1, 3)
    assert host._waveform_presentation_owner == "recent_view"


class _WorkspaceSpy:
    def __init__(self):
        self.channel_calls = []
        self.clear_count = 0

    def set_channels(self, channels):
        self.channel_calls.append(tuple(channels))

    def clear_plots(self):
        self.clear_count += 1


class _RecentOwnershipHost(SequenceWidgetAnalysisOpsMixin):
    refresh_channel_windows = SequenceWidgetUiOpsMixin.refresh_channel_windows
    _apply_input_channel_workspace_mapping = (
        SequenceWidgetUiOpsMixin._apply_input_channel_workspace_mapping
    )

    def __init__(self, *, fail_load=False, refresh_during_load=False):
        from unit_test.test_recent_session_view import _DummySequenceWidget

        base = _DummySequenceWidget()
        self.__dict__.update(base.__dict__)
        self._session_record = {
            "recorded_signal_info": {},
            "recorded_path": "history.wav",
            "sample_rate": 48000,
            "config_snapshot": {
                "sequence_config": copy.deepcopy(self.sequence_config),
                "analysis_config": copy.deepcopy(self.analysis_config),
                "using_config_path": self.using_config_path,
                "active_input_channels": [3],
            },
        }
        self.default_logger = logging.getLogger(__name__)
        self._configured_input_channels = (0,)
        self._recording_input_channels = None
        self._pending_configured_input_channels = None
        self._channel_selection_error = ""
        self._waveform_presentation_owner = "hardware"
        self.mic_channels = [0]
        self.mic = {"max_input_channels": 5}
        self.channel_workspace = _WorkspaceSpy()
        self._fail_load = fail_load
        self._refresh_during_load = refresh_during_load
        self.loader_kwargs = None
        self.warning_messages = []

    def _resolve_recent_session(self, session_id):
        return self._session_record if session_id == "recent_1" else None

    def _resolve_recent_session_path(self, _session_record):
        return "history.wav"

    def _close_analysis_windows(self):
        return None

    def _load_audio_file_to_data_struct(self, file_path, sample_rate=None, **kwargs):
        self.loader_kwargs = {
            "saved_active_input_channels": getattr(
                self,
                "_pending_recent_saved_active_input_channels",
                None,
            ),
            "presentation_owner": getattr(
                self,
                "_pending_recent_presentation_owner",
                None,
            ),
        }
        self.data_struct.store_wave_data = np.asarray([3.0, 4.0], dtype=np.float32)
        self.data_struct.store_wave_data_multi = np.asarray(
            [[3.0], [4.0]], dtype=np.float32
        )
        self.data_struct.sample_rate = sample_rate
        self.data_struct.audio_lenth = 2
        self._active_input_channels = [3]
        if self._refresh_during_load:
            self.mic_channels = [0, 2]
            self.refresh_channel_windows()
        if self._fail_load:
            raise RuntimeError("recent load failed")

    def run(self, *args, **kwargs):
        return None

    def plot_waveform_to_workspace(self, data, sample_rate, *, channel_mapping=None):
        self.plot_calls.append(
            (
                np.asarray(data).copy(),
                sample_rate,
                None if channel_mapping is None else tuple(channel_mapping),
            )
        )

    def _clear_plot_area(self):
        self.channel_workspace.clear_plots()


class _RealRecentWorkspaceHost(
    _RecentOwnershipHost,
    SequenceWidgetStreamingOpsMixin,
):
    _decode_audio_file = SequenceWidgetAnalysisOpsMixin._decode_audio_file
    _apply_audio_to_data_struct = (
        SequenceWidgetAnalysisOpsMixin._apply_audio_to_data_struct
    )
    _load_audio_file_to_data_struct = (
        SequenceWidgetAnalysisOpsMixin._load_audio_file_to_data_struct
    )
    plot_waveform_to_workspace = (
        SequenceWidgetStreamingOpsMixin.plot_waveform_to_workspace
    )

    def init_fft_and_stft_flag(self):
        return None

    def __init__(self):
        super().__init__()
        self._session_record["config_snapshot"]["active_input_channels"] = [2, 4]
        self._active_input_channels = [7]
        self.channel_workspace = ChannelPlotWorkspace()
        self.channel_workspace.set_channels((7,))


def test_recent_view_rebuilds_real_workspace_before_committing_mapping(
    qapp,
    monkeypatch,
):
    host = _RealRecentWorkspaceHost()
    channels_first = np.asarray(
        [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]],
        dtype=np.float32,
    )
    render_events = []
    original_set_data = ChannelPlotSubWindow.set_data

    def observe_set_data(window, x_data, y_data):
        if window.channel_index in (2, 4):
            render_events.append(
                (
                    window.channel_index,
                    tuple(host._active_input_channels),
                    host._waveform_presentation_owner,
                    np.asarray(y_data).copy(),
                )
            )
        return original_set_data(window, x_data, y_data)

    monkeypatch.setattr(ChannelPlotSubWindow, "set_data", observe_set_data)
    diagnostic = _metadata_result(
        WavCalibrationMetadataReadStatus.VALID,
        [_metadata_channel(0), _metadata_channel(1)],
    )
    try:
        with (
            patch(
                "librosa.load",
                return_value=(channels_first, 48000),
            ),
            patch(
                "ui.sequence.sequence_widget_analysis_ops.inspect_wav_calibration_metadata",
                return_value=diagnostic,
            ),
            patch(
                "ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"
            ) as warning,
        ):
            host._show_recent_session_analysis_by_id("recent_1")

        warning.assert_not_called()
        assert [event[0] for event in render_events] == [2, 4]
        assert all(event[1] == (7,) for event in render_events)
        assert all(event[2] == "recent_view" for event in render_events)
        np.testing.assert_allclose(render_events[0][3], channels_first[0])
        np.testing.assert_allclose(render_events[1][3], channels_first[1])
        assert host._active_input_channels == [7]
        assert host._waveform_presentation_owner == "hardware"
        assert [
            window.channel_index
            for window in host.channel_workspace.all_subwindows()
        ] == [7]
    finally:
        host.channel_workspace.close()
        qapp.processEvents()


def test_recent_view_restores_prior_presentation_without_pending_refresh():
    host = _RecentOwnershipHost()
    previous_multi = host.data_struct.store_wave_data_multi.copy()

    with patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"):
        host._show_recent_session_analysis_by_id("recent_1")

    assert host.loader_kwargs == {
        "saved_active_input_channels": [3],
        "presentation_owner": "recent_view",
    }
    assert host._waveform_presentation_owner == "hardware"
    assert host._active_input_channels == [0]
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, previous_multi)
    np.testing.assert_array_equal(host.plot_calls[-1][0], previous_multi)


@pytest.mark.parametrize("fail_load", [False, True])
def test_recent_view_pending_hardware_refresh_wins_once_on_exit(fail_load):
    host = _RecentOwnershipHost(
        fail_load=fail_load,
        refresh_during_load=True,
    )

    with patch("ui.sequence.sequence_widget_analysis_ops.QMessageBox.warning"):
        host._show_recent_session_analysis_by_id("recent_1")

    assert host._waveform_presentation_owner == "hardware"
    assert host._pending_configured_input_channels is None
    assert host._active_input_channels == [0, 2]
    assert host.channel_workspace.channel_calls == [(0, 2)]
    assert host.channel_workspace.clear_count == 1
