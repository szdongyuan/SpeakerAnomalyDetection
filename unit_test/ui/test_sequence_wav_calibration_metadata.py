import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from base.data_struct.data_deal_struct import DataDealStruct
from consts import error_code
from ui.sequence.analysis_channel_preflight import AnalysisChannelSkip
from ui.sequence.analysis_report_snapshot import build_analysis_report_items
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from unit_test.ui.test_sequence_channel_calibration import StreamingAnalysisHost


STREAMING_OPS_PATH = (
    Path(__file__).resolve().parents[2]
    / "ui"
    / "sequence"
    / "sequence_widget_streaming_ops.py"
)


def test_data_struct_initializes_and_clears_imported_wav_metadata_state():
    previous_instance = DataDealStruct._instance
    try:
        DataDealStruct._instance = None
        data_struct = DataDealStruct()

        assert data_struct.wav_calibration_metadata is None
        assert data_struct.wav_calibration_metadata_authoritative is False
        assert data_struct.wav_calibration_warning_shown is False

        data_struct.wav_calibration_metadata = {"recorded_channels": []}
        data_struct.wav_calibration_metadata_authoritative = True
        data_struct.wav_calibration_warning_shown = True

        data_struct.clear_data()

        assert data_struct.wav_calibration_metadata is None
        assert data_struct.wav_calibration_metadata_authoritative is False
        assert data_struct.wav_calibration_warning_shown is False
    finally:
        DataDealStruct._instance = previous_instance


def test_replacing_or_clearing_audio_clears_imported_factor_and_warning_state():
    data_struct = SimpleNamespace(
        wav_calibration_metadata={"recorded_channels": []},
        wav_calibration_metadata_authoritative=True,
        wav_calibration_warning_shown=True,
    )
    host = SimpleNamespace(
        data_struct=data_struct,
        _imported_wav_channel_v2pa_factors={"spl": 2.5},
        _analysis_preflight_warning_shown=True,
        _analysis_preflight_skips={"spl": object()},
        _analysis_channel_local_columns={"spl": 0},
    )

    SequenceWidgetAnalysisOpsMixin._clear_imported_wav_calibration_state(host)

    assert data_struct.wav_calibration_metadata is None
    assert data_struct.wav_calibration_metadata_authoritative is False
    assert data_struct.wav_calibration_warning_shown is False
    assert host._imported_wav_channel_v2pa_factors == {}
    assert host._analysis_preflight_warning_shown is False
    assert host._analysis_preflight_skips == {}
    assert host._analysis_channel_local_columns == {}


def test_audio_lifecycle_reset_removes_stale_skip_from_okng_and_report_state():
    config = {
        "display_sequence": ["spl"],
        "spl": {"type": "SPL", "limit_checked": True},
    }
    stale_skip = AnalysisChannelSkip(
        item_key="spl",
        item_type="SPL",
        requested_channel=1,
        available_channels=(0,),
        reason="stale channel skip",
    )
    host = SimpleNamespace(
        data_struct=SimpleNamespace(
            wav_calibration_metadata={"recorded_channels": []},
            wav_calibration_metadata_authoritative=True,
            wav_calibration_warning_shown=True,
        ),
        analysis_config=config,
        _analysis_preflight_warning_shown=True,
        _analysis_preflight_skips={"spl": stale_skip},
        _analysis_channel_local_columns={"spl": 0},
        _imported_wav_channel_v2pa_factors={"spl": 2.5},
    )
    can_output_ok_ng = SequenceWidgetStreamingOpsMixin._can_output_ok_ng.__get__(
        host
    )

    assert can_output_ok_ng()[0] is False

    SequenceWidgetAnalysisOpsMixin._clear_imported_wav_calibration_state(host)

    assert can_output_ok_ng() == (True, "")
    assert build_analysis_report_items(
        [],
        config,
        {},
        host._analysis_preflight_skips,
    ) == []


class _Writer:
    def __init__(self, calls):
        self.calls = calls

    def finalize(self):
        self.calls.append(("finalize",))


class _PublicationTrackingDataStruct:
    def __init__(self, *, previous_mono, previous_multi):
        self._store_wave_data = previous_mono
        self._store_wave_data_multi = previous_multi
        self.mono_publications = []
        self.multi_publications = []
        self.sample_rate = 1000
        self.analysis_result_dict = {}

    @property
    def store_wave_data(self):
        return self._store_wave_data

    @store_wave_data.setter
    def store_wave_data(self, value):
        self._store_wave_data = value
        self.mono_publications.append(np.asarray(value).copy())

    @property
    def store_wave_data_multi(self):
        return self._store_wave_data_multi

    @store_wave_data_multi.setter
    def store_wave_data_multi(self, value):
        self._store_wave_data_multi = value
        self.multi_publications.append(np.asarray(value).copy())


def _complete_recording(
    *,
    trim_samples,
    quality_ok=True,
    append_result=True,
    data_struct=None,
):
    calls = []
    host = StreamingAnalysisHost()
    if data_struct is not None:
        host.data_struct = data_struct
    host.channel_workspace = _FinalWorkspace((0,))
    snapshot = {
        "recorded_channels": [
            {
                "wav_channel_index": 0,
                "physical_input_channel": 0,
                "v2pa_factor": 2.5,
                "standard_spl": 94.0,
                "calibrated": True,
            }
        ]
    }
    host._recording_wav_calibration_metadata = snapshot
    host.streaming_wav_writer = _Writer(calls)
    host._rewrite_recorded_wav = lambda *_args: calls.append(("rewrite",))
    host._should_run_silent_analysis_after_recording = lambda: False
    host._handle_invalid_recording = lambda reason: calls.append(("discard", reason))
    host._project_normalized_waveform_to_workspace = mock.Mock()
    host.run = mock.Mock(return_value=True)

    def save_signal_info_to_db(*_args):
        calls.append(("db",))
        return error_code.OK, "saved"

    recording_manager = SimpleNamespace(
        save_signal_info_to_db=save_signal_info_to_db
    )

    def append(path, metadata, logger=None):
        calls.append(("append", path, metadata, logger))
        return append_result

    with mock.patch(
        "ui.sequence.sequence_widget_streaming_ops.resolve_startup_trim_samples",
        return_value=trim_samples,
    ), mock.patch(
        "ui.sequence.sequence_widget_streaming_ops.validate_recorded_audio",
        return_value=(quality_ok, "invalid audio", {}),
    ), mock.patch(
        "ui.sequence.sequence_widget_streaming_ops.RecordingManager",
        return_value=recording_manager,
    ), mock.patch(
        "ui.sequence.sequence_widget_streaming_ops.append_wav_calibration_metadata",
        side_effect=append,
    ):
        host._on_streaming_complete(
            recorded_mono=np.asarray([0.1, -0.1], dtype=np.float32),
            recorded_multi=np.asarray([[0.1], [-0.1]], dtype=np.float32),
            sample_rate=1000,
            completion_source="test",
        )

    return host, snapshot, calls


def test_streaming_finalize_and_startup_trim_precede_single_metadata_append():
    host, snapshot, calls = _complete_recording(trim_samples=1)

    assert calls[:3] == [
        ("finalize",),
        ("rewrite",),
        ("append", "recorded.wav", snapshot, host.default_logger),
    ]


def test_streaming_without_trim_appends_once_after_finalize():
    host, snapshot, calls = _complete_recording(trim_samples=0)

    assert calls[:2] == [
        ("finalize",),
        ("append", "recorded.wav", snapshot, host.default_logger),
    ]
    assert [call[0] for call in calls].count("append") == 1
    assert not any(call[0] == "rewrite" for call in calls)


def test_failed_quality_gate_discards_without_publishing_metadata():
    _host, _snapshot, calls = _complete_recording(
        trim_samples=0,
        quality_ok=False,
    )

    assert calls[0] == ("finalize",)
    assert any(call[0] == "discard" for call in calls)
    assert not any(call[0] == "append" for call in calls)


def test_failed_quality_gate_does_not_publish_or_replace_authoritative_audio():
    previous_mono = np.asarray([7.0, 8.0], dtype=np.float32)
    previous_multi = np.asarray([[7.0], [8.0]], dtype=np.float32)
    data_struct = _PublicationTrackingDataStruct(
        previous_mono=previous_mono,
        previous_multi=previous_multi,
    )

    host, _snapshot, calls = _complete_recording(
        trim_samples=0,
        quality_ok=False,
        data_struct=data_struct,
    )

    assert data_struct.mono_publications == []
    assert data_struct.multi_publications == []
    assert data_struct.store_wave_data is previous_mono
    assert data_struct.store_wave_data_multi is previous_multi
    assert any(call[0] == "discard" for call in calls)
    assert not any(call[0] == "db" for call in calls)
    host._project_normalized_waveform_to_workspace.assert_not_called()
    host.run.assert_not_called()


def test_success_publishes_trimmed_multi_and_mean_exactly_once():
    data_struct = _PublicationTrackingDataStruct(
        previous_mono=np.asarray([7.0], dtype=np.float32),
        previous_multi=np.asarray([[7.0]], dtype=np.float32),
    )

    _host, _snapshot, calls = _complete_recording(
        trim_samples=1,
        quality_ok=True,
        data_struct=data_struct,
    )

    assert len(data_struct.multi_publications) == 1
    assert len(data_struct.mono_publications) == 1
    np.testing.assert_array_equal(
        data_struct.multi_publications[0],
        np.asarray([[-0.1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        data_struct.mono_publications[0],
        np.asarray([-0.1], dtype=np.float32),
    )
    assert [call[0] for call in calls].count("db") == 1


def test_append_failure_logs_warning_but_keeps_successful_recording():
    host, _snapshot, calls = _complete_recording(
        trim_samples=0,
        append_result=False,
    )

    assert [call[0] for call in calls].count("append") == 1
    assert not any(call[0] == "discard" for call in calls)
    assert host._record_workflow_busy is False
    assert any(
        "recording_calibration_metadata_append_failed" in call.args[0]
        for call in host.default_logger.warning.call_args_list
    )


def test_empty_snapshot_is_not_appended():
    host = SimpleNamespace(
        recorded_path="recorded.wav",
        _recording_wav_calibration_metadata=None,
        default_logger=mock.Mock(),
    )

    with mock.patch(
        "ui.sequence.sequence_widget_streaming_ops.append_wav_calibration_metadata"
    ) as append:
        result = SequenceWidgetStreamingOpsMixin._append_recording_wav_calibration_metadata(
            host
        )

    assert result is False
    append.assert_not_called()


def test_append_is_only_reached_from_final_completion_not_cleanup_or_rename():
    tree = ast.parse(STREAMING_OPS_PATH.read_text(encoding="utf-8"))
    mixin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SequenceWidgetStreamingOpsMixin"
    )
    callers = []
    for method in (node for node in mixin.body if isinstance(node, ast.FunctionDef)):
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_append_recording_wav_calibration_metadata"
            for node in ast.walk(method)
        ):
            callers.append(method.name)

    assert callers == ["_on_streaming_complete"]


class _FinalProcessor:
    target_samples = 3

    def __init__(self, recorded_multi, retained_channels):
        self.recorded_multi = np.asarray(recorded_multi, dtype=np.float32)
        self._rec_in_sel = list(retained_channels)
        self.process_queue_calls = 0

    def process_queue(self):
        self.process_queue_calls += 1

    def get_recorded_data(self):
        return self.recorded_multi.mean(axis=1).astype(np.float32, copy=False)

    def get_recorded_data_multi(self):
        return self.recorded_multi


class _FinalWorkspace:
    def __init__(self, channels):
        self._windows = [
            SimpleNamespace(channel_index=channel) for channel in channels
        ]

    def all_subwindows(self):
        return list(self._windows)

    def clear_plots(self):
        return None


def _prepare_final_completion_host(
    recorded_multi,
    retained_channels=(0, 2),
    workspace_channels=(0, 2),
):
    calls = []
    host = StreamingAnalysisHost()
    host._recording_input_channels = (0, 2)
    host._active_input_channels = [0, 2]
    host._pending_configured_input_channels = (0, 2)
    host.channel_workspace = _FinalWorkspace(workspace_channels)
    host._streaming_completion_processor = None
    host.streaming_processor = _FinalProcessor(recorded_multi, retained_channels)
    host.streaming_wav_writer = _Writer(calls)
    host._recording_wav_calibration_metadata = {
        "recorded_channels": [
            {"wav_channel_index": 0, "physical_input_channel": 0},
            {"wav_channel_index": 1, "physical_input_channel": 2},
        ]
    }
    host._rewrite_recorded_wav = mock.Mock()
    host._append_recording_wav_calibration_metadata = mock.Mock(return_value=True)
    host._handle_invalid_recording = mock.Mock()
    host._should_run_silent_analysis_after_recording = lambda: True
    host.run = mock.Mock(return_value=True)
    host._is_manual_product_condition_cycle_active = lambda: False
    host._finalize_serial_product_condition_after_analysis = lambda: True
    host._advance_manual_product_condition_cycle_after_recording = mock.Mock()
    host._on_serial_product_condition_completed = mock.Mock()
    host._on_directional_recording_completed = mock.Mock()
    host._project_normalized_waveform_to_workspace = mock.Mock()

    def abort_selection():
        calls.append(("abort_run",))
        host._recording_input_channels = None
        host._pending_configured_input_channels = None

    host._abort_recording_channel_selection = abort_selection

    def finalize_selection():
        calls.append(("release_run",))
        host._recording_input_channels = None
        host._pending_configured_input_channels = None

    host._finalize_recording_channel_selection = finalize_selection
    return host, calls


def test_final_projection_failure_is_presentation_only_and_releases_run_state(
    monkeypatch,
):
    recorded_multi = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=np.float32,
    )
    host, calls = _prepare_final_completion_host(recorded_multi)
    host._project_normalized_waveform_to_workspace = mock.Mock(
        side_effect=RuntimeError("Qt plot failed")
    )
    database = mock.Mock()
    database.save_signal_info_to_db.return_value = (error_code.OK, "saved")
    warning = mock.Mock()
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_streaming_ops.resolve_startup_trim_samples",
        lambda *_args: 0,
    )
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_streaming_ops.validate_recorded_audio",
        lambda *_args: (True, "", {}),
    )
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_streaming_ops.RecordingManager",
        lambda: database,
    )
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_streaming_ops.QMessageBox.warning",
        warning,
    )

    processor = host.streaming_processor
    host._on_streaming_recording_finished(processor)

    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, recorded_multi)
    np.testing.assert_array_equal(
        host.data_struct.store_wave_data,
        recorded_multi.mean(axis=1),
    )
    database.save_signal_info_to_db.assert_called_once_with(
        host.recorded_signal_info,
        None,
    )
    host.run.assert_called_once_with(show_windows=False)
    host._handle_invalid_recording.assert_not_called()
    host._project_normalized_waveform_to_workspace.assert_called_once()
    projection_args = host._project_normalized_waveform_to_workspace.call_args.args
    np.testing.assert_array_equal(projection_args[0], recorded_multi)
    assert projection_args[1] == 48000
    assert [window.channel_index for window in projection_args[2]] == [0, 2]
    warning.assert_called_once()
    assert "波形刷新失败" in warning.call_args.args[2]
    assert host._condition_record_cache["condition-1"] == {
        "recorded_path": "recorded.wav",
        "recorded_signal_info": {
            "labels": "not_labeled",
            "sample_rate": 48000,
        },
        "session_id": "session-1",
    }
    assert host.streaming_processor is None
    assert host._recording_input_channels is None
    assert calls[-1] == ("release_run",)


@pytest.mark.parametrize(
    ("recorded_multi", "retained_channels"),
    [
        (np.ones((3, 1), dtype=np.float32), (0, 2)),
        (np.ones((3, 3), dtype=np.float32), (0, 2)),
        (np.ones((3, 2), dtype=np.float32), (0, 1)),
    ],
)
def test_final_contract_mismatch_rejects_before_db_and_analysis(
    monkeypatch,
    recorded_multi,
    retained_channels,
):
    host, _calls = _prepare_final_completion_host(
        recorded_multi,
        retained_channels=retained_channels,
    )
    terminate = mock.Mock()
    host._terminate_invalid_streaming_recording = terminate
    database = mock.Mock()
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_streaming_ops.RecordingManager",
        lambda: database,
    )

    host._on_streaming_complete(
        recorded_mono=recorded_multi.mean(axis=1),
        recorded_multi=recorded_multi,
        sample_rate=48000,
        completion_source="test",
    )

    terminate.assert_called_once()
    database.save_signal_info_to_db.assert_not_called()
    host.run.assert_not_called()


@pytest.mark.parametrize(
    ("workspace_channels", "reason_fragment"),
    [
        ((0,), "count"),
        ((2, 0), "order"),
    ],
)
def test_final_workspace_contract_mismatch_is_invalid_before_publication(
    monkeypatch,
    workspace_channels,
    reason_fragment,
):
    recorded_multi = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=np.float32,
    )
    previous_mono = np.asarray([7.0], dtype=np.float32)
    previous_multi = np.asarray([[7.0]], dtype=np.float32)
    data_struct = _PublicationTrackingDataStruct(
        previous_mono=previous_mono,
        previous_multi=previous_multi,
    )
    host, calls = _prepare_final_completion_host(
        recorded_multi,
        workspace_channels=workspace_channels,
    )
    host.data_struct = data_struct
    host._project_normalized_waveform_to_workspace = mock.Mock()
    database = mock.Mock()
    database.save_signal_info_to_db.return_value = (error_code.OK, "saved")
    monkeypatch.setattr(
        "ui.sequence.sequence_widget_streaming_ops.RecordingManager",
        lambda: database,
    )

    host._on_streaming_complete(
        recorded_mono=recorded_multi.mean(axis=1),
        recorded_multi=recorded_multi,
        sample_rate=48000,
        completion_source="test",
    )

    assert data_struct.mono_publications == []
    assert data_struct.multi_publications == []
    assert data_struct.store_wave_data is previous_mono
    assert data_struct.store_wave_data_multi is previous_multi
    host._append_recording_wav_calibration_metadata.assert_not_called()
    host._project_normalized_waveform_to_workspace.assert_not_called()
    database.save_signal_info_to_db.assert_not_called()
    host.run.assert_not_called()
    host._handle_invalid_recording.assert_called_once()
    assert reason_fragment in host._handle_invalid_recording.call_args.args[0]
    assert host.streaming_processor is None
    assert host.streaming_wav_writer is None
    assert host._recording_input_channels is None
    assert ("abort_run",) in calls
