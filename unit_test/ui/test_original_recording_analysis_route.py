"""Accepted capture uses the original automatic subprocess queue for every type."""
from types import SimpleNamespace
from unittest import mock
import threading
import numpy as np
import pytest
from unit_test.ui.test_recording_process_integration import main_host
from ui.sequence.recording_process_context import RecordingProcessContext

@pytest.mark.parametrize("types", [("SPL",), ("FBA",), ("SPEC",), ("AI", "FFT")])
def test_accepted_released_capture_enters_original_queue_once(ui_qapp, tmp_path, monkeypatch, types):
    from ui.sequence import sequence_widget_streaming_ops as streaming
    from consts import error_code
    monkeypatch.setattr(streaming, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=lambda *a: (error_code.OK, "saved")))
    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    host.analysis_config = {"auto_analysis": True, "display_sequence": list(types)}
    host._enqueue_automatic_analysis_current_recording = mock.Mock(return_value=True)
    host.run = mock.Mock(side_effect=AssertionError("recording must use subprocess queue"))
    host._submit_request_scoped_recording_publication = mock.Mock(return_value=True)
    host._publish_serialized_recording_context = mock.Mock()
    host._schedule_raw_audio_csv_export = mock.Mock()
    host._finalize_recording_channel_selection = mock.Mock()
    request = SimpleNamespace(request_id="accepted", channels=(0, 2), path=str(tmp_path / "accepted.wav"), sample_rate=100, device={}, calibration_metadata=None)
    multi = np.ones((9, 2), dtype=np.float32)
    audio = SimpleNamespace(multi=multi, mono=multi.mean(axis=1), descriptor=SimpleNamespace(sample_rate=100, warnings=()))
    session = SimpleNamespace(request=request, state="completed", released=threading.Event(), release_error=None)
    session.released.set()
    context = RecordingProcessContext(request=request, session=session, direction="condition", tcp_completion=None, preview_enabled=False, accepted_audio=audio, validated_audio=audio, final_windows=host.channel_workspace.all_subwindows())
    # These fields exist only in the rejected architecture; the replacement ignores them.
    context.enabled_analysis_identifiers = types
    host._recording_process_contexts = {request.request_id: context}
    host._active_recording_process_id = request.request_id
    host._recording_process_direction = "condition"
    host._recording_process_request = request
    host._recording_process_id = request.request_id
    host._recording_process_session = session
    host._record_workflow_busy = True
    host._publish_process_recording(session)
    host._publish_process_recording(session)
    host._enqueue_automatic_analysis_current_recording.assert_called_once_with()
    host.run.assert_not_called()
    host._submit_request_scoped_recording_publication.assert_not_called()
    host._publish_serialized_recording_context.assert_not_called()
    np.testing.assert_array_equal(host.data_struct.store_wave_data_multi, multi)


def queue_host(ui_qapp, tmp_path, monkeypatch, types=("SPL",), auto=True):
    from collections import deque
    from datetime import datetime
    import soundfile as sf
    from base.analysis_artifact_paths import AnalysisStorageContext
    from base.wav_calibration_metadata import append_wav_calibration_metadata
    from ui.sequence import sequence_widget_streaming_ops as streaming
    from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin as Queue
    from consts import error_code
    monkeypatch.setattr(streaming, "RecordingManager", lambda: SimpleNamespace(save_signal_info_to_db=lambda *a: (error_code.OK, "saved")))
    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    request = SimpleNamespace(request_id="queue-capture", channels=(0, 2), path=str(tmp_path / "queue.wav"), sample_rate=48000, device={}, calibration_metadata=None)
    t = np.arange(12000, dtype=np.float32) / request.sample_rate
    multi = np.column_stack((.01*np.sin(2*np.pi*1000*t), .02*np.sin(2*np.pi*1000*t))).astype(np.float32)
    sf.write(request.path, multi, request.sample_rate, subtype="FLOAT")
    assert append_wav_calibration_metadata(request.path, {"recorded_channels": [dict(wav_channel_index=i, physical_input_channel=channel, v2pa_factor=2.0+i, standard_spl=94., calibrated=True) for i,channel in enumerate(request.channels)]})
    host.analysis_config = {"auto_analysis": auto, "display_sequence": list(types), **{kind: {"type": kind, "analysis_channels": [0, 2], "limit_checked": False, "weighting": "Z", "show_overall_spl": True} for kind in types}}
    host._should_run_silent_analysis_after_recording = streaming.SequenceWidgetStreamingOpsMixin._should_run_silent_analysis_after_recording.__get__(host)
    host.recorded_signal_info.update(active_input_channels=[0,2], analysis_storage=AnalysisStorageContext(str(tmp_path.resolve()), "project", "model", "sample", 1, "port", "condition", datetime(2026,9,1)).to_metadata())
    host._current_recent_session_id = "record-session"
    host.channel_workspace.channel_layout = {"CH1": "front", "CH3": "rear"}
    host._get_active_product_condition_key = lambda: "condition"
    host._cache_condition_record = streaming.SequenceWidgetStreamingOpsMixin._cache_condition_record.__get__(host)
    host._resolve_condition_record = lambda key: host._condition_record_cache.get(key)
    host._schedule_raw_audio_csv_export = mock.Mock()
    host._set_condition_analysis_stage = mock.Mock()
    host._record_analysis_admission_state = mock.Mock()
    host._refresh_analysis_action_state = mock.Mock()
    host._analysis_task_records = {}
    host._analysis_task_queue = deque()
    host._analysis_active_request = None
    host._analysis_process_service = None
    host.run = mock.Mock(side_effect=AssertionError("must not run direct analysis"))
    for name in ['_enqueue_automatic_analysis_current_recording','_build_process_analysis_request','_analysis_record_wav_path','_start_next_queued_analysis']:
        setattr(host, name, getattr(Queue, name).__get__(host))
    audio = SimpleNamespace(multi=multi, mono=multi.mean(axis=1), descriptor=SimpleNamespace(sample_rate=48000, warnings=()))
    session = SimpleNamespace(request=request, state="completed", released=threading.Event(), release_error=None)
    session.released.set()
    context = RecordingProcessContext(request=request, session=session, direction="condition", tcp_completion=None, preview_enabled=False, accepted_audio=audio, validated_audio=audio, recorded_signal_info=dict(host.recorded_signal_info), final_windows=host.channel_workspace.all_subwindows())
    host._recording_process_contexts = {request.request_id: context}
    host._active_recording_process_id = request.request_id
    host._recording_process_id = request.request_id
    host._recording_process_session = session
    host._record_workflow_busy = True
    return host, session, context


@pytest.mark.parametrize("types", [("SPL",), ("FBA",), ("Spec",), ("AI", "FFT")])
@pytest.mark.parametrize("auto", [True, False])
def test_completion_uses_original_task_builder_and_disabled_analysis_setting(ui_qapp, tmp_path, monkeypatch, types, auto):
    host, session, context = queue_host(ui_qapp, tmp_path, monkeypatch, types, auto)
    host._publish_process_recording(session)
    host._on_process_recording_released(session)
    assert len(host._analysis_task_queue) == int(auto)
    host.run.assert_not_called()
    assert host._condition_record_cache["condition"]["recorded_path"] == session.request.path
    if auto:
        task = host._analysis_task_queue[0]
        assert task.wav_path == session.request.path
        assert task.condition_key == "condition"
        assert {item.raw_channel for item in task.instances} == {0,2}
        assert {item.analysis_type for item in task.instances} == set(types)
        assert len(host._analysis_task_records) == 1
    else:
        assert host._record_analysis_admission_state.call_args.kwargs['state'] == 'not_required'
    assert host._can_start_recording_workflow()


@pytest.mark.parametrize("terminal", ["cancelled", "failed", "cleanup_owned", "unreleased", "stale", "release_error", "wrong_audio"])
def test_terminal_or_unreleased_result_cannot_enqueue(ui_qapp, tmp_path, monkeypatch, terminal):
    host, session, context = queue_host(ui_qapp, tmp_path, monkeypatch)
    if terminal in {"cancelled", "failed", "cleanup_owned"}:
        setattr(context, terminal, True)
    elif terminal == "unreleased":
        session.released.clear()
    elif terminal == "stale":
        host._active_recording_process_id = "replacement"
    elif terminal == "release_error":
        session.release_error = "lease held"
    else:
        context.validated_audio = object()
    host._publish_process_recording(session)
    assert not host._analysis_task_queue
    assert host.data_struct.store_wave_data_multi is None


def test_gui_observer_cannot_start_capture_before_original_enqueue(ui_qapp, tmp_path, monkeypatch):
    host, session, context = queue_host(ui_qapp, tmp_path, monkeypatch)
    observations = []
    def observe(*args):
        observations.append(host._can_start_recording_workflow())
        host._record_workflow_busy = False
        host.player_status_flag = False
        assert not host._can_start_recording_workflow()
        with pytest.raises(RuntimeError):
            host._start_process_recording(host._recorded_dict, 100)
    host._update_current_recent_session_result = observe
    host._publish_process_recording(session)
    assert observations == [False]
    assert len(host._analysis_task_queue) == 1
    assert host._can_start_recording_workflow()


def test_gui_cancel_observer_prevents_analysis_enqueue(ui_qapp, tmp_path, monkeypatch):
    host, session, context = queue_host(ui_qapp, tmp_path, monkeypatch)
    session.cancel = mock.Mock()
    host._update_current_recent_session_result = lambda *a: host._cancel_process_recording()
    host._publish_process_recording(session)
    session.cancel.assert_called_once_with()
    assert not host._analysis_task_queue


def test_recording_completion_starts_original_spawned_spl_worker(ui_qapp, tmp_path, monkeypatch):
    import time
    from pathlib import Path
    from base.analysis_service import AnalysisProcessService
    host, session, context = queue_host(ui_qapp, tmp_path, monkeypatch)
    service = AnalysisProcessService()
    host._analysis_process_service = service
    terminal = None
    try:
        host._publish_process_recording(session)
        task = host._analysis_active_request
        assert task is not None and task.wav_path == session.request.path
        assert service.active
        assert host._can_start_recording_workflow(), "original analysis may overlap the next recording"
        deadline = time.monotonic()+30
        while service.active and time.monotonic()<deadline:
            events, logs = service.poll()
            for kind, payload in events:
                if kind in {"result", "failure"}: terminal = kind, payload
            ui_qapp.processEvents()
            threading.Event().wait(.01)
        assert terminal is not None
        assert terminal[0] == "result", terminal
        result = terminal[1]
        assert result.execution_status == "分析完成"
        assert len(result.instance_results) == 2
        artifacts = [a for item in result.instance_results for a in item.artifacts]
        assert artifacts and all(a.status == "已保存" and Path(a.path).is_file() for a in artifacts)
        assert any("CH3(rear)" in Path(a.path).name for a in artifacts)
    finally:
        if service.active:
            service._process.terminate()
            service._process.join(timeout=5)
            service.poll()
