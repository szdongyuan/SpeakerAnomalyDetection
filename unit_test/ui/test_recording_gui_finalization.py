"""Bounded publication and actual Qt processing-state regressions."""
from dataclasses import replace
import threading
import sys
import logging
from unittest import mock
import numpy as np
import pytest
from PyQt5.QtCore import QThread, QTimer
from unit_test.ui.test_recording_process_integration import main_host, pump, service
from unit_test.base.test_recording_finalization import capture_audio, read_result
from unit_test.base.test_recording_capture import request

def legacy_display(
    waveform,
    sample_rate,
    *,
    max_points=None,
):
    """Build a peak-preserving display copy without changing the full waveform."""
    if max_points is None:
        point_limit = 4000
    elif isinstance(max_points, (bool, np.bool_)) or not isinstance(
        max_points,
        (int, np.integer),
    ) or max_points < 4:
        raise ValueError("max_points must be an integer >= 4")
    else:
        point_limit = int(max_points)

    sample_count = waveform.shape[0]
    if sample_count <= point_limit:
        sample_indices = np.arange(sample_count, dtype=np.int64)
    else:
        peak_bucket_count = (point_limit - 2) // 2
        bucket_size = (sample_count + peak_bucket_count - 1) // peak_bucket_count
        full_block_count = sample_count // bucket_size
        full_sample_count = full_block_count * bucket_size
        blocks = waveform[:full_sample_count].reshape(full_block_count, bucket_size)
        block_starts = np.arange(full_block_count, dtype=np.int64) * bucket_size
        min_indices = block_starts + np.argmin(blocks, axis=1)
        max_indices = block_starts + np.argmax(blocks, axis=1)
        ordered_indices = np.empty(full_block_count * 2, dtype=np.int64)
        ordered_indices[0::2] = np.minimum(min_indices, max_indices)
        ordered_indices[1::2] = np.maximum(min_indices, max_indices)
        peak_indices = [ordered_indices]

        if full_sample_count < sample_count:
            tail = waveform[full_sample_count:]
            tail_min = full_sample_count + int(np.argmin(tail))
            tail_max = full_sample_count + int(np.argmax(tail))
            peak_indices.append(
                np.array([min(tail_min, tail_max), max(tail_min, tail_max)], dtype=np.int64)
            )

        sample_indices = np.unique(
            np.concatenate(
                [
                    np.array([0], dtype=np.int64),
                    *peak_indices,
                    np.array([sample_count - 1], dtype=np.int64),
                ]
            )
        )

    time_axis = sample_indices.astype(np.float64) / float(sample_rate or 1.0)
    return time_axis, waveform[sample_indices]

@pytest.mark.parametrize("length,limit", [(0,4),(3,4),(8,4),(27,10),(101,12),(48011,48000),(210017,48000)])
def test_background_display_matches_original_exactly(length, limit):
    from base.recording_waveform_preparation import prepare_waveform_display_data
    rng = np.random.default_rng(12)
    wave = rng.normal(size=length).astype(np.float32)
    if length > 10:
        wave[3], wave[7], wave[-2] = 90, -90, 100
    expected = legacy_display(wave, 51200, max_points=limit)
    actual = prepare_waveform_display_data(wave, 51200, max_points=limit)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a,b)
    assert len(actual[0]) <= limit


def test_reader_products_are_frozen_and_bound_to_exact_request(tmp_path):
    from base.recording_result_reader import RecordingAudio
    req = request(tmp_path, channels=(2,0))
    _, descriptor = capture_audio(req)
    audio = read_result(descriptor, req).audio
    assert audio.is_prepared_for(req)
    assert not audio.is_prepared_for(replace(req))
    assert not replace(audio).is_prepared_for(req)
    assert not replace(audio, descriptor=replace(descriptor)).is_prepared_for(req)
    assert not RecordingAudio(descriptor, audio.multi, audio.mono).is_prepared_for(req)
    assert len(audio.waveforms) == 2
    for array in (audio.multi, audio.mono, *(a for pair in audio.waveforms for a in pair)):
        with pytest.raises(ValueError):
            array.flat[0] = 1
        with pytest.raises(ValueError):
            array.setflags(write=True)
    for column, waveform in enumerate(audio.waveforms):
        expected = legacy_display(audio.multi[:,column], req.sample_rate, max_points=48000)
        for a,b in zip(waveform, expected):
            np.testing.assert_array_equal(a,b)


def test_calibration_skips_display_preparation(tmp_path):
    req = request(tmp_path, purpose="calibration", channels=(0,), trim_samples=0)
    _, descriptor = capture_audio(req)
    audio = read_result(descriptor, req).audio
    assert audio.waveforms == ()
    assert not audio.is_prepared_for(req)


@pytest.mark.parametrize("stage", ["数据保存中", "正转 数据保存中"])
def test_actual_panel_recognizes_processing(ui_qapp, stage):
    from ui.sequence.motor_ai_result_panel import MotorAiResultPanel
    panel = MotorAiResultPanel()
    panel.set_current_stage(stage)
    assert panel.stage_label.text() == "数据保存中"
    panel.close()


@pytest.mark.parametrize("drop_finalizing", [False, True])
@pytest.mark.parametrize("actual_workspace", [False, True])
def test_background_preparation_keeps_qt_responsive_and_publishes_without_scans(ui_qapp, service, tmp_path, monkeypatch, caplog, drop_finalizing, actual_workspace):
    caplog.set_level(logging.INFO)
    import base.recording_result_reader as reader_module
    from ui.sequence.motor_ai_result_panel import MotorAiResultPanel
    from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    entered, release, ticked = threading.Event(), threading.Event(), threading.Event()
    handle_event = service._event
    if drop_finalizing:
        def terminal_only(worker, event):
            if event.kind != "finalizing":
                handle_event(worker, event)
        monkeypatch.setattr(service, "_event", terminal_only)
    original = reader_module.ResultReader._validate_metadata
    def block_reader(reader):
        entered.set()
        assert release.wait(10)
        return original(reader)
    monkeypatch.setattr(reader_module.ResultReader, "_validate_metadata", block_reader)
    host = main_host(service, tmp_path, streaming=False)
    panel = MotorAiResultPanel()
    panel.set_condition_configs([{"key":"forward", "name":"正转"}])
    panel.resize(640, 500)
    panel.show()
    host.left_panel = panel
    workspace = None
    if actual_workspace:
        workspace = AnalysisWaveformPanel(condition_configs=[{"key":"forward", "name":"正转"}])
        workspace.set_channels((0, 2))
        host.channel_workspace = workspace
    host._get_active_product_condition_key = lambda: "forward"
    host._active_product_condition_config = {"name":"正转"}
    monkeypatch.setattr("ui.sequence.sequence_widget_streaming_ops.RecordingManager", lambda: mock.Mock(save_signal_info_to_db=mock.Mock(return_value=(0,"ok"))))
    host.judge_play_and_record()
    session = host._recording_process_session
    try:
        pump(ui_qapp, entered.is_set)
        QTimer.singleShot(0, ticked.set)
        pump(ui_qapp, ticked.is_set)
        assert panel.stage_label.text() == "数据保存中"
        if workspace is not None:
            assert workspace.status_label.text() == "数据保存中"
        assert panel.rows["forward"]["labels"]["result"].text() == "数据保存中"
        label = panel.rows["forward"]["labels"]["result"]
        assert label.isVisible()
        assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
        assert session.state == "delivering" and service.busy
        assert host.data_struct.store_wave_data is None
        def forbidden(*args, **kwargs):
            raise AssertionError("full waveform preparation on GUI")
        host._prepare_waveform_display_data = forbidden
        original_equal, original_finite = np.array_equal, np.isfinite
        def guarded(original):
            def invoke(*args, **kwargs):
                if QThread.currentThread() is ui_qapp.thread():
                    raise AssertionError("array scan on GUI")
                return original(*args, **kwargs)
            return invoke
        if not actual_workspace:
            # Real plot widgets may scan their bounded display arrays.
            monkeypatch.setattr(np, "array_equal", guarded(original_equal))
            monkeypatch.setattr(np, "isfinite", guarded(original_finite))
        def forbid_gui_mean(frame, event, function):
            if event == "c_call" and getattr(function, "__name__", "") == "mean":
                raise AssertionError("mean on GUI")
        if not actual_workspace:
            sys.setprofile(forbid_gui_mean)
        release.set()
        pump(ui_qapp, lambda: host.data_struct.store_wave_data is not None)
        assert host.data_struct.store_wave_data is session.audio.mono
        assert host.data_struct.store_wave_data_multi is session.audio.multi
        assert session.state == "completed"
        messages = [record.getMessage() for record in caplog.records]
        assert any("stage=gui_publication" in message and session.request.request_id in message for message in messages)
        assert any("stage=completion" in message and "child_descriptor_to_parent_observation_gap=unmeasured" in message for message in messages)
        assert panel.stage_label.text() != "数据保存中"
        if workspace is not None:
            assert workspace.status_label.text() == "待判定"
            assert workspace._condition_contexts["forward"]["status"] == "待判定"
            host._handle_invalid_recording.assert_not_called()
        assert panel.rows["forward"]["labels"]["result"].text() != "数据保存中"
    finally:
        sys.setprofile(None)
        release.set()
        panel.close()
        if workspace is not None:
            workspace.close()


def test_child_and_reader_report_request_correlated_local_durations(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    req = request(tmp_path)
    _, descriptor = capture_audio(req)
    timing = descriptor.finalization_timing
    assert 0 <= timing.target_to_file_closed <= timing.target_to_descriptor
    assert 0 <= timing.metadata_seconds <= timing.target_to_descriptor
    assert read_result(descriptor, req).error is None
    messages = [r.getMessage() for r in caplog.records if req.request_id in r.getMessage()]
    for stage in ("target_samples", "drain_file_close", "metadata", "read_validation", "mono_display", "reader_complete"):
        assert any("stage=" + stage in m for m in messages), stage


def test_finalizing_is_once_only_and_cancel_blocks_fallback():
    from base.recording_service import RecordingService, RecordingSession, RecordingCallbacks
    from types import SimpleNamespace
    service = object.__new__(RecordingService)
    service._logger = mock.Mock()
    seen = []
    session = RecordingSession(service, SimpleNamespace(request_id="once"), RecordingCallbacks(finalizing=lambda s: seen.append(s)))
    service._notify_finalizing(session)
    service._notify_finalizing(session)
    assert seen == [session]
    cancelled = RecordingSession(service, SimpleNamespace(request_id="once"), RecordingCallbacks(finalizing=lambda s: seen.append(s)))
    cancelled.cancel_requested = True
    service._notify_finalizing(cancelled)
    assert seen == [session]


@pytest.mark.parametrize("legacy_prefinalized", [False, True])
def test_changed_legacy_input_recomputes_mono(ui_qapp, tmp_path, monkeypatch, legacy_prefinalized):
    from unit_test.ui.test_sequence_wav_calibration_metadata import _prepare_final_completion_host
    req = request(tmp_path, trim_samples=0)
    _, descriptor = capture_audio(req)
    audio = read_result(descriptor, req).audio
    changed = audio.multi.copy()
    changed[:, 0] += 4
    host, _ = _prepare_final_completion_host(changed)
    host._recording_process_request = req
    host._recording_process_direction = "condition-1"
    monkeypatch.setattr("ui.sequence.sequence_widget_streaming_ops.resolve_startup_trim_samples", lambda *args: 0)
    monkeypatch.setattr("ui.sequence.sequence_widget_streaming_ops.validate_recorded_audio", lambda *args: (True,"",{}))
    monkeypatch.setattr("ui.sequence.sequence_widget_streaming_ops.RecordingManager", lambda: mock.Mock(save_signal_info_to_db=mock.Mock(return_value=(0,"ok"))))
    assert host._on_streaming_complete(recorded_mono=audio.mono, recorded_multi=changed,
        sample_rate=req.sample_rate, prefinalized=legacy_prefinalized,
        completion_source="process" if legacy_prefinalized else "streaming",
        prepared_audio=audio) is True
    np.testing.assert_array_equal(host.data_struct.store_wave_data, changed.mean(axis=1))
    assert host.data_struct.store_wave_data is not audio.mono


@pytest.mark.parametrize("terminal", ["cancel", "cancelled", "failed"])
def test_cancelled_and_replaced_session_cannot_restore_processing(ui_qapp, tmp_path, terminal):
    from types import SimpleNamespace
    from base.recording_service import RecordingSession, RecordingCallbacks
    from ui.sequence.recording_process_context import RecordingProcessContext
    from ui.sequence.motor_ai_result_panel import MotorAiResultPanel
    from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel
    service = SimpleNamespace(cancel=mock.Mock())
    host = main_host(service, tmp_path)
    panel = MotorAiResultPanel(condition_configs=[{"key":"forward", "name":"正转"}])
    host.left_panel = panel
    workspace = AnalysisWaveformPanel(condition_configs=[
        {"key":"forward", "name":"正转"}, {"key":"reverse", "name":"反转"}])
    host.channel_workspace = workspace
    host._get_active_product_condition_key = lambda: "forward"
    req = request(tmp_path)
    old = RecordingSession(service, req, RecordingCallbacks())
    context = RecordingProcessContext(req, "forward", False, session=old)
    host._recording_process_contexts = {req.request_id: context}
    host._recording_process_session = old
    host._active_recording_process_id = req.request_id
    host._on_process_recording_finalizing(old)
    assert panel.stage_label.text() == "数据保存中"
    assert workspace.status_label.text() == "数据保存中"
    if terminal == "cancel":
        host._cancel_process_recording()
        expected = "等待开始"
    elif terminal == "cancelled":
        host._on_process_recording_cancelled(old, None)
        expected = "等待开始"
    else:
        from base.recording_process_protocol import RecordingFailure
        host._on_process_recording_failed(old, RecordingFailure(req.request_id, "read", req.path, "broken"))
        expected = "测试异常"
    assert panel.stage_label.text() == expected
    expected_condition = "测试异常" if terminal == "failed" else "待检测"
    assert workspace.status_label.text() == expected_condition
    assert workspace._condition_contexts["forward"]["status"] == expected_condition
    host._on_process_recording_finalizing(old)
    assert panel.stage_label.text() == expected
    fresh_req = replace(req, request_id="fresh")
    fresh = RecordingSession(service, fresh_req, RecordingCallbacks())
    fresh_context = RecordingProcessContext(fresh_req, "forward", False, session=fresh)
    host._recording_process_contexts[fresh_req.request_id] = fresh_context
    host._active_recording_process_id = fresh_req.request_id
    panel.set_current_stage("分析中")
    host._on_process_recording_finalizing(old)
    assert panel.stage_label.text() == "分析中"
    host._on_process_recording_finalizing(fresh)
    assert panel.stage_label.text() == "数据保存中"
    host._finish_process_recording_status(context, "待检测", "等待开始", "pending")
    assert workspace.status_label.text() == "数据保存中"  # stale terminal
    panel.set_current_stage("分析中")
    workspace.set_condition_context("forward", status="分析中")
    host._finish_process_recording_status(fresh_context, "待检测", "等待开始", "pending")
    assert workspace.status_label.text() == "分析中"  # newer state
    host._on_process_recording_finalizing(fresh)
    assert panel.stage_label.text() == "分析中"  # once per session
    workspace.set_condition_context("forward", status="数据保存中")
    workspace.set_condition_context("reverse", status="分析完成")
    workspace.set_active_condition("reverse")
    host._get_active_product_condition_key = lambda: "reverse"
    host._finish_process_recording_status(fresh_context, "待检测", "等待开始", "pending")
    assert workspace._condition_contexts["forward"]["status"] == "待检测"
    assert workspace.status_label.text() == "分析完成"
    panel.close()
    workspace.close()


def test_ve_trusted_result_admission_never_scans_audio(ui_qapp, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unit_test.base.test_recording_finalization import prepared_ve_result
    from ui.sequence.recording_process_context import RecordingProcessContext
    req, descriptor, _, _ = prepared_ve_result(tmp_path, {"enabled":False})
    audio = read_result(descriptor, req).audio
    host = main_host(SimpleNamespace(), tmp_path)
    from unit_test.ui.test_streaming_event_dispatch import _Workspace
    host.channel_workspace = _Workspace(req.channels)
    host._recording_input_channels = req.channels
    host._active_input_channels = list(req.channels)
    session = SimpleNamespace(request=req, accept_result=mock.Mock(), reject_result=mock.Mock())
    context = RecordingProcessContext(req, "forward", False, session=session)
    host._recording_process_contexts = {req.request_id:context}
    host._active_recording_process_id = req.request_id
    def forbidden(*args, **kwargs):
        raise AssertionError("full audio scan on trusted VE delivery")
    monkeypatch.setattr(np, "isfinite", forbidden)
    monkeypatch.setattr(np, "array_equal", forbidden)
    host._on_process_recording_result(session, audio)
    session.accept_result.assert_called_once_with()
    session.reject_result.assert_not_called()
    assert context.validated_audio is audio


def test_legacy_reader_retains_writable_array_contract(tmp_path):
    req = request(tmp_path)
    _, descriptor = capture_audio(req)
    descriptor = replace(descriptor, sample_digest=None, digest_algorithm=None)
    audio = read_result(descriptor, None).audio
    assert audio.multi.flags.writeable
    assert audio.mono.flags.writeable
    assert audio.waveforms == ()


def test_ve_mono_overflow_is_rejected_in_background_with_quality_disabled(tmp_path):
    import hashlib
    import soundfile as sf
    from unit_test.base.test_recording_finalization import prepared_ve_result
    from base.wav_calibration_metadata import append_owned_recording_calibration_metadata_result
    req, descriptor, multi, _ = prepared_ve_result(tmp_path, {"enabled": False})
    multi.fill(np.finfo(np.float32).max)
    sf.write(req.path, multi, req.sample_rate, subtype="FLOAT")
    assert append_owned_recording_calibration_metadata_result(req.path, req.calibration_metadata).appended
    descriptor = replace(descriptor, sample_digest=hashlib.sha256(multi.astype("<f4").tobytes()).hexdigest())
    outcome = read_result(descriptor, req)
    assert outcome.audio is None
    assert "non-finite mono" in outcome.error


def test_display_shape_changes_invalidate_preparation(tmp_path):
    req = request(tmp_path)
    _, descriptor = capture_audio(req)
    audio = read_result(descriptor, req).audio
    audio.waveforms[0][0].shape = (1, -1)
    assert not audio.is_prepared_for(req)


def test_capture_slot_release_precedes_potentially_blocked_timing_log(tmp_path, monkeypatch):
    from base.log_manager import LogManager
    from unit_test.base.test_ve3668n_capture import start_persistent_capture
    entered, release = threading.Event(), threading.Event()
    logger = LogManager.set_log_handler("core")
    original_info = logger.info
    def blocked_timing(message, *args, **kwargs):
        if "stage=target_samples" in message:
            entered.set()
            assert release.wait(5)
        original_info(message, *args, **kwargs)
    monkeypatch.setattr(logger, "info", blocked_timing)
    capture, controller, _, _ = start_persistent_capture(tmp_path)
    try:
        assert entered.wait(3)
        assert capture.capture_slot_released.is_set()
    finally:
        release.set()
        assert capture.wait(3) is not None
        assert controller.release(.5).success


def test_cancelled_capture_skips_optional_success_timing_io(tmp_path, monkeypatch):
    from base.recording_capture import RecordingCapture
    from base.recording_process_protocol import RecordingCancelled
    from base.streaming_file_writer import StreamingWavWriter
    from unit_test.base.recording_process_fakes import FakeBackend, known_audio
    entered, release = threading.Event(), threading.Event()
    class Writer(StreamingWavWriter):
        def finalize(self):
            entered.set()
            assert release.wait(5)
            return super().finalize()
    backend = FakeBackend()
    capture = RecordingCapture(request(tmp_path), backend=backend, writer_factory=Writer)
    original_info = capture._logger.info
    def reject_success_timing(message, *args, **kwargs):
        assert "Recording timing" not in message
        return original_info(message, *args, **kwargs)
    monkeypatch.setattr(capture._logger, "info", reject_success_timing)
    capture.start()
    try:
        assert capture.started.wait(3)
        backend.stream.feed(known_audio(9))
        assert entered.wait(3)
        capture.cancel()
        release.set()
        assert isinstance(capture.wait(3), RecordingCancelled)
    finally:
        release.set()
        capture.cancel()
        assert capture.join(3)


def test_cancel_during_metadata_skips_optional_success_timing_io(tmp_path, monkeypatch):
    from base.recording_capture import RecordingCapture
    from base.recording_process_protocol import RecordingCancelled
    from base.wav_calibration_metadata import append_owned_recording_calibration_metadata_result
    from unit_test.base.recording_process_fakes import FakeBackend, known_audio
    entered, release = threading.Event(), threading.Event()

    def gated_appender(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        result = append_owned_recording_calibration_metadata_result(*args, **kwargs)
        assert result.appended
        return result

    backend = FakeBackend()
    metadata = {"recorded_channels": [
        {"wav_channel_index": index, "physical_input_channel": physical,
         "calibrated": False, "v2pa_factor": None, "standard_spl": None}
        for index, physical in enumerate((0, 2))
    ]}
    capture = RecordingCapture(request(tmp_path, calibration_metadata=metadata),
        backend=backend, metadata_appender=gated_appender)
    timing_messages = []
    monkeypatch.setattr(capture._logger, "info", lambda message, *args, **kwargs: timing_messages.append(message))
    capture.start()
    try:
        assert capture.started.wait(3)
        backend.stream.feed(known_audio(9))
        assert entered.wait(3)
        capture.cancel()
        timing_messages.clear()
        release.set()
        assert isinstance(capture.wait(3), RecordingCancelled)
        assert not any("Recording timing" in message for message in timing_messages)
        assert capture._handles_released
    finally:
        release.set()
        capture.cancel()
        assert capture.join(3)
