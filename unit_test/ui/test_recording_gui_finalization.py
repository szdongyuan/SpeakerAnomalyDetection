"""Bounded publication and actual Qt processing-state regressions."""
from dataclasses import replace
import threading
import logging
import numpy as np
import pytest
from unit_test.base.test_recording_finalization import capture_audio, read_result
from unit_test.base.test_recording_capture import request
from base.recording_final_result import prepare_main_recording_result
from unit_test.base.recording_process_fakes import known_audio

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


def test_envelope_products_are_frozen_and_bound_to_exact_request(tmp_path):
    from base.recording_final_result import PreparedRecordingResult
    req = request(tmp_path, channels=(2,0))
    _, descriptor = capture_audio(req)
    audio = prepare_main_recording_result(descriptor, req)
    assert audio.is_prepared_for(req)
    assert not audio.is_prepared_for(replace(req))
    assert not replace(audio).is_prepared_for(req)
    assert not replace(audio, descriptor=replace(descriptor)).is_prepared_for(req)
    assert not PreparedRecordingResult(descriptor, audio.waveforms).is_prepared_for(req)
    assert len(audio.waveforms) == 2
    for array in (a for pair in audio.waveforms for a in pair):
        with pytest.raises(ValueError):
            array.flat[0] = 1
        with pytest.raises(ValueError):
            array.setflags(write=True)
    for column, waveform in enumerate(audio.waveforms):
        expected = legacy_display(known_audio()[2:9, req.channels][:,column], req.sample_rate, max_points=48000)
        for a,b in zip(waveform, expected):
            np.testing.assert_array_equal(a,b)


def test_calibration_skips_display_preparation(tmp_path):
    req = request(tmp_path, purpose="calibration", channels=(0,), trim_samples=0)
    _, descriptor = capture_audio(req)
    audio = read_result(descriptor, req).audio
    assert audio.waveforms == ()
    assert not audio.is_prepared_for(req)


def test_child_reports_request_correlated_local_durations(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    req = request(tmp_path)
    _, descriptor = capture_audio(req)
    timing = descriptor.finalization_timing
    assert 0 <= timing.target_to_file_closed <= timing.target_to_descriptor
    assert 0 <= timing.metadata_seconds <= timing.target_to_descriptor
    assert prepare_main_recording_result(descriptor, req).is_prepared_for(req)
    messages = [r.getMessage() for r in caplog.records if req.request_id in r.getMessage()]
    for stage in ("target_samples", "drain_file_close", "metadata", "final_envelope", "descriptor_ready"):
        assert any("stage=" + stage in m for m in messages), stage


def test_finalizing_is_once_only_and_cancel_blocks_fallback():
    from base.recording_service import RecordingService, RecordingSession, RecordingCallbacks
    from base.recording_timing_logger import RecordingTimingLogger
    from base.recording_diagnostics import RecordingDiagnostics
    from types import SimpleNamespace
    service = object.__new__(RecordingService)
    service._logger = logging.Logger("finalizing-callback-test", logging.ERROR)
    service._timing_logger = RecordingTimingLogger()
    service._recording_diagnostics = RecordingDiagnostics(
        service._logger, timing_logger=service._timing_logger, categories=("callback_finalizing",))
    service._capture_session = None
    seen = []
    session = RecordingSession(service, SimpleNamespace(request_id="once"), RecordingCallbacks(finalizing=lambda s: seen.append(s)))
    service._notify_finalizing(session)
    service._notify_finalizing(session)
    assert seen == [session]
    cancelled = RecordingSession(service, SimpleNamespace(request_id="once"), RecordingCallbacks(finalizing=lambda s: seen.append(s)))
    cancelled.cancel_requested = True
    service._notify_finalizing(cancelled)
    assert seen == [session]


def test_legacy_reader_retains_writable_array_contract(tmp_path):
    req = request(tmp_path)
    _, descriptor = capture_audio(req)
    descriptor = replace(descriptor, sample_digest=None, digest_algorithm=None)
    audio = read_result(descriptor, None).audio
    assert audio.multi.flags.writeable
    assert audio.mono.flags.writeable
    assert audio.waveforms == ()


def test_ve_overrange_pcm24_has_finite_mono_with_quality_disabled(tmp_path):
    import hashlib
    import soundfile as sf
    from base.save_data import save_audio_simple
    from unit_test.base.test_recording_finalization import prepared_ve_result
    from base.wav_calibration_metadata import append_owned_recording_calibration_metadata_result
    req, descriptor, multi, _ = prepared_ve_result(tmp_path, {"enabled": False})
    multi.fill(np.finfo(np.float32).max)
    save_audio_simple(req.path, multi, req.sample_rate)
    expected = np.full(multi.shape, 8388607 / 8388608, dtype=np.float32)
    assert sf.info(req.path).subtype == "PCM_24"
    assert append_owned_recording_calibration_metadata_result(req.path, req.calibration_metadata).appended
    descriptor = replace(descriptor, sample_digest=hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest())
    outcome = read_result(descriptor, req)
    assert outcome.error is None
    np.testing.assert_array_equal(outcome.audio.multi, expected)
    np.testing.assert_array_equal(outcome.audio.mono, expected.mean(axis=1))


def test_display_shape_changes_invalidate_preparation(tmp_path):
    req = request(tmp_path)
    _, descriptor = capture_audio(req)
    audio = prepare_main_recording_result(descriptor, req)
    audio.waveforms[0][0].shape = (1, -1)
    assert not audio.is_prepared_for(req)


def test_capture_slot_release_precedes_potentially_blocked_timing_log(tmp_path, monkeypatch):
    from base.log_manager import LogManager
    from unit_test.base.test_vkinging_capture import start_persistent_capture
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
        # Startup observations precede cancellation; success-only tail records do not.
        assert not any(f"stage={stage}" in message for stage in (
            "target_samples", "drain_file_close", "metadata", "descriptor_ready"))
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
