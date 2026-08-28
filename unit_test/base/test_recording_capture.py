import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf

from base.recording_capture import RecordingCapture, capture_queue_capacity
from base.recording_process_protocol import (
    RecordingCancelled, RecordingEvent, RecordingFailure, RecordingPreview, RecordingRequest, RecordingResult,
)
from base.streaming_waveform_accumulator import StreamingWaveformSnapshot
from base.wav_calibration_metadata import read_wav_calibration_metadata
from unit_test.base.recording_process_fakes import (
    ControlledWriter, FakeBackend, FakeStatus, MetadataFileFaults, device_info, known_audio,
)


def request(tmp_path, **overrides):
    values = dict(request_id="capture-1", purpose="main", sample_rate=100,
                  target_samples=9, channels=(0, 2), device=device_info(),
                  path=str(tmp_path / "recording.wav"), streaming=True,
                  trim_samples=2, monitor={}, calibration_metadata=None,
                  validation_thresholds={"enabled": False})
    values.update(overrides)
    return RecordingRequest(**values)


@pytest.fixture
def captures():
    active = []
    def make(req, **kwargs):
        backend = kwargs.pop("backend", FakeBackend())
        capture = RecordingCapture(req, backend=backend, **kwargs)
        active.append((capture, kwargs.get("writer_factory")))
        capture.start()
        assert capture.started.wait(3) or capture.done.is_set()
        return capture, backend
    yield make
    for capture, writer in active:
        if isinstance(writer, ControlledWriter):
            writer.release.set()
        capture.cancel()
        capture.wait(5)
        capture._thread.join(3)
        assert not capture._thread.is_alive()


def feed_all(backend, *, mutate=False):
    data = known_audio()
    for chunk in (data[:2], data[2:5], data[5:]):
        backend.stream.feed(chunk, mutate=mutate)
    return data


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("channels", [(0,), (0, 2), (2, 0)])
def test_capture_owns_input_and_trims_once(tmp_path, captures, streaming, channels):
    writer = ControlledWriter(pause=True)
    capture, backend = captures(request(tmp_path, streaming=streaming, channels=channels), writer_factory=writer)
    expected = feed_all(backend, mutate=True)[:9, channels][2:]
    writer.release.set()
    result = capture.wait(3)
    assert isinstance(result, RecordingResult)
    audio, rate = sf.read(result.path, dtype="float32", always_2d=True)
    np.testing.assert_array_equal(audio, expected)
    np.testing.assert_array_equal(audio.mean(axis=1), expected.mean(axis=1))
    assert (result.raw_frames, result.final_frames, rate) == (9, 7, 100)
    assert result.channels == channels
    assert result.handles_released and backend.stream.closed and writer.closed
    assert sf.info(result.path).subtype == "FLOAT"
    assert backend.stream.capture_pid == writer.writer_pid == os.getpid()


def test_request_is_validated_picklable_snapshot(tmp_path):
    device = device_info()
    thresholds = {"enabled": False}
    req = request(tmp_path, device=device, validation_thresholds=thresholds)
    device["name"] = "changed"
    thresholds["enabled"] = True
    assert req.device["name"] == "fake recording device"
    assert req.validation_thresholds["enabled"] is False
    with pytest.raises(TypeError):
        req.device["name"] = "changed"
    assert pickle.loads(pickle.dumps(req)) == req
    for override in ({"channels": (0, 0)}, {"channels": (3,)}, {"target_samples": 0},
                     {"sample_rate": True}, {"path": "relative.wav"}, {"trim_samples": -1}):
        with pytest.raises(ValueError):
            request(tmp_path, **override)


def test_queue_capacity_is_explicit_and_saturation_fails(tmp_path, captures):
    assert capture_queue_capacity(100, 2, blocksize=8) == (200, 1600)
    assert capture_queue_capacity(100, 2, blocksize=300) == (300, 2400)
    writer = ControlledWriter(pause=True)
    capture, backend = captures(request(tmp_path, target_samples=500), writer_factory=writer, blocksize=8)
    backend.stream.feed(known_audio(4))
    assert writer.entered.wait(3)
    backend.stream.feed(known_audio(200))
    backend.stream.feed(known_audio(1))
    assert capture.queued_frames <= capture.queue_capacity_frames == 200
    writer.release.set()
    result = capture.wait(3)
    assert isinstance(result, RecordingFailure)
    assert result.stage == "capture" and "queue" in result.message
    assert result.request_id == "capture-1" and result.path == str(tmp_path / "recording.wav")
    assert result.handles_released and backend.stream.closed


@pytest.mark.parametrize("overflow", [True, False])
def test_input_overflow_fails_but_output_warning_does_not(tmp_path, captures, overflow):
    capture, backend = captures(request(tmp_path))
    backend.stream.feed(known_audio(), FakeStatus(input_overflow=overflow, output_underflow=not overflow))
    result = capture.wait(3)
    assert isinstance(result, RecordingFailure if overflow else RecordingResult)
    if overflow:
        assert "overflow" in result.message
    else:
        assert any("output underflow" in text for text in result.warnings)


@pytest.mark.parametrize("streaming,monitor", [(False, False), (True, False), (False, True), (True, True)])
def test_effective_mode_and_monitor_gain_mute_fade(tmp_path, captures, streaming, monitor):
    settings = dict(enabled=monitor, device=device_info(), channels=(1,), gain_db=6.020599913279624,
                    mute_leading_samples=3, fade_in_samples=4)
    capture, backend = captures(request(tmp_path, streaming=streaming, monitor=settings, trim_samples=0))
    assert capture.request.effective_streaming is (streaming or monitor)
    outputs = []
    expected = known_audio()
    for chunk in (expected[:2], expected[2:5], expected[5:]):
        outputs.append(backend.stream.feed(chunk))
    assert isinstance(capture.wait(3), RecordingResult)
    if monitor:
        actual = np.concatenate(outputs)
        mono = expected[:9, [0, 2]].mean(axis=1)
        play = np.pad(np.clip(mono * 2, -1, 1), (0, 3))
        play[:3] = 0
        play[3:5] *= np.array([0, .5], dtype=np.float32)
        np.testing.assert_allclose(actual[:, 1], play, atol=1e-7)
        np.testing.assert_array_equal(actual[:, 0], 0)


@pytest.mark.parametrize("fail_at", ["write", "close"])
def test_wav_failures_never_complete(tmp_path, captures, fail_at):
    capture, backend = captures(request(tmp_path), writer_factory=ControlledWriter(fail_at=fail_at))
    feed_all(backend)
    result = capture.wait(3)
    assert isinstance(result, RecordingFailure)
    assert "injected" in result.message


def test_missing_or_rejected_metadata_is_warning(tmp_path, captures):
    capture, backend = captures(request(tmp_path), metadata_appender=lambda *args, **kwargs: False)
    feed_all(backend)
    result = capture.wait(3)
    assert isinstance(result, RecordingResult) and not result.metadata_appended
    assert any("metadata" in text for text in result.warnings)


@pytest.mark.parametrize("stage", ["source", "temporary", "validation"])
@pytest.mark.parametrize("close_fails", [True, False])
def test_actual_metadata_file_failures_preserve_capture_ownership(
    tmp_path, captures, monkeypatch, stage, close_fails,
):
    faults = MetadataFileFaults(stage, close_fails=close_fails)
    faults.install(monkeypatch)
    metadata = {"recorded_channels": [
        {"wav_channel_index": 0, "physical_input_channel": 0, "calibrated": False},
        {"wav_channel_index": 1, "physical_input_channel": 2, "calibrated": False},
    ]}
    capture, backend = captures(request(tmp_path, trim_samples=0, calibration_metadata=metadata))
    feed_all(backend)
    try:
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure if close_fails else RecordingResult)
        assert outcome.handles_released is not close_fails
        assert outcome.path == str(tmp_path / "recording.wav")
        assert all(boundary.close_attempts == 1 for boundary in faults.files)
        if close_fails:
            assert outcome.stage == "metadata"
            assert f"metadata {stage} close failure" in outcome.message
            assert outcome.cleanup_paths == tuple(faults.temporary_paths)
            assert pickle.loads(pickle.dumps(outcome)).cleanup_paths == outcome.cleanup_paths
            assert all(os.path.exists(path) for path in outcome.cleanup_paths)
            assert any(not boundary.wrapped.closed for boundary in faults.files)
        else:
            assert not outcome.metadata_appended
            assert any("metadata" in warning for warning in outcome.warnings)
            assert all(boundary.wrapped.closed for boundary in faults.files)
            assert not any(os.path.exists(path) for path in faults.temporary_paths)
        np.testing.assert_array_equal(sf.read(outcome.path, dtype="float32")[0], known_audio()[:9, [0, 2]])
    finally:
        faults.release_all()


@pytest.mark.parametrize("cancel", [False, True])
def test_metadata_cleanup_failure_keeps_warning_and_owned_paths(
    tmp_path, captures, monkeypatch, cancel,
):
    from base import wav_calibration_metadata as module
    faults = MetadataFileFaults("temporary", close_fails=False)
    faults.install(monkeypatch)
    real_unlink = module.os.unlink

    def fail_owned_unlink(path):
        if os.fspath(path) in faults.temporary_paths:
            if cancel:
                capture.cancel()
            raise OSError("injected metadata cleanup failure")
        return real_unlink(path)

    monkeypatch.setattr(module.os, "unlink", fail_owned_unlink)
    metadata = {"recorded_channels": [{"wav_channel_index": 0, "calibrated": False}]}
    capture, backend = captures(request(tmp_path, trim_samples=0, channels=(0,), calibration_metadata=metadata))
    feed_all(backend)
    try:
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingCancelled if cancel else RecordingResult)
        assert outcome.handles_released
        assert outcome.cleanup_paths == tuple(faults.temporary_paths)
        assert outcome.path not in outcome.cleanup_paths
        assert pickle.loads(pickle.dumps(outcome)).cleanup_paths == outcome.cleanup_paths
        assert all(boundary.wrapped.closed for boundary in faults.files)
        assert all(os.path.exists(path) for path in outcome.cleanup_paths)
        if not cancel:
            assert not outcome.metadata_appended
            assert any("metadata" in warning for warning in outcome.warnings)
    finally:
        faults.release_all()


def test_quality_gate_runs_after_trim(tmp_path, captures):
    capture, backend = captures(request(tmp_path, validation_thresholds={"enabled": True}))
    data = np.zeros((9, 3), dtype=np.float32)
    data[:2] = 1
    backend.stream.feed(data)
    result = capture.wait(3)
    assert isinstance(result, RecordingFailure) and result.stage == "validation"
    np.testing.assert_array_equal(sf.read(result.path, dtype="float32")[0], np.zeros((7, 2)))


def test_calibration_skips_product_trim_quality_and_preview(tmp_path, captures):
    capture, backend = captures(request(tmp_path, purpose="calibration", channels=(2,),
                                       validation_thresholds={"enabled": True}))
    backend.stream.feed(np.zeros((9, 3), dtype=np.float32))
    result = capture.wait(3)
    assert isinstance(result, RecordingResult)
    assert result.raw_frames == result.final_frames == 9
    assert result.purpose == "calibration" and not result.warnings
    assert capture.snapshot(generation=1, sequence=1) is None


def test_cancel_drains_accepted_audio_without_success(tmp_path, captures):
    writer = ControlledWriter(pause=True)
    capture, backend = captures(request(tmp_path), writer_factory=writer)
    backend.stream.feed(known_audio(5))
    assert writer.entered.wait(3)
    capture.cancel()
    backend.stream.feed(known_audio(5))
    writer.release.set()
    result = capture.wait(3)
    assert isinstance(result, RecordingCancelled)
    assert result.raw_frames == result.final_frames == 5
    np.testing.assert_array_equal(sf.read(result.path, dtype="float32")[0], known_audio(5)[:, [0, 2]])


def test_wrong_device_identity_fails_before_open(tmp_path, captures):
    backend = FakeBackend()
    backend.device["name"] = "different device at same index"
    capture, backend = captures(request(tmp_path), backend=backend)
    result = capture.wait(3)
    assert isinstance(result, RecordingFailure) and result.stage == "device"
    assert backend.stream is None


def test_callback_exception_becomes_diagnostic_failure(tmp_path, captures):
    class UnreadableDriverArray(np.ndarray):
        def __getitem__(self, key):
            raise RuntimeError("injected borrowed-buffer read failure")
    capture, backend = captures(request(tmp_path))
    data = known_audio().view(UnreadableDriverArray)
    # A PortAudio callback must not just escape and leave its owner waiting forever.
    backend.stream.config["callback"](data, len(data), None, FakeStatus())
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingFailure)
    assert outcome.stage == "capture" and "borrowed-buffer" in outcome.message
    assert outcome.handles_released and backend.stream.closed


def test_preview_is_cumulative_owned_bounded_and_nonblocking(tmp_path, captures):
    capture, backend = captures(request(tmp_path, sample_rate=10000, target_samples=12000))
    data = known_audio(12000)
    backend.stream.feed(data[:6000])
    deadline = time.monotonic() + 3
    preview = None
    while time.monotonic() < deadline:
        preview = capture.snapshot(generation=2, sequence=1)
        if preview is not None and preview.sample_stop == 5998:
            break
        time.sleep(.005)
    assert preview.sample_stop == 5998 and preview.channels == (0, 2)
    saved = preview.waveforms[0].amplitude.copy()
    backend.stream.feed(data[6000:])
    assert isinstance(capture.wait(3), RecordingResult)
    final = capture.snapshot(generation=2, sequence=2)
    assert final.sample_stop == 11998
    for waveform in final.waveforms:
        assert len(waveform.time) <= 4000
        assert waveform.time[0] == 0 and waveform.time[-1] == 11997 / 10000
        assert not waveform.time.flags.writeable and not waveform.amplitude.flags.writeable
    np.testing.assert_array_equal(preview.waveforms[0].amplitude, saved)
    # The sender never waits for the consumer's accumulator mutation lock.
    with capture._waveform_lock:
        assert capture.snapshot(generation=2, sequence=3) is None


def test_capture_dependencies_import_without_qt_or_device_initialization():
    code = """
import sys
import base.recording_capture
import base.multichannel_waveform_session
assert not any(name == 'ui' or name.startswith(('ui.', 'PyQt', 'PySide')) for name in sys.modules)
assert 'sounddevice' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_preview_protocol_owns_arrays_and_validates_shape_after_pickle():
    time_axis = np.array([0, .1], dtype=np.float64)
    amplitude = np.array([.2, .3], dtype=np.float32)
    waveform = StreamingWaveformSnapshot(time_axis, amplitude, 2)
    preview = RecordingPreview("one", 1, 1, 2, (0,), (waveform,))
    amplitude.fill(99)
    np.testing.assert_array_equal(preview.waveforms[0].amplitude, np.array([.2, .3], dtype=np.float32))
    restored = pickle.loads(pickle.dumps(preview))
    assert not restored.waveforms[0].amplitude.flags.writeable
    with pytest.raises(ValueError):
        RecordingPreview("one", 1, 1, 2, (0, 2), (waveform,))
    with pytest.raises(ValueError):
        RecordingEvent(1, "one", "start", payload=object())


def test_cancel_during_trim_reports_actual_final_frames(tmp_path, captures):
    writer = ControlledWriter(pause=True)
    capture, backend = captures(request(tmp_path), writer_factory=writer)
    rewrite = capture._rewrite_trimmed
    def trim_then_cancel(audio):
        rewrite(audio)
        capture.cancel()
    capture._rewrite_trimmed = trim_then_cancel
    feed_all(backend)
    writer.release.set()
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingCancelled)
    assert outcome.final_frames == sf.info(outcome.path).frames == 7


def test_real_metadata_and_overlarge_trim_keep_samples(tmp_path, captures):
    metadata = {"recorded_channels": [
        {"wav_channel_index": 0, "physical_input_channel": 0, "calibrated": False},
        {"wav_channel_index": 1, "physical_input_channel": 2, "calibrated": False},
    ]}
    capture, backend = captures(request(tmp_path, trim_samples=9, calibration_metadata=metadata))
    feed_all(backend)
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingResult)
    assert outcome.final_frames == 9 and outcome.metadata_appended
    assert any("trim skipped" in message for message in outcome.warnings)
    assert read_wav_calibration_metadata(outcome.path) is not None
    np.testing.assert_array_equal(sf.read(outcome.path, dtype="float32")[0], known_audio()[:9, [0, 2]])


def test_trim_replace_failure_is_terminal_and_cleans_only_own_temp(tmp_path, captures, monkeypatch):
    import base.recording_capture as module
    unrelated = tmp_path / "unrelated.wav"
    unrelated.write_bytes(b"keep me")
    def fail_replace(source, target):
        raise OSError("injected trim replacement failure")
    monkeypatch.setattr(module.os, "replace", fail_replace)
    capture, backend = captures(request(tmp_path))
    feed_all(backend)
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingFailure) and outcome.stage == "trim"
    assert not list(tmp_path.glob(".recording-trim-*"))
    assert unrelated.read_bytes() == b"keep me"


@pytest.mark.parametrize("target,close_fails,processing_fails", [
    ("initial_reader", True, False),
    ("metadata_reader", True, False),
    ("trim_writer", True, False),
    ("initial_reader", False, True),
    ("metadata_reader", False, True),
    ("trim_writer", False, True),
    ("initial_reader", True, True),
    ("metadata_reader", True, True),
    ("trim_writer", True, True),
])
def test_finalization_file_ownership_tracks_close_separately(
    tmp_path, captures, monkeypatch, target, close_fails, processing_fails,
):
    import base.recording_capture as module
    real_sound_file = sf.SoundFile
    opened = []
    read_count = 0

    class ControlledSoundFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.close_attempts = 0

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def __len__(self):
            if processing_fails and target == "metadata_reader":
                raise OSError("injected metadata verification failure")
            return len(self.wrapped)

        def read(self, *args, **kwargs):
            if processing_fails:
                raise OSError("injected finalization read failure")
            return self.wrapped.read(*args, **kwargs)

        def write(self, *args, **kwargs):
            if processing_fails:
                raise OSError("injected trim write failure")
            return self.wrapped.write(*args, **kwargs)

        def close(self):
            self.close_attempts += 1
            if close_fails:
                raise OSError("injected finalization close failure")
            self.wrapped.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def open_file(path, *args, **kwargs):
        nonlocal read_count
        wrapped = real_sound_file(path, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else "r")
        if mode == "r":
            read_count += 1
            current = "initial_reader" if read_count == 1 else "metadata_reader"
        else:
            current = "trim_writer" if os.path.basename(str(path)).startswith(".recording-trim-") else "capture_writer"
        if current == target:
            controlled = ControlledSoundFile(wrapped)
            opened.append(controlled)
            return controlled
        return wrapped

    monkeypatch.setattr(module.sf, "SoundFile", open_file)
    capture, backend = captures(request(tmp_path))
    feed_all(backend)
    try:
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure)
        assert outcome.handles_released is not close_fails
        assert len(opened) == 1 and opened[0].close_attempts == 1
        assert opened[0].wrapped.closed is not close_fails
        retained = tuple(str(path) for path in tmp_path.glob(".recording-trim-*"))
        assert outcome.cleanup_paths == retained
        assert outcome.path == str(tmp_path / "recording.wav")
        assert outcome.path not in outcome.cleanup_paths
        assert pickle.loads(pickle.dumps(outcome)).cleanup_paths == retained
        if target == "trim_writer":
            # No replacement or removal while close is uncertain. A processing
            # failure with successful close still removes its owned temporary.
            assert bool(list(tmp_path.glob(".recording-trim-*"))) is close_fails
            with real_sound_file(outcome.path) as source:
                assert len(source) == 9
    finally:
        for controlled in opened:
            controlled.wrapped.close()


def test_trim_descriptor_close_uncertainty_prevents_cleanup(tmp_path, captures, monkeypatch):
    import base.recording_capture as module
    real_mkstemp = module.tempfile.mkstemp
    real_close = module.os.close
    allocated = []

    def allocate_temp(*args, **kwargs):
        descriptor, path = real_mkstemp(*args, **kwargs)
        allocated.append((descriptor, path))
        return descriptor, path

    def fail_descriptor_close(descriptor):
        if allocated and descriptor == allocated[0][0]:
            raise OSError("injected trim descriptor close failure")
        return real_close(descriptor)

    monkeypatch.setattr(module.tempfile, "mkstemp", allocate_temp)
    monkeypatch.setattr(module.os, "close", fail_descriptor_close)
    capture, backend = captures(request(tmp_path))
    feed_all(backend)
    try:
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure)
        assert not outcome.handles_released
        assert "descriptor close failure" in outcome.message
        assert len(allocated) == 1 and os.path.exists(allocated[0][1])
        assert outcome.cleanup_paths == (allocated[0][1],)
        assert outcome.path == str(tmp_path / "recording.wav")
    finally:
        for descriptor, _ in allocated:
            real_close(descriptor)


def test_failure_cleanup_paths_default_to_no_owned_temporary_files():
    outcome = RecordingFailure("one", "device", "recording.wav", "device unavailable")
    assert outcome.cleanup_paths == ()


def test_preview_failure_does_not_invalidate_audio(tmp_path, captures):
    capture, backend = captures(request(tmp_path))
    def fail_append(block):
        raise RuntimeError("injected envelope failure")
    capture._waveforms.append = fail_append
    feed_all(backend)
    outcome = capture.wait(3)
    assert isinstance(outcome, RecordingResult)
    assert sum("preview disabled" in warning for warning in outcome.warnings) == 1
    assert capture.snapshot(generation=1, sequence=1) is None


@pytest.mark.parametrize("sizes,mute,fade", [((2, 3, 7), 3, 4), ((3, 3, 6), 3, 4), ((4, 8), 0, 4)])
def test_legacy_monitor_characterization(sizes, mute, fade):
    from base.recording_capture import apply_monitor_startup_mute
    from base.streaming_audio_processor import StreamingAudioProcessor
    processor = StreamingAudioProcessor()
    processor._monitor_mute_leading_samples = mute
    offset = 0
    for size in sizes:
        data = np.ones(size, dtype=np.float32)
        actual = processor._apply_monitor_startup_mute(data, fade)
        shared = apply_monitor_startup_mute(data, mute_total=mute, emitted_before=offset, fade_len=fade)
        np.testing.assert_array_equal(shared, actual)
        offset += size
