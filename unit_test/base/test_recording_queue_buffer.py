"""Backend queue defaults and deterministic blocked-writer regressions."""
import time
from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf

from base.recording_capture import RecordingCapture, capture_queue_capacity
from base.recording_process_protocol import RecordingFailure, RecordingRequest, RecordingResult
from base.ve3668n_capture_timing import VeCaptureProgress
from base.wav_pcm24 import quantize_pcm24
from unit_test.base.recording_process_fakes import ControlledWriter, device_info
from unit_test.base.ve3668n_fakes import (
    capture_request, device_info as ve_device_info, input_config, wav_metadata,
)


def request(path, backend, *, target_samples=9, trim_samples=0, sample_rate=None):
    if sample_rate is None:
        sample_rate = 44100 if backend == "ve" else 100
    if backend == "ve":
        channels = (0, 1, 2, 3, 4)
        device = ve_device_info(
            physical_channels=list(channels), input_config=input_config(sample_rate))
        metadata = wav_metadata(("none",) * 5, sample_rate=sample_rate,
                                physical_channels=channels)
        metadata["acquisition"]["machine_id"] = device["machine_id"]
        return capture_request(
            path, sample_rate=sample_rate, channels=channels, device=device,
            calibration_metadata=metadata, target_samples=target_samples,
            trim_samples=trim_samples)
    return RecordingRequest(
        request_id="soundcard-queue", purpose="main", sample_rate=sample_rate,
        target_samples=target_samples, channels=(0, 2), device=device_info(),
        path=str(path), streaming=False, trim_samples=trim_samples, monitor={},
        calibration_metadata=None, validation_thresholds={"enabled": False})


@pytest.mark.parametrize("backend,sample_rate,frames,byte_count", [
    ("ve", 44100, 1323000, 26460000),
    ("ve", 51200, 1536000, 30720000),
    ("soundcard", 100, 200, 1600),
])
@pytest.mark.parametrize("options", [{}, {"queue_seconds": None}], ids=["omitted", "none"])
def test_backend_default_capacity(tmp_path, backend, sample_rate, frames, byte_count, options):
    capture = RecordingCapture(request(tmp_path / "default.wav", backend, sample_rate=sample_rate),
                               blocksize=8, **options)
    assert (capture.queue_capacity_frames, capture.queue_capacity_bytes) == (frames, byte_count)


def test_capacity_helper_keeps_two_second_default():
    assert capture_queue_capacity(44100, 5) == (88200, 1764000)
    assert capture_queue_capacity(100, 2, blocksize=300) == (300, 2400)


@pytest.mark.parametrize("backend", ["ve", "soundcard"])
@pytest.mark.parametrize("seconds", [.25, 3, 21.0])
def test_explicit_duration_overrides_backend_default(tmp_path, backend, seconds):
    req = request(tmp_path / "override.wav", backend)
    capture = RecordingCapture(req, blocksize=8, queue_seconds=seconds)
    assert (capture.queue_capacity_frames, capture.queue_capacity_bytes) == capture_queue_capacity(
        req.sample_rate, len(req.channels), blocksize=8, seconds=seconds)


@pytest.mark.parametrize("backend", ["ve", "soundcard"])
@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_explicit_duration_is_rejected(tmp_path, backend, seconds):
    with pytest.raises(ValueError, match="queue dimensions and duration must be positive"):
        RecordingCapture(request(tmp_path / "invalid.wav", backend), queue_seconds=seconds)


class ManualVeStream:
    """Injected VE boundary; the test feeds frames without native I/O or a producer thread."""

    def __init__(self, *, request, callback, fail, stop_event):
        self.request = request
        self.callback = callback
        self.stop_event = stop_event
        self.started_at = None
        self.progress = VeCaptureProgress(None, 0, None)
        self.handles_released = False
        self.diagnostics = ()
        self.closed = False

    def start(self):
        self.started_at = time.monotonic()
        self.progress = VeCaptureProgress(self.started_at, 0, self.started_at)
        return True

    def feed(self, block):
        assert not self.stop_event.is_set()
        assert 0 < len(block) <= 2048
        self.progress = VeCaptureProgress(
            self.started_at, self.progress.frames + len(block), time.monotonic())
        self.callback(block, len(block), None, None)

    def progress_snapshot(self):
        return self.progress

    def stop(self):
        self.handles_released = True

    def close(self):
        assert self.handles_released
        self.closed = True


@contextmanager
def blocked_capture(req):
    writer = ControlledWriter(pause=True)
    capture = RecordingCapture(req, ve_stream_factory=ManualVeStream, writer_factory=writer)
    capture.start()
    try:
        assert capture.started.wait(3)
        yield capture, capture._native_stream, writer
    finally:
        writer.release.set()
        if not capture.done.is_set():
            capture.cancel()
        assert capture.join(5), "test capture did not stop"


def ordered_audio(frames):
    # Every frame/channel has a distinct bounded, exactly representable PCM24 value.
    values = np.arange(frames * 5, dtype=np.int32).reshape(frames, 5) - 2000000
    return values.astype(np.float32) / np.float32(8388608)


def test_ve_recovers_eight_seconds_of_backlog_with_exact_wav_content(tmp_path):
    initial_frames, backlog_frames, trim = 2048, 8 * 44100, 13
    audio = ordered_audio(initial_frames + backlog_frames)
    req = request(tmp_path / "recovered.wav", "ve", target_samples=len(audio), trim_samples=trim)
    with blocked_capture(req) as (capture, stream, writer):
        stream.feed(audio[:initial_frames])
        assert writer.entered.wait(3)
        assert capture.queued_frames == 0
        for start in range(initial_frames, len(audio), 2048):
            stream.feed(audio[start:start + 2048])
            assert capture._failure is None
        assert capture.queued_frames == backlog_frames > 2 * req.sample_rate
        assert capture.raw_frames == len(audio)
        writer.release.set()
        outcome = capture.wait(5)
        assert isinstance(outcome, RecordingResult), outcome
        assert capture.queued_frames == 0
        assert capture.consumed_frames == len(audio)
        assert outcome.raw_frames == len(audio)
        assert outcome.final_frames == len(audio) - trim
        assert outcome.handles_released and stream.closed and writer.closed
        assert capture.capture_slot_released.is_set()

    saved, rate = sf.read(outcome.path, dtype="float32", always_2d=True)
    assert rate == 44100
    assert sf.info(outcome.path).subtype == "PCM_24"
    np.testing.assert_array_equal(saved, quantize_pcm24(audio[trim:]))


@pytest.mark.parametrize("sample_rate,byte_count", [(44100, 26460000), (51200, 30720000)])
def test_ve_accepts_exact_thirty_second_capacity_then_fails_one_frame_over(
        tmp_path, sample_rate, byte_count):
    initial_frames, capacity = 2048, 30 * sample_rate
    req = request(tmp_path / "overflow.wav", "ve", sample_rate=sample_rate,
                  target_samples=initial_frames + capacity + 1, trim_samples=0)
    block = np.zeros((2048, 5), dtype=np.float32)
    with blocked_capture(req) as (capture, stream, writer):
        stream.feed(block)
        assert writer.entered.wait(3)
        assert capture.queued_frames == 0
        for start in range(0, capacity, len(block)):
            stream.feed(block[:min(len(block), capacity - start)])
            assert capture._failure is None
        assert capture.queued_frames == capture.queue_capacity_frames == capacity
        assert capture.raw_frames == initial_frames + capacity
        assert not capture._stop_requested.is_set()

        stream.feed(block[:1])
        assert capture._failure == (
            "capture", f"audio queue capacity exceeded ({byte_count} bytes)")
        assert capture._stop_requested.is_set()
        assert capture.queued_frames == capacity
        assert capture.raw_frames == initial_frames + capacity
        writer.release.set()
        outcome = capture.wait(5)
        assert isinstance(outcome, RecordingFailure), outcome
        assert outcome.stage == "capture"
        assert outcome.message == f"audio queue capacity exceeded ({byte_count} bytes)"
        assert outcome.handles_released and stream.closed and writer.closed
        assert capture.queued_frames == 0
        assert not capture.capture_slot_released.is_set()
