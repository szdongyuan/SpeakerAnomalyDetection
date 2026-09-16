"""Streaming persistence and request-bound, single-read finalization contracts."""
from dataclasses import replace
import hashlib
import math

import numpy as np
import pytest
import soundfile as sf

from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingCancelled, RecordingResult
from base.recording_result_reader import ResultReader
from base.streaming_file_writer import StreamingWavWriter
from unit_test.base.recording_process_fakes import FakeBackend, known_audio
from unit_test.base.test_recording_capture import request
from unit_test.base.test_recording_service import Events, services


def capture_audio(req, *, data=None, writer_factory=StreamingWavWriter, **kwargs):
    backend = FakeBackend()
    capture = RecordingCapture(req, backend=backend, writer_factory=writer_factory, **kwargs)
    capture.start()
    assert capture.started.wait(3)
    data = known_audio(9) if data is None else data
    try:
        for block in (data[:2], data[2:5], data[5:]):
            if len(block):
                backend.stream.feed(block)
        if len(data) < req.target_samples:
            capture.cancel()
        outcome = capture.wait(3)
        assert capture.join(3)
        return capture, outcome
    finally:
        capture.cancel()
        capture.join(3)


def read_result(descriptor, req, **kwargs):
    outcomes = []
    reader = ResultReader(descriptor, outcomes.append, request=req, **kwargs)
    reader.start()
    reader.thread.join(3)
    assert not reader.thread.is_alive()
    assert len(outcomes) == 1
    return outcomes[0]


@pytest.mark.parametrize("trim", [0, 4, 9, 12])
@pytest.mark.parametrize("channels", [(0,), (2, 0)])
def test_writer_only_receives_retained_samples(tmp_path, trim, channels):
    written = []

    class Writer(StreamingWavWriter):
        def write_chunk(self, chunk):
            written.append(chunk.copy())
            super().write_chunk(chunk)

    req = request(tmp_path, trim_samples=trim, channels=channels)
    capture, result = capture_audio(req, writer_factory=Writer)
    assert isinstance(result, RecordingResult)
    effective_trim = trim if trim < 9 else 0
    expected = known_audio(9)[effective_trim:, channels]
    np.testing.assert_array_equal(np.concatenate(written), expected)
    assert capture.raw_frames == capture.consumed_frames == 9
    assert capture.written_frames == result.final_frames == len(expected)
    assert result.sample_digest == hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest()
    assert result.digest_algorithm == "sha256-float32-le"


def test_capture_never_reads_audio_or_rewrites_trim(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("capture attempted a full audio read or rewrite")
    monkeypatch.setattr(sf.SoundFile, "read", forbidden)
    monkeypatch.setattr(sf, "read", forbidden)
    capture, result = capture_audio(request(tmp_path, trim_samples=4))
    assert isinstance(result, RecordingResult), result
    assert not list(tmp_path.glob(".recording-trim-*"))


@pytest.mark.parametrize("block_frames", [1, 3, 65536])
def test_parent_reads_each_sample_once_and_validates_digest(tmp_path, block_frames):
    req = request(tmp_path, trim_samples=4)
    _, result = capture_audio(req)
    reads = []

    class Source(sf.SoundFile):
        def read(self, frames, **kwargs):
            block = super().read(frames, **kwargs)
            reads.append(len(block))
            return block

    outcome = read_result(result, req, block_frames=block_frames, opener=Source)
    assert outcome.error is None
    expected = known_audio(9)[4:, req.channels]
    assert sum(reads) == len(expected)
    np.testing.assert_array_equal(outcome.audio.multi, expected)
    np.testing.assert_array_equal(outcome.audio.mono, expected.mean(axis=1))


@pytest.mark.parametrize("damage", ["samples", "truncated", "missing_digest", "wrong_request"])
def test_invalid_disk_or_evidence_never_delivers_audio(tmp_path, damage):
    req = request(tmp_path)
    _, result = capture_audio(req)
    if damage in ("samples", "truncated"):
        audio = known_audio(9)[2:, req.channels].copy()
        if damage == "samples":
            audio[0, 0] += 1
        else:
            audio = audio[:-1]
        sf.write(req.path, audio, req.sample_rate, subtype="FLOAT")
    elif damage == "missing_digest":
        result = replace(result, sample_digest=None)
    else:
        req = replace(req, request_id="different")
    outcome = read_result(result, req)
    assert outcome.audio is None
    assert outcome.error
    assert outcome.handles_released


def test_quality_gate_uses_retained_audio_in_parent(tmp_path):
    req = request(tmp_path, validation_thresholds={"enabled": True})
    data = np.zeros((9, 3), dtype=np.float32)
    data[:2] = 1
    _, result = capture_audio(req, data=data)
    assert isinstance(result, RecordingResult), "child completion is provisional"
    outcome = read_result(result, req)
    assert outcome.audio is None
    assert "weak_signal" in outcome.error


@pytest.mark.parametrize("frames", [1, 3, 6])
def test_partial_cancel_counts_only_saved_trimmed_frames(tmp_path, frames):
    req = request(tmp_path, trim_samples=4)
    capture, result = capture_audio(req, data=known_audio(frames))
    assert isinstance(result, RecordingCancelled)
    assert result.raw_frames == capture.consumed_frames == frames
    assert result.final_frames == capture.written_frames == max(0, frames - 4)
    assert sf.info(req.path).frames == result.final_frames


def test_calibration_parent_skips_trim_and_quality(tmp_path):
    req = request(tmp_path, purpose="calibration", channels=(2,), trim_samples=4,
                  validation_thresholds={"enabled": True})
    _, result = capture_audio(req, data=np.zeros((9, 3), dtype=np.float32))
    outcome = read_result(result, req)
    assert outcome.error is None
    assert outcome.audio.multi.shape == (9, 1)


@pytest.mark.parametrize("calibrated", [False, True])
def test_soundcard_metadata_survives_capture_to_reader(tmp_path, calibrated):
    from base.wav_calibration_metadata import read_wav_calibration_metadata

    channels = (2, 0)
    metadata = {"recorded_channels": [
        {"wav_channel_index": index, "physical_input_channel": physical,
         "calibrated": calibrated, "v2pa_factor": 2.5 + index if calibrated else None,
         "standard_spl": 94.0 if calibrated else None}
        for index, physical in enumerate(channels)
    ]}
    req = request(tmp_path, channels=channels, calibration_metadata=metadata)
    _, descriptor = capture_audio(req)
    assert isinstance(descriptor, RecordingResult), descriptor
    assert descriptor.metadata_appended
    assert read_wav_calibration_metadata(req.path) == metadata

    outcome = read_result(descriptor, req)
    assert outcome.error is None
    assert outcome.handles_released
    assert outcome.audio is not None
    assert outcome.audio.descriptor == descriptor
    expected = known_audio(9)[req.trim_samples:, channels]
    np.testing.assert_array_equal(outcome.audio.multi, expected)
    np.testing.assert_array_equal(outcome.audio.mono, expected.mean(axis=1))
    assert read_wav_calibration_metadata(req.path) == metadata
    assert req.calibration_metadata.to_dict() == metadata


def prepared_ve_result(tmp_path, thresholds, *, limit=.02):
    from base.wav_calibration_metadata import append_owned_recording_calibration_metadata_result
    from unit_test.base.ve3668n_fakes import capture_request
    seed = capture_request(tmp_path / "ve.wav")
    device, metadata = seed.device.to_dict(), seed.calibration_metadata.to_dict()
    device["input_config"].update(range_min=-limit, range_max=limit)
    metadata["acquisition"].update(range_min=-limit, range_max=limit)
    req = replace(seed, target_samples=257, trim_samples=0, device=device,
                  calibration_metadata=metadata, validation_thresholds=thresholds)
    multi = np.random.default_rng(97).normal(0, .005, (257, 2)).astype(np.float32)
    sf.write(req.path, multi, req.sample_rate, subtype="FLOAT")
    assert append_owned_recording_calibration_metadata_result(req.path, metadata).appended
    descriptor = RecordingResult(req.request_id, req.purpose, req.path, req.sample_rate,
                                 req.channels, 257, 257, True,
                                 digest_algorithm="sha256-float32-le",
                                 sample_digest=hashlib.sha256(multi.astype("<f4").tobytes()).hexdigest())
    return req, descriptor, multi, device


@pytest.mark.parametrize("metric", ["min_peak", "min_rms_dbfs", "min_variance"])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_quality_threshold_boundaries_match_legacy_operation_order(tmp_path, metric, direction):
    from base.recording_settings import validate_recorded_audio
    thresholds = {"enabled": True, "min_peak": 0, "min_rms_dbfs": -200, "min_variance": 0}
    req, descriptor, multi, _ = prepared_ve_result(tmp_path, thresholds)
    scaled = multi / req.device["input_config"]["range_max"]
    mono = scaled.mean(axis=1)
    values = {"min_peak": float(np.max(np.abs(mono))),
              "min_rms_dbfs": 20 * math.log10(float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))),
              "min_variance": float(np.var(mono, dtype=np.float64))}
    boundary = values[metric]
    thresholds[metric] = boundary if direction == 0 else float(np.nextafter(boundary, direction * np.inf))
    req = replace(req, validation_thresholds=thresholds)
    expected, reason, detail = validate_recorded_audio(scaled, thresholds)
    result = read_result(descriptor, req)
    assert (result.audio is not None) is expected
    if not expected:
        assert f"{reason} {detail}" in result.error


def test_quality_uses_frozen_range_and_thresholds(tmp_path):
    thresholds = {"enabled": True, "min_peak": .1, "min_rms_dbfs": -100, "min_variance": 0}
    req, descriptor, _, device = prepared_ve_result(tmp_path, thresholds)
    thresholds["min_peak"] = 1000
    device["input_config"]["range_max"] = 10
    assert read_result(descriptor, req).error is None


def test_disabled_quality_never_allocates_scaled_voltage_array(tmp_path, monkeypatch):
    from base import recording_result_reader as module
    req, descriptor, _, _ = prepared_ve_result(tmp_path, {"enabled": False})
    real_empty = np.empty

    class NoScaling(np.ndarray):
        def __truediv__(self, other):
            raise AssertionError("disabled quality attempted voltage scaling")

    monkeypatch.setattr(module.np, "empty", lambda *args, **kwargs: real_empty(*args, **kwargs).view(NoScaling))
    monkeypatch.setattr(module, "validate_recorded_audio", lambda *args: pytest.fail("quality disabled"))
    assert read_result(descriptor, req).error is None


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_parent_revalidates_required_metadata(tmp_path, damage):
    req, descriptor, multi, _ = prepared_ve_result(tmp_path, {"enabled": False})
    sf.write(req.path, multi, req.sample_rate, subtype="FLOAT")
    if damage == "changed":
        from base.wav_calibration_metadata import append_owned_recording_calibration_metadata_result
        metadata = req.calibration_metadata.to_dict()
        metadata["acquisition"]["machine_id"] = "different-device"
        assert append_owned_recording_calibration_metadata_result(req.path, metadata).appended
    result = read_result(descriptor, req)
    assert result.audio is None and "metadata" in result.error


@pytest.mark.parametrize("fault", ["rollback", "retained"])
def test_optional_metadata_cannot_hide_uncertain_restoration_or_handles(tmp_path, fault):
    from base.recording_process_protocol import RecordingFailure
    from base.wav_calibration_metadata import WavCalibrationMetadataAppendResult
    def append(*args, **kwargs):
        return WavCalibrationMetadataAppendResult(
            appended=False, handles_released=True, primary_error="append failed",
            rollback_succeeded=False if fault == "rollback" else None,
            rollback_errors=("rollback failed",) if fault == "rollback" else (),
            retained_handles=((str(tmp_path / "recording.wav"), object()),) if fault == "retained" else ())
    _, result = capture_audio(request(tmp_path), metadata_appender=append)
    assert isinstance(result, RecordingFailure)
    assert result.stage == "metadata" and "append failed" in result.message
    assert result.handles_released is (fault == "rollback")


@pytest.mark.parametrize("fault", ["quality", "digest", "missing_evidence"])
def test_service_never_offers_or_accepts_invalid_parent_result(tmp_path, services, fault):
    def reader(descriptor, completed, *, request):
        if fault == "digest":
            with sf.SoundFile(descriptor.path, mode="r+") as target:
                target.write(np.ones((1, len(descriptor.channels)), dtype=np.float32))
        elif fault == "missing_evidence":
            descriptor = replace(descriptor, sample_digest=None)
        return ResultReader(descriptor, completed, request=request)
    service = services(reader_factory=reader)
    events = Events()
    req = request(tmp_path, validation_thresholds={"enabled": fault == "quality", "min_peak": 100})
    session = service.start(req, events.callbacks)
    failure = events.failed.get(timeout=10)
    assert failure.stage == "read"
    assert session.released.wait(5)
    assert events.results.empty() and events.accepted.empty()


def test_evidence_without_request_cannot_use_legacy_direct_reader(tmp_path):
    _, descriptor = capture_audio(request(tmp_path))
    outcomes = []
    reader = ResultReader(descriptor, outcomes.append)
    reader.start()
    reader.thread.join(3)
    assert outcomes[0].audio is None and "request context" in outcomes[0].error


def test_digest_failure_retains_actual_successful_write_count(tmp_path):
    from base.recording_process_protocol import RecordingFailure

    class FailingDigest:
        def update(self, block):
            raise MemoryError("digest allocation failed")

    backend = FakeBackend()
    capture = RecordingCapture(request(tmp_path, trim_samples=4), backend=backend)
    capture._sample_digest = FailingDigest()
    capture.start()
    try:
        assert capture.started.wait(3)
        backend.stream.feed(known_audio(9))
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure)
        assert "digest allocation failed" in outcome.message
        assert outcome.written_frames == sf.info(outcome.path).frames == 5
    finally:
        capture.cancel()
        assert capture.join(3)


def test_parent_metadata_uncertain_release_keeps_reader_lease(tmp_path, monkeypatch):
    from base import recording_result_reader as module
    req, descriptor, _, _ = prepared_ve_result(tmp_path, {"enabled": False})
    diagnostic = module.inspect_wav_calibration_metadata(req.path)
    monkeypatch.setattr(module, "inspect_wav_calibration_metadata", lambda path: replace(
        diagnostic, handles_released=False, close_errors=("metadata close uncertain",)))
    outcome = read_result(descriptor, req)
    assert outcome.audio is None and "metadata close uncertain" in outcome.error
    assert not outcome.handles_released


def test_paced_process_consumes_fully_trimmed_blocks_without_waiting_for_write(tmp_path, services):
    from unit_test.base.recording_process_fakes import generated_audio
    service = services(dict(frames=9, chunk_frames=2, pace_writer=True))
    events = Events()
    session = service.start(request(tmp_path, trim_samples=4), events.callbacks)
    result = events.results.get(timeout=5)
    np.testing.assert_array_equal(result.multi, generated_audio(4, 5)[:, (0, 2)])
    session.accept_result()
    assert session.released.wait(5)
