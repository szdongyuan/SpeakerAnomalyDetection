"""Exercise grouped capture against real PCM24 files and controlled writer faults."""
from contextlib import contextmanager
from dataclasses import replace
import ctypes
import logging
import os
import struct
import sys
from pathlib import Path
import threading

import numpy as np
import pytest
import soundfile as sf

from base.log_manager import LogManager
from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingCancelled, RecordingFailure, RecordingRequest, RecordingResult
from base.streaming_file_writer import StreamingWavWriter
from base.wav_calibration_metadata import normalize_wav_calibration_metadata, read_wav_calibration_metadata
from base.wav_pcm24 import quantize_pcm24
from base.workbench_recording_protocol import StationRecordingGroup, StationRecordingPlan, StationWav, StationWavChannel
from unit_test.base.recording_process_fakes import ControlledWriter, FakeBackend, device_info
from unit_test.base.test_recording_queue_buffer import ManualVeStream
from unit_test.base.vkinging_fakes import device_info as ve_device_info, input_config, wav_metadata


def request(tmp_path, backend="soundcard", *, filenames=("A.wav", "B.wav"), thresholds=None):
    is_ve = backend == "ve"
    sample_rate = 51200 if is_ve else 100
    channels = (7, 1, 4) if is_ve else (0, 1, 2)
    device = (ve_device_info(physical_channels=channels, input_config=input_config(sample_rate))
              if is_ve else device_info())
    stations = []
    for station_id, filename, physical in zip(
        ("A", "B"), filenames, ((channels[2], channels[0]), (channels[0],)),
    ):
        mapping = tuple(
            StationWavChannel(f"ch-{index}", f"input {index}", "sound", "device", index, position)
            for position, index in enumerate(physical)
        )
        if is_ve:
            metadata = wav_metadata(("none",) * len(physical), sample_rate, physical)
            metadata["acquisition"]["machine_id"] = device["machine_id"]
        else:
            metadata = {"recorded_channels": [
                {"wav_channel_index": position, "physical_input_channel": index, "calibrated": False}
                for position, index in enumerate(physical)
            ]}
        stations.append(StationRecordingPlan(station_id, filename, mapping, metadata))
    stage = tmp_path / "stage"
    stage.mkdir()
    return RecordingRequest(
        request_id="group", purpose="main", sample_rate=sample_rate, target_samples=9,
        channels=channels, device=device, path=str(stage), streaming=True, trim_samples=3,
        monitor={}, calibration_metadata=None,
        validation_thresholds=thresholds if thresholds is not None else {"enabled": False},
        station_group=StationRecordingGroup("product", "condition", str(tmp_path / "final"), stations),
    )


def audio(req):
    width = max(req.channels) + 1
    return np.arange(12 * width, dtype=np.float32).reshape(12, width) / 111 + .000001


@contextmanager
def running_capture(req, **options):
    backend = FakeBackend()
    capture = RecordingCapture(req, backend=backend, ve_stream_factory=ManualVeStream, **options)
    capture.start()
    try:
        yield capture, backend
    finally:
        if not capture.done.is_set():
            capture.cancel()
        assert capture.join(5)


@pytest.mark.parametrize("backend_name", ["soundcard", "ve"])
def test_group_captures_all_channels_and_writes_trimmed_pcm24_in_station_order(tmp_path, backend_name):
    req = request(tmp_path, backend_name)
    calls = []

    class Writer(StreamingWavWriter):
        def write_chunk(self, chunk):
            calls.append((Path(self.file_path).name, len(chunk)))
            return super().write_chunk(chunk)

    data = audio(req)
    with running_capture(req, writer_factory=Writer) as (capture, backend):
        assert capture.started.wait(3), capture.outcome
        if backend_name == "ve":
            stream = capture._native_stream
            assert stream.request.channels == (7, 1, 4)
        else:
            stream = backend.stream
            assert stream.config["channels"] == len(req.channels) == 3
        stream.feed(data[:2])
        stream.feed(data[2:6])
        assert capture.snapshot(generation=1, sequence=1) is None
        stream.feed(data[6:9] if backend_name == "ve" else data[6:])
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingResult), outcome
        assert capture.consumed_frames == outcome.raw_frames == 9
        assert capture.written_frames == outcome.final_frames == 6
        assert capture.snapshot(generation=1, sequence=2) is None
        assert capture.capture_slot_released.is_set() is (backend_name == "ve")
        assert stream.closed

    assert outcome.path == req.path and outcome.handles_released and outcome.metadata_appended
    assert outcome.final_envelope is None and outcome.sample_digest is None
    assert calls == [("A.wav", 3), ("B.wav", 3), ("A.wav", 3), ("B.wav", 3)]
    assert outcome.station_files == tuple(
        StationWav(plan.station_id, str(Path(req.path) / plan.filename), plan.channels)
        for plan in req.station_group.stations
    )
    assert {path.name for path in Path(req.path).iterdir()} == {"A.wav", "B.wav"}
    assert not Path(req.station_group.final_path).exists()
    saved = []
    for descriptor, plan in zip(outcome.station_files, req.station_group.stations):
        info = sf.info(descriptor.path)
        assert (info.frames, info.channels, info.samplerate, info.subtype) == (
            6, len(plan.channels), req.sample_rate, "PCM_24")
        samples, rate = sf.read(descriptor.path, dtype="float32", always_2d=True)
        assert rate == req.sample_rate
        np.testing.assert_array_equal(samples, quantize_pcm24(data[3:9, plan.physical_channels]))
        assert read_wav_calibration_metadata(descriptor.path) == normalize_wav_calibration_metadata(
            plan.calibration_metadata.to_dict())
        saved.append(samples)
    np.testing.assert_array_equal(saved[0][:, 1], saved[1][:, 0])


@pytest.mark.parametrize("failure,stage,released,frames", [
    ("open", "open_wav", True, 0),
    ("write", "write", True, 3),
    ("close", "close_wav", False, 6),
    ("close_and_log", "close_wav", False, 6),
])
def test_group_writer_failure_closes_every_open_writer(tmp_path, failure, stage, released, frames):
    req = request(tmp_path, "ve")
    opened, closed = [], []

    class Writer(StreamingWavWriter):
        def __init__(self, path, **kwargs):
            name = Path(path).name
            if failure == "open" and name == "B.wav":
                raise OSError("injected open fault")
            super().__init__(path, **kwargs)
            opened.append(name)
            self.write_calls = 0

        def write_chunk(self, chunk):
            self.write_calls += 1
            if failure == "write" and Path(self.file_path).name == "B.wav" and self.write_calls == 2:
                raise OSError("injected write fault")
            return super().write_chunk(chunk)

        def finalize(self):
            super().finalize()
            name = Path(self.file_path).name
            closed.append(name)
            if failure in ("close", "close_and_log") and name == "A.wav":
                raise OSError("injected close fault")

    class FailingDiagnostic(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                raise OSError("injected diagnostic fault")

    with running_capture(req, writer_factory=Writer) as (capture, _):
        if failure != "open":
            assert capture.started.wait(3), capture.outcome
            if failure == "close_and_log":
                logger = logging.Logger("station-close-diagnostic")
                logger.addHandler(FailingDiagnostic())
                capture._logger = logger
            data = audio(req)
            capture._native_stream.feed(data[:6])
            capture._native_stream.feed(data[6:9])
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure), outcome
        assert (outcome.stage, outcome.handles_released, outcome.written_frames) == (stage, released, frames)
        assert f"injected {failure.removesuffix('_and_log')} fault" in outcome.message
        assert ("'A'" if stage == "close_wav" else "'B'") in outcome.message
        if failure != "write":
            assert not capture.capture_slot_released.is_set()
    assert opened == closed == (["A.wav"] if failure == "open" else ["A.wav", "B.wav"])
    if failure == "write":
        assert sf.info(Path(req.path) / "A.wav").frames == 6
        assert sf.info(Path(req.path) / "B.wav").frames == 3
    assert not Path(req.station_group.final_path).exists()


@pytest.mark.parametrize("alias_type", ["hardlink", "short_name"])
def test_existing_alias_is_rejected_before_second_writer_can_truncate_first(tmp_path, alias_type):
    long_name = "Station Alpha Recording.wav"
    alias_name = "alias.wav"
    if alias_type == "short_name":
        if os.name != "nt":
            pytest.skip("Windows short path aliases")
        probe = tmp_path / long_name
        probe.touch()
        get_short_path = ctypes.windll.kernel32.GetShortPathNameW
        short_path = ctypes.create_unicode_buffer(32768)
        assert 0 < get_short_path(str(probe), short_path, len(short_path)) < len(short_path)
        alias_name = Path(short_path.value).name
        if alias_name.casefold() == long_name.casefold():
            pytest.skip("8.3 names are disabled on the test volume")

    req = request(tmp_path, filenames=(long_name, alias_name))
    first, alias = (Path(req.path) / plan.filename for plan in req.station_group.stations)
    samples = audio(req)[3:9, (2, 0)]
    opened = []

    def factory(path, **options):
        opened.append(Path(path))
        writer = StreamingWavWriter(path, **options)
        if Path(path) == first:
            writer.write_chunk(samples)
            if alias_type == "hardlink":
                alias.hardlink_to(first)
            assert first.samefile(alias)
        return writer

    with running_capture(req, writer_factory=factory) as (capture, _):
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure), outcome
        assert outcome.stage == "open_wav" and outcome.handles_released
        assert "already exists" in outcome.message and "'B'" in outcome.message
    assert opened == [first]
    assert first.read_bytes() == alias.read_bytes()
    saved, _ = sf.read(first, dtype="float32", always_2d=True)
    np.testing.assert_array_equal(saved, quantize_pcm24(samples))


def test_default_group_rejects_hardlink_created_after_target_check(tmp_path, monkeypatch):
    req = request(tmp_path)
    target = Path(req.path) / "B.wav"
    victim = tmp_path / "outside.wav"
    original = b"existing file outside staging"
    victim.write_bytes(original)
    real_lstat = os.lstat
    injected = []

    def lstat(path, *args, **kwargs):
        try:
            return real_lstat(path, *args, **kwargs)
        except FileNotFoundError:
            if os.fspath(path) == str(target):
                target.hardlink_to(victim)
                injected.append(target)
            raise

    monkeypatch.setattr(os, "lstat", lstat)
    with running_capture(req) as (capture, backend):
        if not capture.done.wait(1):
            # Force the vulnerable writer to finish and flush any truncation.
            assert capture.started.is_set()
            backend.stream.feed(audio(req)[:req.target_samples])
        outcome = capture.wait(3)
        assert injected == [target] and target.samefile(victim)
        assert victim.read_bytes() == original
        assert isinstance(outcome, RecordingFailure), outcome
        assert outcome.stage == "open_wav" and outcome.path == req.path and outcome.handles_released
        assert "station 'B'" in outcome.message
        assert backend.stream is None


@pytest.mark.parametrize("fallback", [False, True], ids=["soundfile", "wave"])
def test_default_group_writes_and_closes_each_exclusive_stream_once(tmp_path, monkeypatch, fallback):
    import base.streaming_file_writer as module

    req = replace(request(tmp_path), target_samples=8)
    data = audio(req)
    opens, backend_inputs = [], []
    real_open = open
    native_open = module.wave.open if fallback else sf.SoundFile

    def track_open(file, mode):
        handle = real_open(file, mode)
        opens.append((Path(file).name, mode, handle))
        return handle

    def track_native_open(file, *args, **kwargs):
        if fallback or kwargs.get("mode") == "w":
            backend_inputs.append(file if fallback else file.stream)
        return native_open(file, *args, **kwargs)

    monkeypatch.setattr(module, "open", track_open, raising=False)
    if fallback:
        monkeypatch.setitem(sys.modules, "soundfile", None)
        monkeypatch.setattr(module.wave, "open", track_native_open)
    else:
        monkeypatch.setattr(sf, "SoundFile", track_native_open)
    with running_capture(req) as (capture, backend):
        assert capture.started.wait(3), capture.outcome
        backend.stream.feed(data[:req.target_samples])
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingResult) and outcome.handles_released, outcome

    assert [(name, mode) for name, mode, _ in opens] == [("A.wav", "x+b"), ("B.wav", "x+b")]
    assert backend_inputs == [handle for _, _, handle in opens]
    assert all(handle.closed for handle in backend_inputs)
    second = Path(outcome.station_files[1].path)
    saved, rate = sf.read(second, dtype="float32", always_2d=True)
    assert rate == req.sample_rate
    np.testing.assert_array_equal(saved, quantize_pcm24(data[3:8, (0,)]))
    raw = second.read_bytes()
    assert struct.unpack_from("<I", raw, 4)[0] + 8 == len(raw)
    data_offset = raw.index(b"data") + 8
    assert struct.unpack_from("<I", raw, data_offset - 4)[0] == 15
    assert raw[data_offset + 15] == 0


@pytest.mark.parametrize("fault,stage,reason", [
    ("open_seek", "open_wav", "injected virtual seek failure"),
    ("short_write", "write", "short WAV stream write"),
    ("pad_write", "close_wav", "injected virtual padding failure"),
])
def test_group_reports_underlying_soundfile_virtual_io_fault(tmp_path, monkeypatch, fault, stage, reason):
    import base.streaming_file_writer as module

    req = replace(request(tmp_path), target_samples=8)
    raw_handles, sound_files, injected = [], [], []
    raw_open, soundfile_open = open, sf.SoundFile

    def open_raw(path, mode):
        handle = raw_open(path, mode)
        raw_handles.append(handle)
        if Path(path).name == "B.wav":
            original_seek, original_write = handle.seek, handle.write

            def seek(offset, whence=0):
                if fault == "open_seek" and not injected:
                    injected.append(fault)
                    raise OSError(reason)
                return original_seek(offset, whence)

            def write(data):
                if fault == "short_write" and len(data) == 15 and not injected:
                    injected.append(fault)
                    return original_write(data[:-3])
                if fault == "pad_write" and data == b"\x00" and not injected:
                    injected.append(fault)
                    raise OSError(reason)
                return original_write(data)

            monkeypatch.setattr(handle, "seek", seek)
            monkeypatch.setattr(handle, "write", write)
        return handle

    def open_sound_file(*args, **kwargs):
        sound_file = soundfile_open(*args, **kwargs)
        if kwargs.get("mode") == "w":
            sound_files.append(sound_file)
        return sound_file

    monkeypatch.setattr(module, "open", open_raw, raising=False)
    monkeypatch.setattr(sf, "SoundFile", open_sound_file)
    with running_capture(req) as (capture, backend):
        if fault != "open_seek":
            assert capture.started.wait(3), capture.outcome
            backend.stream.feed(audio(req)[:req.target_samples])
        elif not capture.done.wait(1):
            assert capture.started.is_set()
            backend.stream.feed(audio(req)[:req.target_samples])
        outcome = capture.wait(3)
        assert injected == [fault]
        assert isinstance(outcome, RecordingFailure), outcome
        assert outcome.stage == stage
        assert "station 'B'" in outcome.message and reason in outcome.message
    assert len(raw_handles) == 2 and all(handle.closed for handle in raw_handles)
    assert sound_files and all(handle.closed for handle in sound_files)


def test_group_metadata_append_checks_identity_of_open_file(tmp_path, monkeypatch):
    import base.wav_calibration_metadata as metadata_module

    req = request(tmp_path)
    target = Path(req.path) / "B.wav"
    victim = tmp_path / "outside.wav"
    sf.write(victim, np.zeros(6, dtype=np.float32), req.sample_rate, subtype="PCM_24")
    original = victim.read_bytes()
    raw_open = open
    substituted = []

    def open_for_metadata(path, mode):
        if Path(path) == target and mode == "r+b":
            target.unlink()
            target.hardlink_to(victim)
            handle = raw_open(path, mode)
            substituted.append(handle)
            return handle
        return raw_open(path, mode)

    monkeypatch.setattr(metadata_module, "open", open_for_metadata, raising=False)
    with running_capture(req) as (capture, backend):
        assert capture.started.wait(3), capture.outcome
        backend.stream.feed(audio(req)[:req.target_samples])
        outcome = capture.wait(3)
    assert len(substituted) == 1 and substituted[0].closed
    assert target.samefile(victim) and victim.read_bytes() == original
    assert isinstance(outcome, RecordingFailure), outcome
    assert outcome.stage == "metadata" and outcome.handles_released
    assert "station 'B'" in outcome.message and "identity changed" in outcome.message


@pytest.mark.parametrize("substitute", [False, True], ids=["owned", "substituted"])
def test_group_optional_metadata_open_denial_requires_owned_wav(tmp_path, monkeypatch, substitute):
    import base.wav_calibration_metadata as metadata_module

    req = request(tmp_path)
    target = Path(req.path) / "B.wav"
    victim = tmp_path / "outside.wav"
    sf.write(victim, np.zeros(6, dtype=np.float32), req.sample_rate, subtype="PCM_24")
    original = victim.read_bytes()
    raw_open = open
    denied = []

    def open_for_metadata(path, mode):
        if Path(path) == target and mode == "r+b":
            denied.append(target)
            if substitute:
                target.unlink()
                target.hardlink_to(victim)
            raise PermissionError("injected read-only metadata target")
        return raw_open(path, mode)

    monkeypatch.setattr(metadata_module, "open", open_for_metadata, raising=False)
    with running_capture(req) as (capture, backend):
        assert capture.started.wait(3), capture.outcome
        backend.stream.feed(audio(req)[:req.target_samples])
        outcome = capture.wait(3)
    assert denied == [target] and victim.read_bytes() == original
    assert target.samefile(victim) is substitute
    if substitute:
        assert isinstance(outcome, RecordingFailure), outcome
        assert outcome.stage == "metadata" and outcome.handles_released
        assert "identity changed" in outcome.message and "B.wav" in outcome.message
    else:
        assert isinstance(outcome, RecordingResult) and outcome.handles_released
        assert not outcome.metadata_appended
        assert any("station 'B': WAV calibration metadata was not appended" in warning
                   for warning in outcome.warnings)
        saved, _ = sf.read(target, dtype="float32", always_2d=True)
        np.testing.assert_array_equal(saved, quantize_pcm24(audio(req)[3:9, (0,)]))


def test_group_metadata_readback_rejects_substituted_wav(tmp_path, monkeypatch):
    import base.wav_calibration_metadata as metadata_module

    req = request(tmp_path)
    target = Path(req.path) / "B.wav"
    victim = tmp_path / "outside.wav"
    sf.write(victim, np.zeros(6, dtype=np.float32), req.sample_rate, subtype="PCM_24")
    metadata = req.station_group.stations[1].calibration_metadata.to_dict()
    assert metadata_module.append_owned_recording_calibration_metadata_result(victim, metadata).appended
    original = victim.read_bytes()
    raw_open = open
    substituted = []

    def open_for_metadata(path, mode):
        if Path(path) == target and mode == "rb":
            target.unlink()
            target.hardlink_to(victim)
            handle = raw_open(path, mode)
            substituted.append(handle)
            return handle
        return raw_open(path, mode)

    monkeypatch.setattr(metadata_module, "open", open_for_metadata, raising=False)
    with running_capture(req) as (capture, backend):
        assert capture.started.wait(3), capture.outcome
        backend.stream.feed(audio(req)[:req.target_samples])
        outcome = capture.wait(3)
    assert len(substituted) == 1 and substituted[0].closed
    assert target.samefile(victim) and victim.read_bytes() == original
    assert isinstance(outcome, RecordingFailure), outcome
    assert outcome.stage == "metadata" and outcome.handles_released
    assert "station 'B'" in outcome.message and "identity changed" in outcome.message


def test_group_readback_retains_both_handles_when_closes_fail(tmp_path, monkeypatch):
    import base.recording_capture as module

    req = request(tmp_path)
    handles, attempted = [], []
    raw_open, soundfile_open = open, sf.SoundFile

    def fail_close(handle, label):
        handles.append((handle, handle.close))

        def close():
            attempted.append(label)
            raise OSError(f"injected {label} close fault")

        monkeypatch.setattr(handle, "close", close)
        return handle

    def open_raw(path, mode):
        assert mode == "rb"
        return fail_close(raw_open(path, mode), "raw reader")

    def open_sound_file(path, *args, **kwargs):
        handle = soundfile_open(path, *args, **kwargs)
        return handle if kwargs.get("mode") == "w" else fail_close(handle, "native reader")

    monkeypatch.setattr(module, "open", open_raw, raising=False)
    monkeypatch.setattr(sf, "SoundFile", open_sound_file)
    try:
        with running_capture(req) as (capture, backend):
            assert capture.started.wait(3), capture.outcome
            backend.stream.feed(audio(req)[:req.target_samples])
            outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure) and outcome.stage == "read_wav"
        assert not outcome.handles_released
        assert "injected native reader close fault" in outcome.message
        assert attempted == ["native reader", "raw reader"]
        path = str(Path(req.path) / "A.wav")
        assert capture._unreleased_finalization_handles == [(path, handle) for handle, _ in reversed(handles)]
    finally:
        for _, close in reversed(handles):
            close()


@pytest.mark.parametrize("diagnostic_fails", [False, True])
def test_one_station_quality_failure_rejects_group(tmp_path, monkeypatch, diagnostic_fails):
    if diagnostic_fails:
        class FailingDiagnostic(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    raise OSError("injected quality diagnostic fault")

        logger = logging.Logger("station-quality-diagnostic", logging.INFO)
        logger.addHandler(FailingDiagnostic())
        monkeypatch.setattr(LogManager, "set_log_handler", lambda _name: logger)

    req = request(tmp_path, thresholds={"enabled": True, "min_peak": .1})
    data = audio(req)
    data[:, 0] = 0
    with running_capture(req) as (capture, backend):
        assert capture.started.wait(3), capture.outcome
        backend.stream.feed(data[:6])
        backend.stream.feed(data[6:])
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure), outcome
        assert outcome.stage == "quality" and outcome.handles_released
        assert outcome.message.startswith("station 'B':") and "weak_signal" in outcome.message
        if diagnostic_fails:
            assert "injected quality diagnostic fault" in outcome.message
        assert outcome.written_frames == 6
    assert not Path(req.station_group.final_path).exists()


def test_cancel_during_second_station_write_closes_group_without_completion(tmp_path):
    req = request(tmp_path)
    writers = [ControlledWriter(), ControlledWriter(pause=True)]
    opened = iter(writers)
    with running_capture(req, writer_factory=lambda *args, **kwargs: next(opened)(*args, **kwargs)) as (
        capture, backend,
    ):
        try:
            assert capture.started.wait(3), capture.outcome
            backend.stream.feed(audio(req)[:6])
            assert writers[1].entered.wait(3)
            assert capture.written_frames == 0
            capture.cancel()
        finally:
            writers[1].release.set()
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingCancelled), outcome
        assert (outcome.raw_frames, outcome.final_frames, outcome.handles_released) == (6, 3, True)
        assert all(writer.closed for writer in writers)
        assert backend.stream.closed
    assert not Path(req.station_group.final_path).exists()


@pytest.mark.parametrize("grouped", [False, True], ids=["single", "group"])
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_initialization_log_failure_accounts_for_acquired_file(tmp_path, monkeypatch, grouped, rollback_fails):
    req = request(tmp_path)
    if not grouped:
        req = replace(req, station_group=None, path=str(tmp_path / "B.wav"))
    native_files = {}
    sound_files = []
    rollback_attempts = []
    real_open = sf.SoundFile

    def open_sound_file(target, **kwargs):
        sound_file = real_open(target, **kwargs)
        sound_files.append(sound_file)
        handle = target.stream if grouped else sound_file
        name = Path(handle.name if grouped else target).name
        native_files[name] = (handle, handle.close)
        if name == "B.wav" and rollback_fails:
            def fail_close():
                rollback_attempts.append(handle)
                raise OSError("injected initialization rollback fault")

            monkeypatch.setattr(handle, "close", fail_close)
        return sound_file

    class FailingDiagnostic(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith("StreamingWavWriter initialized") and message.endswith("B.wav"):
                raise OSError("injected initialization log fault")
            if message.startswith("Capture ") and record.levelno >= logging.ERROR:
                raise OSError("injected capture diagnostic fault")

    logger = logging.Logger("initialization-fault", logging.INFO)
    logger.addHandler(FailingDiagnostic())
    monkeypatch.setattr(sf, "SoundFile", open_sound_file)
    monkeypatch.setattr(LogManager, "set_log_handler", lambda _name: logger)

    try:
        with running_capture(req) as (capture, backend):
            outcome = capture.wait(3)
            assert isinstance(outcome, RecordingFailure), outcome
            assert outcome.stage == "open_wav"
            assert outcome.handles_released is not rollback_fails
            assert "injected initialization log fault" in outcome.message
            assert "injected capture diagnostic fault" in outcome.message
            if grouped:
                assert "'B'" in outcome.message
                assert native_files["A.wav"][0].closed
            assert backend.stream is None and not capture.started.is_set()

        handle, _ = native_files["B.wav"]
        if grouped:
            assert len(sound_files) == 2 and all(sound_file.closed for sound_file in sound_files)
        if rollback_fails:
            assert not handle.closed and rollback_attempts == [handle]
            assert (str(Path(req.path) / "B.wav") if grouped else req.path, handle) in (
                capture._unreleased_finalization_handles
            )
            assert "injected initialization rollback fault" in outcome.message
        else:
            assert handle.closed
            assert not capture._unreleased_finalization_handles
    finally:
        for _, close in native_files.values():
            close()


def test_failing_diagnostic_deferral_cannot_prevent_station_closes(tmp_path):
    req = request(tmp_path, "ve")
    writers = []

    class Writer(StreamingWavWriter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_attempts = 0
            writers.append(self)

        def defer_finalization_log(self):
            if Path(self.file_path).name == "A.wav":
                raise OSError("injected diagnostic deferral fault")
            super().defer_finalization_log()

        def finalize(self):
            self.close_attempts += 1
            super().finalize()

    try:
        with running_capture(req, writer_factory=Writer) as (capture, _):
            assert capture.started.wait(3), capture.outcome
            capture._native_stream.feed(audio(req)[:req.target_samples])
            outcome = capture.wait(3)
            assert isinstance(outcome, RecordingResult) and outcome.handles_released, outcome
            assert capture.capture_slot_released.is_set()
        assert len(writers) == 2
        assert all(writer.close_attempts == 1 and writer.sf_file.closed for writer in writers)
    finally:
        for writer in writers:
            writer.sf_file.close()


def test_cancel_between_station_opens_closes_first_without_binding_device(tmp_path):
    req = request(tmp_path, "ve")
    first_open, proceed = threading.Event(), threading.Event()
    writers = []

    def factory(*args, **kwargs):
        writer = StreamingWavWriter(*args, **kwargs)
        writers.append(writer)
        if len(writers) == 1:
            first_open.set()
            assert proceed.wait(3)
        return writer

    with running_capture(req, writer_factory=factory) as (capture, _):
        try:
            assert first_open.wait(3)
            capture.cancel()
        finally:
            proceed.set()
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingCancelled) and outcome.handles_released, outcome
        assert capture._native_stream is None
        assert not capture.started.is_set()
        assert not capture.capture_slot_released.is_set()
    writer, = writers
    assert Path(writer.file_path).name == "A.wav" and writer.sf_file.closed
    assert not (Path(req.path) / "B.wav").exists()

def test_failed_native_close_finishes_other_station_before_logging(tmp_path, monkeypatch):
    req = request(tmp_path, "ve")
    writers = []
    logging_entered, release_log = threading.Event(), threading.Event()

    class DiagnosticGate(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                logging_entered.set()
                assert release_log.wait(5)

    logger = logging.Logger("station-close-gate", logging.INFO)
    logger.addHandler(DiagnosticGate())
    monkeypatch.setattr(LogManager, "set_log_handler", lambda _name: logger)

    def factory(*args, **kwargs):
        writer = StreamingWavWriter(*args, **kwargs)
        writers.append((writer, writer.sf_file.close))
        if Path(writer.file_path).name == "A.wav":
            def fail_close():
                raise OSError("injected native close fault")

            monkeypatch.setattr(writer.sf_file, "close", fail_close)
        return writer

    try:
        with running_capture(req, writer_factory=factory) as (capture, _):
            try:
                assert capture.started.wait(3), capture.outcome
                capture._native_stream.feed(audio(req)[:req.target_samples])
                assert logging_entered.wait(3)
                assert len(writers) == 2
                assert not writers[0][0].sf_file.closed
                assert writers[1][0].sf_file.closed
            finally:
                release_log.set()
            outcome = capture.wait(3)
            assert isinstance(outcome, RecordingFailure)
            assert outcome.stage == "close_wav" and not outcome.handles_released
            assert "injected native close fault" in outcome.message
    finally:
        for _, close in writers:
            close()

def test_native_stop_failure_closes_stream_and_station_files_before_diagnosing(tmp_path, monkeypatch):
    req = request(tmp_path)
    writers = []
    logging_entered, release_log = threading.Event(), threading.Event()

    class DiagnosticGate(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                logging_entered.set()
                assert release_log.wait(5)
                raise OSError("injected stream cleanup diagnostic fault")

    logger = logging.Logger("stream-close-gate", logging.INFO)
    logger.addHandler(DiagnosticGate())
    monkeypatch.setattr(LogManager, "set_log_handler", lambda _name: logger)

    def factory(*args, **kwargs):
        writer = StreamingWavWriter(*args, **kwargs)
        writers.append(writer)
        return writer

    def fail_stop():
        raise OSError("injected native stop fault")

    with running_capture(req, writer_factory=factory) as (capture, backend):
        try:
            assert capture.started.wait(3), capture.outcome
            monkeypatch.setattr(backend.stream, "stop", fail_stop)
            backend.stream.feed(audio(req)[:req.target_samples])
            assert logging_entered.wait(3)
            assert backend.stream.closed
            assert len(writers) == 2 and all(writer.sf_file.closed for writer in writers)
        finally:
            release_log.set()
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure), outcome
        assert outcome.stage == "close_stream" and not outcome.handles_released
        assert "injected native stop fault" in outcome.message
        assert "injected stream cleanup diagnostic fault" in outcome.message


def test_failed_native_write_closes_station_files_before_diagnosing(tmp_path, monkeypatch):
    req = request(tmp_path)
    writers = []
    logging_entered, release_log = threading.Event(), threading.Event()

    class DiagnosticGate(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                logging_entered.set()
                assert release_log.wait(5)
                raise OSError("injected write diagnostic fault")

    logger = logging.Logger("station-write-gate", logging.INFO)
    logger.addHandler(DiagnosticGate())
    monkeypatch.setattr(LogManager, "set_log_handler", lambda _name: logger)

    def factory(*args, **kwargs):
        writer = StreamingWavWriter(*args, **kwargs)
        writers.append(writer)
        if Path(writer.file_path).name == "B.wav":
            def fail_write(_samples):
                raise OSError("injected native WAV write fault")

            monkeypatch.setattr(writer.sf_file, "write", fail_write)
        return writer

    with running_capture(req, writer_factory=factory) as (capture, backend):
        try:
            assert capture.started.wait(3), capture.outcome
            backend.stream.feed(audio(req)[:req.target_samples])
            assert logging_entered.wait(3)
            closed_before_diagnostic = len(writers) == 2 and all(writer.sf_file.closed for writer in writers)
        finally:
            release_log.set()
        outcome = capture.wait(3)
        assert isinstance(outcome, RecordingFailure), outcome
        assert (outcome.stage, outcome.handles_released, outcome.written_frames) == ("write", True, 0)
        assert closed_before_diagnostic, outcome.message
        assert outcome.message.startswith("station 'B': injected native WAV write fault")
        assert "injected write diagnostic fault" in outcome.message
        assert backend.stream.closed
