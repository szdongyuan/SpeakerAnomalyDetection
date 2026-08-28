"""Importable audio boundaries; no hardware and no process-global test state."""
import os
import threading
import time
import json
from pathlib import Path

import numpy as np

from base.streaming_file_writer import StreamingWavWriter


def known_audio(frames=12, channels=3):
    return (np.arange(frames * channels, dtype=np.float32).reshape(frames, channels) - 8) / 32


def generated_audio(start, frames, channels=3):
    """Bounded deterministic signal generated per block for long fake captures."""
    indexes = np.arange(start * channels, (start + frames) * channels, dtype=np.int64)
    return ((indexes % 61 - 30).astype(np.float32) / 64).reshape(frames, channels)


def device_info():
    return dict(index=7, name="fake recording device", hostapi=0,
                max_input_channels=3, max_output_channels=2)


class FakeStatus:
    def __init__(self, *, input_overflow=False, output_underflow=False):
        self.input_overflow = input_overflow
        self.output_underflow = output_underflow

    def __bool__(self):
        return self.input_overflow or self.output_underflow

    def __str__(self):
        return "input overflow" if self.input_overflow else "output underflow"


class FakeStream:
    def __init__(self, backend, **config):
        self.backend = backend
        self.config = config
        self.closed = False
        self.active = False
        self.capture_pid = None

    def start(self):
        self.active = True

    def feed(self, data, status=None, *, mutate=False):
        self.capture_pid = os.getpid()
        borrowed = np.array(data, dtype=np.float32, copy=True)
        callback = self.config["callback"]
        if isinstance(self.config["channels"], tuple):
            out = np.full((len(data), self.config["channels"][1]), np.nan, dtype=np.float32)
            callback(borrowed, out, len(data), None, status or FakeStatus())
        else:
            out = None
            callback(borrowed, len(data), None, status or FakeStatus())
        if mutate:
            borrowed.fill(-999)
        return out

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True


class FakeBackend:
    def __init__(self):
        self.device = device_info()
        self.stream = None

    def query_devices(self, index):
        if index != self.device["index"]:
            raise ValueError("unknown fake device")
        return self.device.copy()

    def InputStream(self, **config):
        self.stream = FakeStream(self, **config)
        return self.stream

    Stream = InputStream


class ControlledWriter:
    """A real WAV writer with deterministic write/close failure and pause hooks."""
    def __init__(self, *, pause=False, fail_at=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        if not pause:
            self.release.set()
        self.fail_at = fail_at
        self.writer = None
        self.writer_pid = None
        self.closed = False

    def __call__(self, *args, **kwargs):
        self.writer = StreamingWavWriter(*args, **kwargs)
        return self

    def write_chunk(self, chunk):
        self.writer_pid = os.getpid()
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("test writer release was not signalled")
        if self.fail_at == "write":
            raise OSError("injected disk failure")
        self.writer.write_chunk(chunk)

    def finalize(self):
        self.writer.finalize()
        self.closed = True
        if self.fail_at == "close":
            raise OSError("injected close failure")


class MetadataFileFaults:
    """Exercise the actual metadata helper with real files and one fault boundary."""
    def __init__(self, target, *, close_fails=True):
        self.target = target
        self.close_fails = close_fails
        self.files = []
        self.temporary_paths = []

    def install(self, monkeypatch):
        from base import wav_calibration_metadata as module
        real_open = open
        real_temporary = module.tempfile.NamedTemporaryFile
        owner = self

        class FileBoundary:
            def __init__(self, wrapped, stage):
                self.wrapped = wrapped
                self.stage = stage
                self.close_attempts = 0
                owner.files.append(self)

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def read(self, *args, **kwargs):
                if self.stage == owner.target and not owner.close_fails:
                    raise OSError(f"injected metadata {self.stage} read failure")
                return self.wrapped.read(*args, **kwargs)

            def write(self, *args, **kwargs):
                if self.stage == owner.target and not owner.close_fails:
                    raise OSError("injected metadata temporary write failure")
                return self.wrapped.write(*args, **kwargs)

            def close(self):
                self.close_attempts += 1
                if self.stage == owner.target and owner.close_fails:
                    raise OSError(f"injected metadata {self.stage} close failure")
                self.wrapped.close()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def tracked_open(path, *args, **kwargs):
            stage = "validation" if os.fspath(path) in self.temporary_paths else "source"
            return FileBoundary(real_open(path, *args, **kwargs), stage)

        def tracked_temporary(*args, **kwargs):
            wrapped = real_temporary(*args, **kwargs)
            self.temporary_paths.append(wrapped.name)
            return FileBoundary(wrapped, "temporary")

        monkeypatch.setattr(module, "open", tracked_open, raising=False)
        monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", tracked_temporary)

    def release_all(self):
        for boundary in self.files:
            boundary.wrapped.close()


def process_dependencies(**options):
    """Importable injection for actual spawn; all options are plain scalar values."""
    if options.get("hang_ready"):
        threading.Event().wait()
    trace_dir = Path(options["trace_dir"])
    trace = {}
    lock = threading.Lock()
    writer_consumed = threading.Event()
    pace_writer = options.get("pace_writer", "frames" in options)

    def update(**changes):
        with lock:
            trace.update(changes)
            temporary = trace_dir / "trace.tmp"
            temporary.write_text(json.dumps(trace))
            deadline = time.monotonic() + 1
            while True:
                try:
                    os.replace(temporary, trace_dir / "trace.json")
                    break
                except PermissionError:
                    # Windows readers briefly deny replace while read_text holds
                    # a handle. Trace instrumentation must not break fake audio.
                    if time.monotonic() >= deadline:
                        raise
                    threading.Event().wait(.002)

    class ProcessStream(FakeStream):
        def start(self):
            self.backend.round += 1
            update(open_round=self.backend.round)
            if options.get("hang_start_round") == self.backend.round:
                threading.Event().wait()
            super().start()
            self.stop_feed = threading.Event()

            def feed():
                data = known_audio(110 if options.get("manual") else 12)
                if "frames" in options:
                    frames, block = options["frames"], options.get("chunk_frames", 4096)
                    chunks = (generated_audio(start, min(block, frames - start))
                              for start in range(0, frames, block))
                elif options.get("manual"):
                    data = (data % .5).astype(np.float32)
                    data[5, 0] = .95
                    chunks = (data[:3], data[3:7], data[7:])
                else:
                    chunks = (data[:2], data[2:5], data[5:])
                for index, chunk in enumerate(chunks):
                    if options.get("manual"):
                        while not (trace_dir / f"feed-{index}").exists():
                            if self.stop_feed.wait(.005):
                                return
                    if self.stop_feed.is_set():
                        return
                    writer_consumed.clear()
                    update(capture_pid=os.getpid())
                    self.feed(chunk, mutate=True)
                    update(fed_chunks=index + 1, producer_waiting_for_writer=pace_writer)
                    if pace_writer:
                        # Synthetic duration is unrelated to wall-clock speed.
                        # Acknowledge each write before producing another block,
                        # so host scheduling cannot masquerade as device overflow.
                        deadline = time.monotonic() + 10
                        while not writer_consumed.is_set():
                            if self.stop_feed.wait(.005):
                                return
                            if time.monotonic() >= deadline:
                                update(feeder_error="synthetic writer acknowledgement timed out")
                                return
                    else:
                        self.stop_feed.wait(.03)
            self.feeder = threading.Thread(target=feed, name="fake-audio", daemon=True)
            self.feeder.start()

        def stop(self):
            update(stop_entered=True)
            if options.get("hang_close"):
                threading.Event().wait()
            if hasattr(self, "stop_feed"):
                self.stop_feed.set()
                self.feeder.join(1)
            super().stop()

    class ProcessBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.round = 0

        def InputStream(self, **config):
            self.stream = ProcessStream(self, **config)
            return self.stream

        Stream = InputStream

    class ProcessWriter(StreamingWavWriter):
        def write_chunk(self, chunk):
            if options.get("fail_write"):
                raise OSError("injected synthetic disk write failure")
            if options.get("pause_first_write") and self.total_frames == 0:
                update(writer_entered=True)
                deadline = time.monotonic() + 10
                while not (trace_dir / "release-writer").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("test must release the paused synthetic writer")
                    threading.Event().wait(.005)
            super().write_chunk(chunk)
            self.sf_file.flush()
            update(writer_pid=os.getpid(), written_frames=self.total_frames)
            writer_consumed.set()

        def finalize(self):
            if not self._terminal_attempted:
                update(finalize_entered=True)
                if options.get("pause_finalize"):
                    update(finalize_waiting_for_release=True)
                    deadline = time.monotonic() + 10
                    while not (trace_dir / "release-finalize").exists():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("test must release synthetic finalization")
                        threading.Event().wait(.005)
                if options.get("hang_finalize"):
                    threading.Event().wait()
                if options.get("fail_close"):
                    self._terminal_attempted = True
                    raise OSError("injected writer close failure with retained handle")
            super().finalize()

    dependencies = dict(backend=ProcessBackend(), writer_factory=ProcessWriter)
    if options.get("metadata_retained") or options.get("metadata_leftover"):
        def metadata_retained(path, metadata, **kwargs):
            from base.wav_calibration_metadata import WavCalibrationMetadataAppendResult
            owned = str(Path(path).with_name("exact-owned-temporary.wav"))
            handle = open(owned, "wb")
            handle.write(b"owned metadata temporary")
            handle.flush()
            if options.get("metadata_leftover"):
                handle.close()
                return WavCalibrationMetadataAppendResult(
                    appended=False, handles_released=True, cleanup_paths=(owned,))
            return WavCalibrationMetadataAppendResult(
                appended=False, handles_released=False, cleanup_paths=(owned,),
                close_errors=("injected metadata close failure",), retained_handles=((owned, handle),))
        dependencies["metadata_appender"] = metadata_retained
    return dependencies


def orphan_service_parent(connection, options):
    """An intermediate parent intentionally terminated by the liveness test."""
    from base.recording_service import RecordingCallbacks, RecordingService
    from base.recording_process_protocol import RecordingRequest
    service = RecordingService(
        backend_factory="unit_test.base.recording_process_fakes:process_dependencies",
        backend_options=options, cancel_timeout=.25, terminate_timeout=.2)
    started = threading.Event()
    request = RecordingRequest("orphan", "main", 100, 100, (0, 2), device_info(),
                               str(Path(options["trace_dir"]) / "orphan.wav"), True, 0, {},
                               None, {"enabled": False})
    service.start(request, RecordingCallbacks(started=lambda session: started.set()))
    if not started.wait(10):
        raise RuntimeError("intermediate parent's recording did not start")
    connection.send(service.worker_pid)
    threading.Event().wait()


def open_process_observer(pid):
    """Windows process handle, kept open so PID reuse cannot confuse assertions."""
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x00100000 | 0x0001, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    return kernel, handle
