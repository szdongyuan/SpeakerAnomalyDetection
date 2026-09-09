"""Importable audio boundaries; no hardware and no process-global test state."""
import os
import threading
import time
import json
from pathlib import Path

import numpy as np

from base.streaming_file_writer import StreamingWavWriter


_REAL_EVENT = threading.Event
_REAL_LOCK = threading.Lock
_REAL_THREAD = threading.Thread
_REAL_MONOTONIC = time.monotonic
_TRACE_COALESCE_SECONDS = .1


class _TracePublicationCancelled(Exception):
    """Internal cooperative stop for the supported fake trace boundary."""


def _check_trace_publication_cancelled(cancelled):
    if cancelled.is_set():
        raise _TracePublicationCancelled("fake trace publication was cancelled")


def _atomic_publish_trace(trace_dir, trace, cancelled):
    """Publish one complete fake trace snapshot despite transient Windows readers."""
    temporary = trace_dir / "trace.tmp"
    _check_trace_publication_cancelled(cancelled)
    try:
        temporary.write_text(json.dumps(trace))
        _check_trace_publication_cancelled(cancelled)
        deadline = _REAL_MONOTONIC() + 1
        while True:
            _check_trace_publication_cancelled(cancelled)
            try:
                os.replace(temporary, trace_dir / "trace.json")
                _check_trace_publication_cancelled(cancelled)
                return
            except PermissionError:
                _check_trace_publication_cancelled(cancelled)
                if _REAL_MONOTONIC() >= deadline:
                    raise
                cancelled.wait(.002)
    finally:
        # Cancellation and failed Windows replacement must not leave a stale
        # temporary snapshot that a later publisher could mistake for its own.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class _CoalescedTracePublisher:
    """Keep fake trace disk I/O off simulated audio producer/writer threads."""

    def __init__(self, trace_dir):
        self._trace_dir = trace_dir
        self._trace = {}
        self._lock = _REAL_LOCK()
        self._wake = _REAL_EVENT()
        self._published = _REAL_EVENT()
        self._cancel = _REAL_EVENT()
        self._revision = 0
        self._published_revision = 0
        self._flush_revision = 0
        self._closed = False
        self._error = None
        self._thread = _REAL_THREAD(
            target=self._run, name="fake-trace-publisher", daemon=True)
        self._thread.start()

    def update(self, **changes):
        with self._lock:
            if self._closed:
                raise RuntimeError("fake trace publisher is closed")
            if self._error is not None:
                raise self._error
            self._trace.update(changes)
            needs_wake = self._revision == self._published_revision
            self._revision += 1
        if needs_wake:
            self._wake.set()

    def flush(self, timeout=3):
        with self._lock:
            target = self._revision
            self._flush_revision = max(self._flush_revision, target)
        deadline = _REAL_MONOTONIC() + timeout
        while True:
            with self._lock:
                if self._error is not None:
                    raise self._error
                if self._published_revision >= target:
                    return
            remaining = deadline - _REAL_MONOTONIC()
            if remaining <= 0:
                raise TimeoutError("fake trace publication did not flush")
            self._wake.set()
            self._published.wait(min(.05, remaining))
            self._published.clear()

    def close(self, timeout=3):
        timeout = max(0.0, float(timeout))
        deadline = _REAL_MONOTONIC() + timeout
        with self._lock:
            already_closed = self._closed
            self._closed = True
        self._wake.set()
        if already_closed:
            self._cancel.set()
            self._published.set()
            self._thread.join(max(0.0, deadline - _REAL_MONOTONIC()))
            if self._thread.is_alive():
                raise TimeoutError("fake trace publisher did not stop")
            return
        flush_error = None
        try:
            # Keep half of the one close budget available for cooperative
            # cancellation and joining if final publication cannot progress.
            remaining = max(0.0, deadline - _REAL_MONOTONIC())
            self.flush(remaining / 2)
        except Exception as exc:
            # Publication is diagnostic test I/O. Closing remains terminal,
            # but the first caller still receives the exact cleanup failure.
            flush_error = exc
        finally:
            self._cancel.set()
            self._wake.set()
            self._published.set()
            self._thread.join(max(0.0, deadline - _REAL_MONOTONIC()))
        if self._thread.is_alive():
            stopped = TimeoutError("fake trace publisher did not stop")
            if flush_error is not None:
                stopped.add_note(f"trace flush also failed: {flush_error}")
            raise stopped
        if flush_error is not None:
            raise flush_error

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._cancel.is_set():
                return
            # A producer/writer burst represents one observable state advance,
            # not a requirement for one filesystem replace per changed field.
            while True:
                with self._lock:
                    observed_revision = self._revision
                    urgent = self._flush_revision > self._published_revision
                if not urgent:
                    self._wake.wait(_TRACE_COALESCE_SECONDS)
                    self._wake.clear()
                with self._lock:
                    urgent = self._flush_revision > self._published_revision
                    if not urgent and observed_revision != self._revision:
                        continue
                    if self._published_revision >= self._revision:
                        if self._closed:
                            return
                        snapshot = None
                    else:
                        revision = self._revision
                        snapshot = dict(self._trace)
                    break
            if snapshot is None:
                continue
            try:
                _atomic_publish_trace(self._trace_dir, snapshot, self._cancel)
            except _TracePublicationCancelled:
                self._published.set()
                return
            except Exception as exc:
                with self._lock:
                    self._error = exc
                self._published.set()
                return
            with self._lock:
                self._published_revision = revision
                if self._flush_revision <= revision:
                    self._flush_revision = 0
                pending = self._published_revision < self._revision
                closed = self._closed
            self._published.set()
            if pending:
                self._wake.set()
            elif closed:
                return


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
        self.finalize_entered = threading.Event()
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
        self.finalize_entered.set()
        if self.fail_at == "close":
            raise OSError("injected close failure")
        self.writer.finalize()
        self.closed = True


class ControlledMetadataAppender:
    """Real metadata append with a deterministic finalizer pause."""

    def __init__(self, *, pause=True):
        self.entered = threading.Event()
        self.release = threading.Event()
        if not pause:
            self.release.set()

    def __call__(self, path, metadata, **kwargs):
        from base.wav_calibration_metadata import append_wav_calibration_metadata_result

        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("test metadata finalizer release was not signalled")
        return append_wav_calibration_metadata_result(path, metadata, **kwargs)


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
    trace_publisher = _CoalescedTracePublisher(trace_dir)
    writer_consumed = threading.Event()
    pace_writer = options.get("pace_writer", "frames" in options)

    def update(**changes):
        trace_publisher.update(**changes)

    preview_fault = options.get("preview_fault")
    if preview_fault is not None:
        if preview_fault not in ("construct", "begin", "append", "snapshot"):
            raise ValueError("unknown injected preview fault")
        from base import recording_capture as recording_capture_module

        real_session = recording_capture_module.MultichannelWaveformSession
        preview_session_constructions = 0

        class FaultyPreviewSession(real_session):
            def __init__(self, **kwargs):
                nonlocal preview_session_constructions
                preview_session_constructions += 1
                update(
                    preview_session_constructions=preview_session_constructions,
                    preview_rolling_window_seconds=kwargs.get("rolling_window_seconds"),
                )
                if preview_fault == "construct":
                    update(preview_fault_observed=preview_fault)
                    raise RuntimeError("injected preview construction failure")
                super().__init__(**kwargs)

            def begin(self, **kwargs):
                if preview_fault == "begin":
                    update(preview_fault_observed=preview_fault)
                    raise RuntimeError("injected preview begin failure")
                return super().begin(**kwargs)

            def append(self, block):
                if preview_fault == "append":
                    update(preview_fault_observed=preview_fault)
                    raise RuntimeError("injected preview append failure")
                return super().append(block)

            def snapshots(self):
                if preview_fault == "snapshot":
                    update(preview_fault_observed=preview_fault)
                    raise RuntimeError("injected preview snapshot failure")
                return super().snapshots()

        recording_capture_module.MultichannelWaveformSession = FaultyPreviewSession

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
                            # The writer can wake this wait; stop is still
                            # checked each bounded wait, before the deadline.
                            writer_consumed.wait(.005)
                            if self.stop_feed.is_set():
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
            trace_publisher.flush()

    class ProcessBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.round = 0

        def InputStream(self, **config):
            self.stream = ProcessStream(self, **config)
            return self.stream

        def flush_trace(self):
            trace_publisher.flush()

        def close_trace(self):
            trace_publisher.close()

        Stream = InputStream

    class ProcessWriter(StreamingWavWriter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fake_chunks_written = 0

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
            self._fake_chunks_written += 1
            # Long synthetic captures validate the finalized WAV, so bounded
            # flush batching avoids turning 300 tiny fake chunks into a disk
            # throughput benchmark. Manual fixtures retain immediate progress.
            flush_every = 16 if "frames" in options else 1
            if self._fake_chunks_written % flush_every == 0:
                self.sf_file.flush()
            update(writer_pid=os.getpid(), written_frames=self.total_frames)
            writer_consumed.set()

        def finalize(self):
            primary_failure = None
            try:
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
            except Exception as exc:
                primary_failure = exc
                raise
            finally:
                # The spawned worker can serve another recording request; flush
                # this terminal state without retiring its shared trace thread.
                try:
                    trace_publisher.flush()
                except Exception as trace_error:
                    if primary_failure is None:
                        raise
                    primary_failure.add_note(
                        f"fake trace cleanup failed: {trace_error}")

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


def persistent_ve_worker_dependencies(**options):
    """Spawn-safe persistent-controller SDK, writer and finalizer gates."""
    from unit_test.base.ve3668n_fakes import CaptureSDK

    trace_dir = Path(options["trace_dir"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    pause_finalizers = set(options.get("pause_finalizers", ()))
    pause_writers = set(options.get("pause_writers", ()))
    idle_fail_after_reads = options.get("idle_fail_after_reads")
    fail_release_operation = options.get("fail_release_operation")
    pause_release_operation = options.get("pause_release_operation")

    def request_id(path):
        return Path(path).stem

    def wait_for(marker, *, timeout=10):
        deadline = time.monotonic() + timeout
        while not (trace_dir / marker).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"test gate was not released: {marker}")
            threading.Event().wait(.005)

    class PersistentSDK(CaptureSDK):
        def __init__(self):
            failures = () if fail_release_operation is None else (fail_release_operation,)
            super().__init__(failures=failures)
            self._reads = 0

        def _call(self, operation, *args, **kwargs):
            with open(trace_dir / "native.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"operation": operation, "pid": os.getpid(),
                                         "thread_id": threading.get_ident()}) + "\n")
            if operation == pause_release_operation:
                (trace_dir / f"release-{operation}-entered").touch()
                wait_for(f"release-{operation}")
            return super()._call(operation, *args, **kwargs)

        def read_task_data(self, *args, **kwargs):
            self._reads += 1
            if idle_fail_after_reads is not None and self._reads > idle_fail_after_reads:
                wait_for("trigger-idle-failure")
                raise RuntimeError("injected idle native read failure")
            threading.Event().wait(.005)
            return super().read_task_data(*args, **kwargs)

    class PausedWriter(StreamingWavWriter):
        def write_chunk(self, chunk):
            identity = request_id(self.file_path)
            if identity in pause_writers and self.total_frames == 0:
                (trace_dir / f"writer-{identity}-entered").touch()
                wait_for(f"release-writer-{identity}")
            super().write_chunk(chunk)

    def append_metadata(path, metadata, **kwargs):
        from base.wav_calibration_metadata import append_wav_calibration_metadata_result

        identity = request_id(path)
        if identity in pause_finalizers:
            (trace_dir / f"finalizer-{identity}-entered").touch()
            wait_for(f"release-finalizer-{identity}")
        return append_wav_calibration_metadata_result(path, metadata, **kwargs)

    return {
        "ve_sdk_factory": PersistentSDK,
        "writer_factory": PausedWriter,
        "metadata_appender": append_metadata,
    }


def failing_control_worker(control, preview, generation, backend_factory, backend_options,
                           cancel_timeout, preview_interval):
    """Inject a non-IPC sender failure after the controller owns native state."""
    from base.recording_worker import recording_worker

    class Connection:
        def __getattr__(self, name):
            return getattr(control, name)

        def send(self, event):
            if event.kind == backend_options["fail_control_kind"]:
                raise RuntimeError("injected sender serialization failure")
            control.send(event)

    recording_worker(Connection(), preview, generation, backend_factory, backend_options,
                     cancel_timeout, preview_interval)


def outer_exception_control_worker(
        control, preview, generation, backend_factory, backend_options,
        cancel_timeout, preview_interval):
    """Raise from the worker's control receive boundary on a selected command."""
    from base.recording_worker import recording_worker

    class Connection:
        def __getattr__(self, name):
            return getattr(control, name)

        def recv(self):
            command = control.recv()
            if command.kind == backend_options["fail_recv_kind"]:
                raise RuntimeError("injected outer control receive failure")
            return command

    recording_worker(Connection(), preview, generation, backend_factory, backend_options,
                     cancel_timeout, preview_interval)


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
