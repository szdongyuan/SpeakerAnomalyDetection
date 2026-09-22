"""Child-local capture and WAV ownership. Importing this module opens no device.

``start`` and ``cancel`` never wait for native calls. The worker polls ``started``
and ``done`` and enforces its own process-level deadlines. ``wait`` is only for
non-GUI owners/tests. No callbacks, queues or raw audio are sent through IPC here.
"""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
import sys
import threading
import time

import numpy as np
import soundfile as sf

from base.log_manager import LogManager
from base.multichannel_waveform_session import MultichannelWaveformSession
from base.recording_process_protocol import (
    RecordingCancelled, RecordingFailure, RecordingPreview, RecordingRequest, RecordingResult,
    RecordingFinalizationTiming,
)
from base.streaming_file_writer import StreamingWavWriter
from base.wav_calibration_metadata import (
    WavCalibrationMetadataAppendResult,
    WavCalibrationMetadataReadStatus,
    append_owned_recording_calibration_metadata_result,
    inspect_wav_calibration_metadata,
)
from consts.recording_preview_consts import (
    MAIN_RECORDING_LIVE_MAX_POINTS,
    MAIN_RECORDING_LIVE_WINDOW_SECONDS,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
)
from consts.ve3668n_consts import VE_BACKEND
from consts.recording_result_consts import RECORDING_SAMPLE_DIGEST_ALGORITHM


def capture_queue_capacity(sample_rate, channels, *, blocksize=2048, seconds=2.0):
    """Return (frames, bytes), excluding at most one consumer-owned callback block."""
    if sample_rate <= 0 or channels <= 0 or blocksize <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("queue dimensions and duration must be positive")
    frames = max(math.ceil(sample_rate * seconds), blocksize)
    return frames, frames * channels * np.dtype(np.float32).itemsize


def sounddevice_backend():
    # Set before the child first imports sounddevice; never use parent sd.default.
    os.environ["SD_ENABLE_ASIO"] = "1"
    import sounddevice
    return sounddevice


@dataclass(frozen=True)
class CaptureSlotState:
    """Request-local proof that VE capture and WAV writing have ended."""

    target_reached_at: float
    raw_frames: int
    adapter_released: bool
    writer_released: bool

    def __post_init__(self):
        if (type(self.target_reached_at) not in (int, float)
                or not math.isfinite(self.target_reached_at)
                or self.target_reached_at < 0):
            raise ValueError("target_reached_at must be a finite nonnegative time")
        if type(self.raw_frames) is not int or self.raw_frames <= 0:
            raise ValueError("raw_frames must be a positive integer")
        if type(self.adapter_released) is not bool or type(self.writer_released) is not bool:
            raise ValueError("capture-slot release flags must be booleans")


class RecordingCapture:
    def __init__(self, request: RecordingRequest, *, backend=None,
                 ve_stream_factory=None,
                 writer_factory=StreamingWavWriter,
                 metadata_appender=append_owned_recording_calibration_metadata_result,
                 blocksize=2048, queue_seconds=2.0, diagnostics=None):
        self.request = request
        self._backend = backend
        self._is_ve = request.device.get("backend") == VE_BACKEND
        self._diagnostics = diagnostics if self._is_ve else None
        self._consume_lane = (self._diagnostics.new_lane(request=request.request_id)
                              if self._diagnostics is not None else None)
        self._diagnostic_blocks = 0
        self._tail_started_ns = None
        # Request-local tail only; the shared helper's maxima span a generation.
        self._tail_stats = {}
        self._ve_stream_factory = ve_stream_factory
        self._native_stream = None
        self._writer_factory = writer_factory
        self._metadata_appender = metadata_appender
        self._blocksize = blocksize
        self.queue_capacity_frames, self.queue_capacity_bytes = capture_queue_capacity(
            request.sample_rate, len(request.channels), blocksize=blocksize, seconds=queue_seconds)
        self.started = threading.Event()
        self.done = threading.Event()
        self.capture_slot_released = threading.Event()
        self.capture_slot = None
        self._wake = threading.Event()
        self._cancelled = threading.Event()
        self._stop_requested = threading.Event()
        self._queue_lock = threading.Lock()
        self._waveform_lock = threading.Lock()
        self._blocks = deque()
        self._queued_frames = 0
        self.raw_frames = 0
        self.consumed_frames = 0
        self._sample_digest = hashlib.sha256()
        self._target_sample_arrival = None
        self._file_closed_at = None
        self.written_frames = 0
        self._final_frames = 0
        self.outcome = None
        self._failure = None
        self._thread = None
        self._stream = None
        self._writer = None
        self._writer_finalization_log = None
        self._stream_close_attempted = False
        self._writer_close_attempted = False
        self._adapter_released = False
        self._writer_released = False
        self._handles_released = True
        self._unreleased_finalization_handles = []
        self._owned_temporary_paths = set()
        self._stage = "starting"
        self._warnings = []
        self._status_warning = None
        self._logger = LogManager.set_log_handler("core")
        self._effective_trim = request.trim_samples if request.purpose == "main" else 0
        # For a known overlarge trim, finalization retains all audio too.
        if self._effective_trim >= request.target_samples:
            self._effective_trim = 0
        self._preview_enabled = request.effective_streaming
        self._waveforms = None
        if self._preview_enabled:
            try:
                self._waveforms = MultichannelWaveformSession(
                    max_points=MAIN_RECORDING_LIVE_MAX_POINTS,
                    rolling_window_seconds=(
                        MAIN_RECORDING_LIVE_WINDOW_SECONDS
                        if request.preview_time_mode == PREVIEW_TIME_MODE_RELATIVE_LATEST
                        else None
                    ),
                )
                self._waveforms.begin(
                    channels=request.channels,
                    sample_rate=request.sample_rate,
                    startup_trim_samples=self._effective_trim,
                )
            except Exception as exc:
                # Preview setup is optional and cannot own authoritative capture.
                self._disable_preview(exc)

    @property
    def queued_frames(self):
        with self._queue_lock:
            return self._queued_frames

    @property
    def started_at(self):
        """Native start time, not worker-poll time; None for legacy capture."""
        return self._native_stream.started_at if self._native_stream is not None else None

    def progress_snapshot(self):
        """Keep final native progress after the active stream has been closed."""
        if self._native_stream is None:
            return None
        snapshot = self._native_stream.progress_snapshot
        return snapshot() if callable(snapshot) else snapshot

    @property
    def native_diagnostics(self):
        return self._native_stream.diagnostics if self._native_stream is not None else ()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("a capture can only be started once")
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.request.request_id}", daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancelled.set()
        self._stop_requested.set()
        self._wake.set()

    def wait(self, timeout=None):
        if not self.done.wait(timeout):
            raise TimeoutError(f"capture {self.request.request_id} is still {self._stage}")
        return self.outcome

    def join(self, timeout=0):
        """Confirm thread exit; ``done`` alone is published just before return."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def stream_closed(self):
        """Read after joining: no stream remains with uncertain native ownership."""
        return self._stream is None

    def snapshot(self, *, generation, sequence):
        """Copy only the selected bounded envelope; skip a busy consumer."""
        if not self._preview_enabled or self._waveforms is None:
            return None
        with self._diagnostic_phase("snapshot"):
            return self._snapshot(generation=generation, sequence=sequence)

    def _snapshot(self, *, generation, sequence):
        with self._diagnostic_phase("waveform_lock_wait", emit_slow=False):
            acquired = self._waveform_lock.acquire(blocking=False)
        if not acquired:
            return None
        preview_error = None
        try:
            with self._diagnostic_phase("waveform_lock_hold", emit_slow=False):
                snapshots = tuple(self._waveforms.snapshots().values())
                return RecordingPreview(
                    self.request.request_id, generation, sequence,
                    snapshots[0].sample_stop, self.request.channels,
                    snapshots, self.request.preview_time_mode)
        except Exception as exc:
            # Presentation boundary: reducer/snapshot faults disable preview only.
            preview_error = exc
            return None
        finally:
            self._waveform_lock.release()
            if preview_error is not None:
                self._disable_preview(preview_error)

    @contextmanager
    def _diagnostic_phase(self, stage, *, emit_slow=True):
        diagnostic = self._diagnostics
        if diagnostic is None:
            yield
            return
        token = diagnostic.begin(stage, request=self.request.request_id)
        try:
            yield
        finally:
            diagnostic.end(token, emit_slow=emit_slow)

    def _diagnostic_milestone(self, stage, **fields):
        if self._diagnostics is None:
            return
        progress = self.progress_snapshot()
        self._diagnostics.milestone(
            stage, request=self.request.request_id,
            target_samples=self.request.target_samples, raw_frames=self.raw_frames,
            consumed_frames=self.consumed_frames, written_frames=self.written_frames,
            queued_frames=self.queued_frames, adapter_released=self._adapter_released,
            writer_released=self._writer_released,
            target_reached_at=(progress.last_frame_at if progress is not None
                               and progress.frames == self.request.target_samples else None),
            **fields)

    def _diagnostic_tail_summary(self):
        if self._diagnostics is None:
            return
        fields = {}
        for stage, values in self._tail_stats.items():
            for name, value in zip(("count", "total_ns", "max_ns", "max_start_ns", "max_end_ns"), values):
                fields[f"{stage}_{name}"] = value
        self._diagnostics.milestone(
            "capture_tail", request=self.request.request_id,
            scope="request_tail", tail_started_ns=self._tail_started_ns, **fields)
        self._diagnostics.summary("capture_summary", request=self.request.request_id,
                                  scope="generation_cumulative", capture_detail_stride=32,
                                  capture_max_scope="sampled_plus_all_slow",
                                  tail_started_ns=self._tail_started_ns)

    def _disable_preview(self, exc):
        if self._preview_enabled:
            self._preview_enabled = False
            self._waveforms = None
            self._warnings.append(f"preview disabled: {exc}")
            self._logger.exception("Preview failed for %s", self.request.request_id,
                                   exc_info=(type(exc), exc, exc.__traceback__))

    def _fail(self, stage, message):
        with self._queue_lock:
            if self._failure is None:
                self._failure = (stage, message)
        self._stop_requested.set()
        self._wake.set()

    def _accept(self, indata, frames, status):
        if self._stop_requested.is_set():
            return None
        if getattr(status, "input_overflow", False):
            identity = self.request.device["machine_id"] if self._is_ve else self.request.device["index"]
            self._fail("capture", f"input overflow on device {identity}")
            return None
        if status and self._status_warning is None:
            self._status_warning = str(status)
        if (not isinstance(indata, np.ndarray) or indata.ndim != 2 or indata.shape[0] != frames
                or indata.shape[1] <= max(self.request.channels) or frames <= 0
                or (self._is_ve and type(frames) is not int)):
            self._fail("capture", "driver input shape does not match frame/channel contract")
            return None
        with self._queue_lock:
            if self._stop_requested.is_set():
                return None
            accepted = min(frames, self.request.target_samples - self.raw_frames)
            if self._queued_frames + accepted > self.queue_capacity_frames:
                self._failure = ("capture", f"audio queue capacity exceeded ({self.queue_capacity_bytes} bytes)")
                self._stop_requested.set()
                self._wake.set()
                return None
            # Driver memory is borrowed. Never queue a view of it.
            with np.errstate(over="ignore", invalid="ignore"):
                owned = np.array(indata[:accepted, self.request.channels], dtype=np.float32, order="C", copy=True)
            if self._is_ve and not np.isfinite(owned).all():
                self._failure = ("capture", "driver returned non-finite float32 voltage samples")
                self._stop_requested.set()
                self._wake.set()
                return None
            self._blocks.append(owned)
            self._queued_frames += accepted
            self.raw_frames += accepted
            if self.raw_frames == self.request.target_samples:
                self._target_sample_arrival = time.monotonic()
                self._stop_requested.set()
        self._wake.set()
        return owned

    def _input_callback(self, indata, frames, time_info, status):
        try:
            self._accept(indata, frames, status)
        except Exception as exc:
            # PortAudio otherwise suppresses callback exceptions. Record once and
            # wake the owner to stop/close; never log or send IPC on this thread.
            self._fail("capture", f"audio callback failed: {exc}")

    def _validate_device(self, snapshot, channels, direction):
        current = self._backend.query_devices(snapshot["index"])
        for key in ("name", "hostapi"):
            if current.get(key) != snapshot[key]:
                raise ValueError(f"{direction} device identity changed at index {snapshot['index']}: {key}")
        if max(channels) >= int(current.get(f"max_{direction}_channels", 0)):
            raise ValueError(f"{direction} device no longer supports selected channels")

    def _open(self):
        req = self.request
        self._stage = "device"
        if self._is_ve:
            from base.ve3668n_capture import Ve3668nInputStream

            self._stage = "open_wav"
            self._writer = self._writer_factory(req.path, sample_rate=req.sample_rate, channels=len(req.channels))
            self._stage = "device"
            factory = self._ve_stream_factory or Ve3668nInputStream
            self._stream = self._native_stream = factory(
                request=req, callback=self._input_callback, fail=self._fail,
                stop_event=self._stop_requested,
            )
            if not self._cancelled.is_set() and self._stream.start():
                self.started.set()
            return
        if self._backend is None:
            self._backend = sounddevice_backend()
        self._validate_device(req.device, req.channels, "input")
        self._stage = "open_wav"
        self._writer = self._writer_factory(req.path, sample_rate=req.sample_rate, channels=len(req.channels))
        self._stage = "device"
        config = dict(samplerate=req.sample_rate, dtype="float32", blocksize=self._blocksize)
        self._stream = self._backend.InputStream(**config, channels=max(req.channels) + 1,
                                                 device=req.device["index"], callback=self._input_callback)
        if not self._cancelled.is_set():
            self._stream.start()
            self.started.set()

    def _pop_block(self):
        with self._queue_lock:
            if not self._blocks:
                return None
            block = self._blocks.popleft()
            self._queued_frames -= len(block)
            return block

    def _consume(self, block):
        lane = self._consume_lane
        if lane is None:
            self._consume_block(block)
            return
        self._diagnostic_blocks += 1
        measured = self._tail_started_ns is not None or self._diagnostic_blocks % 32 == 1
        phase = lane.enter("consume")
        intervals = []
        try:
            self._consume_block(block, lane, phase, intervals)
        finally:
            ended = lane.perf_ns()
            lane.active = None
            # A slow sub-operation necessarily makes the complete consume slow.
            # Aggregate after all business locks, retaining every slow interval.
            if measured or ended - phase[1] >= 100_000_000:
                intervals.append((phase, ended))
                for current, end in intervals:
                    elapsed = lane.finish(current, ended_ns=end)
                    self._record_tail(current[0], current[1], elapsed)

    def _record_tail(self, stage, started, elapsed):
        if self._tail_started_ns is None or started < self._tail_started_ns:
            return
        stats = self._tail_stats.setdefault(stage, [0, 0, 0, 0, 0])
        stats[0] += 1
        stats[1] += elapsed
        if elapsed > stats[2]:
            stats[2:] = [elapsed, started, started + elapsed]

    def _consume_block(self, block, lane=None, consume_phase=None, intervals=None):
        self._stage = "write"
        skip = min(len(block), max(0, self._effective_trim - self.consumed_frames))
        self.consumed_frames += len(block)
        retained = block[skip:]
        if len(retained):
            if lane is None:
                self._writer.write_chunk(retained)
            else:
                # Publish an immutable current phase; count complete blocks
                # once in _consume, not a dictionary counter per sub-stage.
                lane.active = phase = ("write", lane.perf_ns())
                try:
                    self._writer.write_chunk(retained)
                finally:
                    intervals.append((phase, lane.perf_ns()))
                    lane.active = consume_phase
            self.written_frames += len(retained)
            self._final_frames = self.written_frames
            self._sample_digest.update(retained.astype("<f4", copy=False).tobytes(order="C"))
        if self._preview_enabled and self._waveforms is not None:
            preview_error = None
            if lane is not None:
                lane.active = waiting = ("waveform_lock_wait", lane.perf_ns())
            self._waveform_lock.acquire()
            if lane is not None:
                lane.active = holding = ("waveform_lock_hold", lane.perf_ns())
            try:
                try:
                    self._waveforms.append(block)
                except Exception as exc:
                    # A display reducer is not an audio-integrity dependency.
                    preview_error = exc
            finally:
                self._waveform_lock.release()
                if lane is not None:
                    intervals.append((holding, lane.perf_ns()))
                    intervals.append((waiting, holding[1]))
                    lane.active = consume_phase
            if preview_error is not None:
                self._disable_preview(preview_error)

    def _close_stream(self):
        stream = self._stream
        if stream is None or self._stream_close_attempted:
            return
        self._stream_close_attempted = True
        self._diagnostic_milestone("close_stream_begin")
        try:
            with self._diagnostic_phase("close_stream"):
                self._close_stream_once(stream)
        finally:
            self._diagnostic_milestone("close_stream_end", status=(
                "success" if self._stream is None else "error"))

    def _close_stream_once(self, stream):
        released = True
        operations = [stream.stop]
        if not self._is_ve:
            operations.append(stream.close)
        for operation in operations:
            try:
                operation()
            except Exception as exc:
                # Native backend cleanup can raise arbitrary backend exceptions;
                # still attempt close after stop fails, and forbid successful delivery.
                self._handles_released = False
                released = False
                self._logger.exception("Stream cleanup failed for %s", self.request.request_id)
                self._fail("close_stream", str(exc))
        if self._is_ve:
            if stream.handles_released:
                try:
                    stream.close()
                except Exception as exc:
                    released = False
                    self._handles_released = False
                    self._logger.exception("Stream cleanup failed for %s", self.request.request_id)
                    self._fail("close_stream", str(exc))
            else:
                released = False
                self._handles_released = False
                self._fail("close_stream", "VE native handles are not released: " + "; ".join(stream.diagnostics))
            for diagnostic in stream.diagnostics:
                self._logger.warning("VE capture %s: %s", self.request.request_id, diagnostic)
            self._adapter_released = released and stream.handles_released
        if released:
            self._stream = None

    def _close_writer(self):
        writer = self._writer
        if writer is None or self._writer_close_attempted:
            return
        self._writer_close_attempted = True
        if isinstance(writer, StreamingWavWriter):
            writer.defer_finalization_log()
        finalized = False
        try:
            self._diagnostic_milestone("close_wav_begin")
            with self._diagnostic_phase("close_wav"):
                writer.finalize()
            finalized = True
            self._writer_released = True
            self._writer = None
            if isinstance(writer, StreamingWavWriter):
                self._writer_finalization_log = writer
        except Exception as exc:
            # Writer boundary: normalize a failed close once; handle release is
            # unknown and the worker owner must retire the process before reuse.
            self._handles_released = False
            self._unreleased_finalization_handles.append((self.request.path, writer))
            self._logger.exception("WAV close failed for %s", self.request.request_id)
            self._fail("close_wav", str(exc))
            return
        finally:
            self._diagnostic_milestone("close_wav_end", status=(
                "success" if finalized else "error"), finalize_returned=finalized)

    def _emit_writer_finalization_log(self):
        writer = self._writer_finalization_log
        self._writer_finalization_log = None
        if writer is not None:
            try:
                writer.emit_finalization_log()
            except Exception as exc:
                # External optional logger boundary, after proven physical close.
                # Retain one bounded warning; never turn diagnostic failure into
                # unknown file ownership or recursively write another log.
                self._warnings.append(f"WAV finalization log failed: {type(exc).__name__}: {exc}"[:512])

    def _publish_capture_slot(self):
        if (not self._is_ve or self.capture_slot_released.is_set()
                or not self._adapter_released or not self._writer_released
                or self.raw_frames != self.request.target_samples
                or self.queued_frames != 0):
            return
        progress = self.progress_snapshot()
        if (progress is None or progress.frames != self.raw_frames
                or progress.last_frame_at is None):
            return
        self.capture_slot = CaptureSlotState(
            target_reached_at=progress.last_frame_at,
            raw_frames=self.raw_frames,
            adapter_released=True,
            writer_released=True,
        )
        self.capture_slot_released.set()
        self._diagnostic_milestone("capture_slot_published")

    def _discard_failed_blocks(self):
        with self._queue_lock:
            self._blocks.clear()
            self._queued_frames = 0

    def _run(self):
        try:
            if not self._cancelled.is_set():
                self._open()
            while not self._stop_requested.is_set():
                self._wake.wait()
                self._wake.clear()
                while not self._stop_requested.is_set():
                    block = self._pop_block()
                    if block is None:
                        break
                    self._consume(block)
            self._stage = "finalizing"
            self._tail_started_ns = time.perf_counter_ns()
            self._diagnostic_milestone("finalizing")
            self._close_stream()
            self._diagnostic_milestone("final_drain_begin")
            try:
                with self._diagnostic_phase("final_drain"):
                    while self._writer is not None:
                        block = self._pop_block()
                        if block is None:
                            break
                        self._consume(block)
            finally:
                self._diagnostic_milestone("final_drain_end", status=(
                    "success" if sys.exception() is None else "error"))
            self._close_writer()
            file_closed_at = time.monotonic()
            self._publish_capture_slot()
            if self._target_sample_arrival is not None and self._file_closed_at is None:
                self._file_closed_at = file_closed_at
            self._emit_writer_finalization_log()
            if (self._file_closed_at is not None and self._failure is None
                    and not self._cancelled.is_set()):
                # Cancellation has a bounded worker shutdown deadline. Avoid
                # optional file-log I/O after its cleanup has already begun.
                self._logger.info("Recording timing request=%s process=child stage=target_samples child_monotonic=%.6f",
                    self.request.request_id, self._target_sample_arrival)
                self._logger.info("Recording timing request=%s process=child stage=drain_file_close seconds=%.6f",
                    self.request.request_id, self._file_closed_at - self._target_sample_arrival)
            if self._failure is None:
                self._finish_audio()
        except Exception as exc:
            # Capture thread's external device/file boundary. Unexpected backend,
            # writer and validator faults become one diagnostic failure, never success.
            self._logger.exception("Capture %s failed during %s (%s)",
                                   self.request.request_id, self._stage, self.request.path)
            self._fail(self._stage, str(exc))
        finally:
            self._stop_requested.set()
            self._close_stream()
            if self._failure is not None:
                self._discard_failed_blocks()
            self._close_writer()
            self._publish_capture_slot()
            self._emit_writer_finalization_log()
            if self._failure is not None:
                stage, message = self._failure
                self.outcome = RecordingFailure(self.request.request_id, stage, self.request.path,
                                                message, self.raw_frames, self.written_frames,
                                                self._handles_released,
                                                cleanup_paths=tuple(sorted(self._owned_temporary_paths)))
            elif self._cancelled.is_set():
                self.outcome = RecordingCancelled(self.request.request_id, self.request.path,
                                                  self.raw_frames, self._final_frames,
                                                  cleanup_paths=tuple(sorted(self._owned_temporary_paths)))
            self.done.set()
            if self._consume_lane is not None:
                self._consume_lane.close()
            self._diagnostic_tail_summary()

    def _finish_audio(self):
        req = self.request
        self._stage = "counts"
        if self.raw_frames != self.consumed_frames:
            raise ValueError("accepted and consumed raw frame counts differ")
        if self.written_frames != max(0, self.consumed_frames - self._effective_trim):
            raise ValueError("written frame count differs from streaming trim")
        if self._cancelled.is_set():
            return
        if self.raw_frames != req.target_samples:
            raise ValueError("recording ended before its target sample count")
        self._stage = "read_wav"
        with self._finalization_file(req.path) as source:
            if (source.subtype != "FLOAT" or source.samplerate != req.sample_rate
                    or source.channels != len(req.channels) or len(source) != self.written_frames):
                raise ValueError("saved WAV shape, rate or float32 format differs from request")
        if req.purpose == "main" and req.trim_samples >= self.raw_frames:
            self._warnings.append("startup trim skipped: trim is not smaller than recorded audio")
        metadata_appended = False
        if self._cancelled.is_set():
            return
        metadata_started = time.monotonic()
        if req.purpose == "main":
            self._stage = "metadata"
            metadata = req.calibration_metadata.to_dict() if req.calibration_metadata is not None else None
            metadata_result = self._metadata_appender(req.path, metadata, logger=self._logger)
            if isinstance(metadata_result, WavCalibrationMetadataAppendResult):
                self._owned_temporary_paths.update(metadata_result.cleanup_paths)
                self._unreleased_finalization_handles.extend(metadata_result.retained_handles)
                self._warnings.extend(metadata_result.close_errors)
                self._warnings.extend(metadata_result.rollback_errors)
                if (not metadata_result.handles_released
                        or metadata_result.retained_handles):
                    self._handles_released = False
                    self._fail("metadata", metadata_result.primary_error or "; ".join(metadata_result.close_errors)
                               or "WAV metadata file handles were not released")
                    return
                if metadata_result.rollback_succeeded is False:
                    self._fail("metadata", "; ".join(filter(None, (
                        metadata_result.primary_error, *metadata_result.rollback_errors)))
                        or "WAV metadata rollback could not be confirmed")
                    return
                if self._is_ve and metadata_result.primary_error is not None:
                    self._fail("metadata", metadata_result.primary_error)
                    return
                metadata_appended = metadata_result.appended
            else:
                # Compatibility for explicitly injected bool-only test appenders.
                metadata_appended = bool(metadata_result)
            if not metadata_appended:
                if self._is_ve:
                    raise ValueError("required VE WAV calibration metadata was not appended")
                self._warnings.append("WAV calibration metadata was not appended")
            # Optional metadata failure is a warning only while audio remains readable.
            with self._finalization_file(req.path) as source:
                if (len(source) != self.written_frames or source.channels != len(req.channels)
                        or source.subtype != "FLOAT" or source.samplerate != req.sample_rate):
                    raise ValueError("WAV audio became invalid during metadata finalization")
            if self._is_ve:
                diagnostic = inspect_wav_calibration_metadata(req.path, logger=self._logger)
                self._unreleased_finalization_handles.extend(diagnostic.retained_handles)
                self._warnings.extend(diagnostic.close_errors)
                if not diagnostic.handles_released or diagnostic.retained_handles:
                    self._handles_released = False
                    self._fail("metadata", diagnostic.primary_error or "; ".join(diagnostic.close_errors)
                               or "WAV metadata reader handle was not released")
                    return
                if diagnostic.primary_error is not None:
                    self._fail("metadata", diagnostic.primary_error)
                    return
                if (diagnostic.status is not WavCalibrationMetadataReadStatus.VALID
                        or diagnostic.declared_backend != VE_BACKEND
                        or diagnostic.metadata != req.calibration_metadata.to_dict()):
                    raise ValueError("VE WAV metadata readback differs from the frozen request snapshot")
        # Metadata append/readback may outlast a cancellation request. Preserve
        # its ownership and error checks above, then avoid optional success I/O
        # while the worker is completing bounded cancellation cleanup.
        if self._cancelled.is_set():
            return
        metadata_seconds = time.monotonic() - metadata_started
        self._logger.info("Recording timing request=%s process=child stage=metadata seconds=%.6f applicable=%s",
            req.request_id, metadata_seconds, req.purpose == "main")
        timing = None
        if self._target_sample_arrival is not None and self._file_closed_at is not None:
            timing = RecordingFinalizationTiming(
                self._file_closed_at - self._target_sample_arrival, metadata_seconds,
                time.monotonic() - self._target_sample_arrival)
            self._logger.info("Recording timing request=%s process=child stage=descriptor_ready target_elapsed_seconds=%.6f",
                req.request_id, timing.target_to_descriptor)
        if self._status_warning:
            self._warnings.append(self._status_warning)
        self.outcome = RecordingResult(req.request_id, req.purpose, req.path, req.sample_rate,
                                       req.channels, self.raw_frames, self.written_frames, metadata_appended,
                                       tuple(self._warnings),
                                       cleanup_paths=tuple(sorted(self._owned_temporary_paths)),
                                       digest_algorithm=RECORDING_SAMPLE_DIGEST_ALGORITHM,
                                       sample_digest=self._sample_digest.hexdigest(),
                                       finalization_timing=timing)

    def _close_finalization_handle(self, handle, path, close):
        try:
            close()
        except Exception as exc:
            # Native/library close operations may fail without releasing their
            # handles. Keep ownership until worker retirement, and let the thread
            # boundary report one failure with the exact affected file path.
            self._handles_released = False
            self._unreleased_finalization_handles.append((path, handle))
            raise OSError(f"Finalization file close failed for {path}: {exc}") from exc

    @contextmanager
    def _finalization_file(self, path, **kwargs):
        source = sf.SoundFile(path, **kwargs)
        try:
            yield source
        finally:
            original = sys.exception()
            try:
                self._close_finalization_handle(source, path, source.close)
            except OSError as cleanup:
                # The close helper already retains uncertain file ownership.
                # For VE, preserve a prior processing error and attach cleanup
                # diagnostics to its traceback rather than replacing its cause.
                if not self._is_ve or original is None:
                    raise
                original.add_note(str(cleanup))
