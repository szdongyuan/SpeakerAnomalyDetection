"""Child-local capture and WAV ownership. Importing this module opens no device.

``start`` and ``cancel`` never wait for native calls. The worker polls ``started``
and ``done`` and enforces its own process-level deadlines. ``wait`` is only for
non-GUI owners/tests. No callbacks, queues or raw audio are sent through IPC here.
"""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import math
import os
import sys
import tempfile
import threading

import numpy as np
import soundfile as sf

from base.multichannel_waveform_session import MultichannelWaveformSession
from base.recording_process_protocol import (
    RecordingCancelled, RecordingFailure, RecordingPreview, RecordingRequest, RecordingResult,
)
from base.recording_settings import validate_recorded_audio
from base.streaming_file_writer import StreamingWavWriter
from base.wav_calibration_metadata import (
    WavCalibrationMetadataAppendResult,
    WavCalibrationMetadataReadStatus,
    append_wav_calibration_metadata_result,
    inspect_wav_calibration_metadata,
)
from consts.recording_preview_consts import (
    MAIN_RECORDING_LIVE_MAX_POINTS,
    MAIN_RECORDING_LIVE_WINDOW_SECONDS,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
)
from consts.ve3668n_consts import VE_BACKEND


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
                 metadata_appender=append_wav_calibration_metadata_result,
                 blocksize=2048, queue_seconds=2.0):
        self.request = request
        self._backend = backend
        self._is_ve = request.device.get("backend") == VE_BACKEND
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
        self.written_frames = 0
        self._final_frames = 0
        self.outcome = None
        self._failure = None
        self._thread = None
        self._stream = None
        self._writer = None
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
        self._logger = logging.getLogger(__name__)
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

    def snapshot(self, *, generation, sequence):
        """Copy only the selected bounded envelope; skip a busy consumer."""
        if (not self._preview_enabled or self._waveforms is None
                or not self._waveform_lock.acquire(blocking=False)):
            return None
        try:
            snapshots = tuple(self._waveforms.snapshots().values())
            return RecordingPreview(
                self.request.request_id,
                generation,
                sequence,
                snapshots[0].sample_stop,
                self.request.channels,
                snapshots,
                self.request.preview_time_mode,
            )
        except Exception as exc:
            # Presentation boundary: reducer/snapshot faults disable preview only.
            self._disable_preview(exc)
            return None
        finally:
            self._waveform_lock.release()

    def _disable_preview(self, exc):
        if self._preview_enabled:
            self._preview_enabled = False
            self._waveforms = None
            self._warnings.append(f"preview disabled: {exc}")
            self._logger.exception("Preview failed for %s", self.request.request_id)

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
        self._stage = "write"
        self._writer.write_chunk(block)
        self.written_frames += len(block)
        self._final_frames = self.written_frames
        if self._preview_enabled and self._waveforms is not None:
            with self._waveform_lock:
                try:
                    self._waveforms.append(block)
                except Exception as exc:
                    # A display reducer is not an audio-integrity dependency.
                    self._disable_preview(exc)

    def _close_stream(self):
        stream = self._stream
        if stream is None or self._stream_close_attempted:
            return
        self._stream_close_attempted = True
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
        try:
            writer.finalize()
        except Exception as exc:
            # Writer boundary: normalize a failed close once; handle release is
            # unknown and the worker owner must retire the process before reuse.
            self._handles_released = False
            self._unreleased_finalization_handles.append((self.request.path, writer))
            self._logger.exception("WAV close failed for %s", self.request.request_id)
            self._fail("close_wav", str(exc))
            return
        self._writer_released = True
        self._writer = None

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
            self._close_stream()
            while self._writer is not None:
                block = self._pop_block()
                if block is None:
                    break
                self._consume(block)
            self._close_writer()
            self._publish_capture_slot()
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

    def _finish_audio(self):
        req = self.request
        self._stage = "counts"
        if self.raw_frames != self.written_frames:
            raise ValueError("accepted and written frame counts differ")
        if self._cancelled.is_set():
            return
        if self.raw_frames != req.target_samples:
            raise ValueError("recording ended before its target sample count")
        self._stage = "read_wav"
        with self._finalization_file(req.path) as source:
            if (source.subtype != "FLOAT" or source.samplerate != req.sample_rate
                    or source.channels != len(req.channels) or len(source) != self.raw_frames):
                raise ValueError("saved WAV shape, rate or float32 format differs from request")
            audio = source.read(dtype="float32", always_2d=True)
            if self._is_ve:
                if audio.shape != (self.raw_frames, len(req.channels)):
                    raise ValueError("saved WAV frame/channel shape differs from request")
                if not np.isfinite(audio).all():
                    raise ValueError("saved WAV contains non-finite voltage samples")
        if self._effective_trim:
            audio = audio[self._effective_trim:]
            self._stage = "trim"
            self._rewrite_trimmed(audio)
            self._final_frames = len(audio)
        elif req.purpose == "main" and req.trim_samples >= self.raw_frames:
            self._warnings.append("startup trim skipped: trim is not smaller than recorded audio")
        metadata_appended = False
        if self._cancelled.is_set():
            return
        if req.purpose == "main":
            self._stage = "validation"
            quality_audio = audio / req.device["input_config"]["range_max"] if self._is_ve else audio
            ok, reason, detail = validate_recorded_audio(quality_audio, req.validation_thresholds.to_dict())
            if not ok:
                raise ValueError(f"{reason} {detail}")
            self._stage = "metadata"
            metadata = req.calibration_metadata.to_dict() if req.calibration_metadata is not None else None
            metadata_result = self._metadata_appender(req.path, metadata, logger=self._logger)
            if isinstance(metadata_result, WavCalibrationMetadataAppendResult):
                self._owned_temporary_paths.update(metadata_result.cleanup_paths)
                self._unreleased_finalization_handles.extend(metadata_result.retained_handles)
                self._warnings.extend(metadata_result.close_errors)
                if (not metadata_result.handles_released
                        or (self._is_ve and metadata_result.retained_handles)):
                    self._handles_released = False
                    self._fail("metadata", metadata_result.primary_error or "; ".join(metadata_result.close_errors)
                               or "WAV metadata file handles were not released")
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
                if len(source) != len(audio) or source.channels != len(req.channels) or source.subtype != "FLOAT":
                    raise ValueError("WAV audio became invalid during metadata finalization")
                if self._is_ve and (source.samplerate != req.sample_rate or not np.array_equal(
                        source.read(dtype="float32", always_2d=True), audio)):
                    raise ValueError("VE WAV rate or raw voltage data changed during metadata finalization")
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
        if self._status_warning:
            self._warnings.append(self._status_warning)
        self.outcome = RecordingResult(req.request_id, req.purpose, req.path, req.sample_rate,
                                       req.channels, self.raw_frames, len(audio), metadata_appended,
                                       tuple(self._warnings),
                                       cleanup_paths=tuple(sorted(self._owned_temporary_paths)))

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

    def _rewrite_trimmed(self, audio):
        req = self.request
        descriptor, temporary = tempfile.mkstemp(prefix=".recording-trim-", suffix=".wav", dir=os.path.dirname(req.path))
        self._owned_temporary_paths.add(temporary)
        try:
            self._close_finalization_handle(descriptor, temporary, lambda: os.close(descriptor))
            with self._finalization_file(temporary, mode="w", samplerate=req.sample_rate,
                                         channels=len(req.channels), format="WAV", subtype="FLOAT") as output:
                output.write(audio)
            os.replace(temporary, req.path)
        finally:
            # Never replace/delete a file whose handle release is uncertain.
            # A processing failure with successful close still cleans its temp.
            if self._handles_released:
                if os.path.exists(temporary):
                    os.unlink(temporary)
                self._owned_temporary_paths.discard(temporary)
