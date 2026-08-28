"""Child-local capture and WAV ownership. Importing this module opens no device.

``start`` and ``cancel`` never wait for native calls. The worker polls ``started``
and ``done`` and enforces its own process-level deadlines. ``wait`` is only for
non-GUI owners/tests. No callbacks, queues or raw audio are sent through IPC here.
"""
from collections import deque
from contextlib import contextmanager
import logging
import math
import os
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
    append_wav_calibration_metadata_result,
)


def capture_queue_capacity(sample_rate, channels, *, blocksize=2048, seconds=2.0):
    """Return (frames, bytes), excluding at most one consumer-owned callback block."""
    if sample_rate <= 0 or channels <= 0 or blocksize <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("queue dimensions and duration must be positive")
    frames = max(math.ceil(sample_rate * seconds), blocksize)
    return frames, frames * channels * np.dtype(np.float32).itemsize


def apply_monitor_startup_mute(play, *, mute_total, emitted_before, fade_len):
    """Legacy monitor math, including the ramp confined to the mute-ending block."""
    if mute_total > 0 and emitted_before < mute_total:
        remaining_mute = mute_total - emitted_before
        play = play.copy()
        hard_mute = min(remaining_mute, len(play))
        play[:hard_mute] = 0.0
        if hard_mute < len(play) and fade_len > 0:
            ramp_len = min(fade_len, len(play) - hard_mute)
            ramp = np.linspace(0.0, 1.0, ramp_len, endpoint=False, dtype=np.float32)
            play[hard_mute:hard_mute + ramp_len] *= ramp
    return play


def sounddevice_backend():
    # Set before the child first imports sounddevice; never use parent sd.default.
    os.environ["SD_ENABLE_ASIO"] = "1"
    import sounddevice
    return sounddevice


class RecordingCapture:
    def __init__(self, request: RecordingRequest, *, backend=None,
                 writer_factory=StreamingWavWriter,
                 metadata_appender=append_wav_calibration_metadata_result,
                 blocksize=2048, queue_seconds=2.0):
        self.request = request
        self._backend = backend
        self._writer_factory = writer_factory
        self._metadata_appender = metadata_appender
        self._blocksize = blocksize
        self.queue_capacity_frames, self.queue_capacity_bytes = capture_queue_capacity(
            request.sample_rate, len(request.channels), blocksize=blocksize, seconds=queue_seconds)
        self.started = threading.Event()
        self.done = threading.Event()
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
        self._handles_released = True
        self._unreleased_finalization_handles = []
        self._owned_temporary_paths = set()
        self._stage = "starting"
        self._warnings = []
        self._status_warning = None
        self._preview_enabled = request.effective_streaming
        self._waveforms = MultichannelWaveformSession(max_points=4000)
        self._effective_trim = request.trim_samples if request.purpose == "main" else 0
        # For a known overlarge trim, finalization retains all audio too.
        if self._effective_trim >= request.target_samples:
            self._effective_trim = 0
        self._waveforms.begin(channels=request.channels, sample_rate=request.sample_rate,
                              startup_trim_samples=self._effective_trim)
        self._monitor_emitted = 0
        self._monitor_gain = 10 ** (float(request.monitor.get("gain_db", 0.0)) / 20.0)
        self._logger = logging.getLogger(__name__)

    @property
    def queued_frames(self):
        with self._queue_lock:
            return self._queued_frames

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
        """Copy only the bounded cumulative envelope; skip a busy consumer."""
        if not self._preview_enabled or not self._waveform_lock.acquire(blocking=False):
            return None
        try:
            snapshots = tuple(self._waveforms.snapshots().values())
        except Exception as exc:
            # Presentation boundary: reducer/snapshot faults disable preview only.
            self._disable_preview(exc)
            return None
        finally:
            self._waveform_lock.release()
        return RecordingPreview(self.request.request_id, generation, sequence,
                                snapshots[0].sample_stop, self.request.channels, snapshots)

    def _disable_preview(self, exc):
        if self._preview_enabled:
            self._preview_enabled = False
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
            self._fail("capture", f"input overflow on device {self.request.device['index']}")
            return None
        if status and self._status_warning is None:
            self._status_warning = str(status)
        if (not isinstance(indata, np.ndarray) or indata.ndim != 2 or indata.shape[0] != frames
                or indata.shape[1] <= max(self.request.channels) or frames <= 0):
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
            owned = np.array(indata[:accepted, self.request.channels], dtype=np.float32, order="C", copy=True)
            self._blocks.append(owned)
            self._queued_frames += accepted
            self.raw_frames += accepted
            if self.raw_frames == self.request.target_samples:
                self._stop_requested.set()
        self._wake.set()
        return owned

    def _input_callback(self, indata, frames, time_info, status):
        self._dispatch_callback(indata, frames, status)

    def _monitor_callback(self, indata, outdata, frames, time_info, status):
        self._dispatch_callback(indata, frames, status, outdata)

    def _dispatch_callback(self, indata, frames, status, outdata=None):
        try:
            if outdata is None:
                self._accept(indata, frames, status)
            else:
                self._render_monitor(indata, outdata, frames, status)
        except Exception as exc:
            # PortAudio otherwise suppresses callback exceptions. Record once and
            # wake the owner to stop/close; never log or send IPC on this thread.
            self._fail("capture", f"audio callback failed: {exc}")

    def _render_monitor(self, indata, outdata, frames, status):
        outdata.fill(0)
        owned = self._accept(indata, frames, status)
        if owned is None:
            return
        mono = owned.mean(axis=1).astype(np.float32, copy=False)
        play = np.zeros(frames, dtype=np.float32)
        play[:len(mono)] = mono
        play = np.clip(play * self._monitor_gain, -1.0, 1.0).astype(np.float32, copy=False)
        monitor = self.request.monitor
        play = apply_monitor_startup_mute(play, mute_total=monitor.get("mute_leading_samples", 0),
                                          emitted_before=self._monitor_emitted,
                                          fade_len=monitor.get("fade_in_samples", 0))
        self._monitor_emitted += frames
        for channel in monitor["channels"]:
            outdata[:, channel] = play

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
        if self._backend is None:
            self._backend = sounddevice_backend()
        self._validate_device(req.device, req.channels, "input")
        monitor = req.monitor
        enabled = req.purpose == "main" and monitor.get("enabled", False)
        if enabled:
            self._validate_device(monitor["device"], monitor["channels"], "output")
        self._stage = "open_wav"
        self._writer = self._writer_factory(req.path, sample_rate=req.sample_rate, channels=len(req.channels))
        self._stage = "device"
        config = dict(samplerate=req.sample_rate, dtype="float32", blocksize=self._blocksize)
        if enabled:
            output = monitor["device"]
            out_count = max(monitor["channels"]) + 1
            if output["max_output_channels"] >= 2:
                out_count = max(out_count, 2)
            selector = req.device["index"] if req.device["index"] == output["index"] else (req.device["index"], output["index"])
            self._stream = self._backend.Stream(**config, channels=(max(req.channels) + 1, out_count),
                                                device=selector, callback=self._monitor_callback)
        else:
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
        if self._preview_enabled:
            with self._waveform_lock:
                try:
                    self._waveforms.append(block)
                except Exception as exc:
                    # A display reducer is not an audio-integrity dependency.
                    self._disable_preview(exc)

    def _close_stream(self):
        stream, self._stream = self._stream, None
        if stream is None:
            return
        for operation in (stream.stop, stream.close):
            try:
                operation()
            except Exception as exc:
                # Native backend cleanup can raise arbitrary backend exceptions;
                # still attempt close after stop fails, and forbid successful delivery.
                self._handles_released = False
                self._logger.exception("Stream cleanup failed for %s", self.request.request_id)
                self._fail("close_stream", str(exc))

    def _close_writer(self):
        writer, self._writer = self._writer, None
        if writer is None:
            return
        try:
            writer.finalize()
        except Exception as exc:
            # Writer boundary: normalize a failed close once; handle release is
            # unknown and the worker owner must retire the process before reuse.
            self._handles_released = False
            self._logger.exception("WAV close failed for %s", self.request.request_id)
            self._fail("close_wav", str(exc))

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
            self._close_writer()
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
            ok, reason, detail = validate_recorded_audio(audio, req.validation_thresholds.to_dict())
            if not ok:
                raise ValueError(f"{reason} {detail}")
            self._stage = "metadata"
            metadata = req.calibration_metadata.to_dict() if req.calibration_metadata is not None else None
            metadata_result = self._metadata_appender(req.path, metadata, logger=self._logger)
            if isinstance(metadata_result, WavCalibrationMetadataAppendResult):
                self._owned_temporary_paths.update(metadata_result.cleanup_paths)
                self._unreleased_finalization_handles.extend(metadata_result.retained_handles)
                if not metadata_result.handles_released:
                    self._handles_released = False
                    self._fail("metadata", "; ".join(metadata_result.close_errors))
                    return
                metadata_appended = metadata_result.appended
            else:
                # Compatibility for explicitly injected bool-only test appenders.
                metadata_appended = bool(metadata_result)
            if not metadata_appended:
                self._warnings.append("WAV calibration metadata was not appended")
            # Optional metadata failure is a warning only while audio remains readable.
            with self._finalization_file(req.path) as source:
                if len(source) != len(audio) or source.channels != len(req.channels) or source.subtype != "FLOAT":
                    raise ValueError("WAV audio became invalid during metadata finalization")
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
            self._close_finalization_handle(source, path, source.close)

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
